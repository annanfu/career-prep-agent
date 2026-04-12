"""Chat assistant routes — resume refinement and interview prep Q&A."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Cookie, Form, Request, Response

from src.api.dependencies import (
    ensure_session_cookie,
    get_session,
    persist_session,
)
from src.llm import get_quality_llm

router = APIRouter()


def _chat_refine(
    user_message: str,
    current_resume: str,
    jd_requirements: dict,
    history: list[dict],
    master_resume: str = "",
    rag_chunks: list[dict] | None = None,
) -> str:
    """Refine the resume via LLM. Returns complete updated resume.

    Args:
        user_message: User's refinement instruction.
        current_resume: Current resume Markdown.
        jd_requirements: Parsed JD requirements dict.
        history: List of {role, content} dicts.
        master_resume: Full master resume for restoring deleted content.
        rag_chunks: Retrieved knowledge base chunks for deep research mode.

    Returns:
        Updated resume Markdown string.
    """
    master_section = ""
    if master_resume:
        master_section = (
            "\n\nMASTER RESUME (original, unfiltered — use this as the "
            "source of truth to restore any experiences, projects, or "
            "bullets the user asks to add back; never fabricate anything "
            "not present here):\n"
            f"{master_resume}"
        )

    rag_section = ""
    if rag_chunks:
        chunk_texts = "\n\n---\n\n".join(
            f"[Source: {c.get('source_doc', 'unknown')}]\n"
            f"{c.get('content', '')}"
            for c in rag_chunks
        )
        rag_section = (
            "\n\nKNOWLEDGE BASE EXCERPTS (retrieved based on your "
            "request — use specific facts, metrics, and details from "
            "these to enrich resume bullets; never copy verbatim, "
            "always integrate naturally):\n\n"
            f"{chunk_texts}"
        )

    system = (
        "You are a resume editor. The user will give you instructions to "
        "modify the resume below. Apply the requested changes and return "
        "ONLY the complete updated resume in Markdown. Never fabricate "
        "experiences, skills, or metrics not present in the resume, "
        "master resume, or knowledge base excerpts."
        f"{master_section}"
        f"{rag_section}\n\n"
        f"JOB REQUIREMENTS:\n{json.dumps(jd_requirements, indent=2)}\n\n"
        f"CURRENT RESUME:\n{current_resume}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    llm = get_quality_llm(temperature=0.3)
    resp = llm.invoke(messages)
    return (resp.content or "").strip()


def _summarize_changes(
    user_message: str,
    old_resume: str,
    new_resume: str,
) -> str:
    """Quick LLM call to describe what changed.

    Args:
        user_message: The user's original request.
        old_resume: Resume before edit.
        new_resume: Resume after edit.

    Returns:
        Brief description of changes.
    """
    if old_resume == new_resume:
        return "No changes were needed based on your request."

    llm = get_quality_llm(temperature=0)
    resp = llm.invoke(
        f"The user asked: \"{user_message}\"\n\n"
        f"Here is the OLD resume (first 2000 chars):\n"
        f"{old_resume[:2000]}\n\n"
        f"Here is the NEW resume (first 2000 chars):\n"
        f"{new_resume[:2000]}\n\n"
        "In 2-3 concise sentences, describe exactly what "
        "was changed. Be specific about which bullet points "
        "or sections were modified."
    )
    return (resp.content or "Changes applied.").strip()


def _chat_edit_prep(
    user_message: str,
    prep_doc: str,
    jd_data: dict,
    history: list[dict],
    rag_chunks: list[dict] | None = None,
) -> str:
    """Edit the interview prep document based on user instruction.

    Args:
        user_message: User's edit instruction.
        prep_doc: Current interview prep document.
        jd_data: JD requirements dict.
        history: Prior chat turns.
        rag_chunks: Retrieved knowledge base chunks for deep research mode.

    Returns:
        Complete updated prep document.
    """
    rag_section = ""
    if rag_chunks:
        chunk_texts = "\n\n---\n\n".join(
            f"[Source: {c.get('source_doc', 'unknown')}]\n"
            f"{c.get('content', '')}"
            for c in rag_chunks
        )
        rag_section = (
            "\n\nKNOWLEDGE BASE EXCERPTS (use specific facts, metrics, "
            "and details from these to deepen STARL answers; never copy "
            "verbatim, always integrate naturally):\n\n"
            f"{chunk_texts}\n\n"
        )

    system = (
        "You are an interview prep editor. The user will give you "
        "instructions to modify the interview prep document below. "
        "Apply the requested changes and return ONLY the complete "
        "updated document. Keep the same format and structure. "
        "Use STARL format for behavioral answers. Be specific "
        "and practical. Never fabricate experiences."
        f"{rag_section}\n\n"
        f"JD REQUIREMENTS:\n{json.dumps(jd_data, indent=2)[:2000]}\n\n"
        f"CURRENT INTERVIEW PREP DOCUMENT:\n{prep_doc}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    llm = get_quality_llm(temperature=0.3)
    resp = llm.invoke(messages)
    return (resp.content or "").strip()


@router.post("/send")
def send_chat(
    request: Request,
    response: Response,
    message: str = Form(...),
    mode: str = Form("resume"),
    deep_search: str = Form("false"),
    sid: str | None = Cookie(default=None),
) -> Response:
    """Process a chat message and return the response as HTML."""
    sid_val = ensure_session_cookie(sid, response)
    session = get_session(sid_val)

    history = session.get("chat_history", [])
    result_data = session.get("pipeline_result") or {}
    resume_update_script = ""

    try:
        if mode == "resume":
            current_resume = session.get("current_resume", "")
            if not current_resume:
                display_reply = (
                    "No resume loaded yet. Please run the "
                    "pipeline first to generate a tailored resume."
                )
            else:
                jd_req = result_data.get("jd_requirements", {})
                # Load master resume for reference (restore deleted content)
                master_resume = ""
                base_path = session.get("base_resume_path", "")
                if base_path and Path(base_path).exists():
                    master_resume = Path(
                        base_path,
                    ).read_text(encoding="utf-8")
                # Deep Research: retrieve relevant knowledge base chunks
                rag_chunks: list[dict] = []
                if deep_search == "true":
                    try:
                        from src.rag.retriever import retrieve_experiences
                        rag_chunks = retrieve_experiences(
                            [message], top_k=3,
                        )
                    except Exception:
                        pass  # Silently skip if ChromaDB unavailable
                updated = _chat_refine(
                    message, current_resume, jd_req,
                    history, master_resume, rag_chunks,
                )
                if updated and updated != current_resume:
                    session["current_resume"] = updated
                    # Write updated resume to disk too
                    rpath = session.get("saved_resume_path")
                    if rpath:
                        Path(rpath).write_text(
                            updated, encoding="utf-8",
                        )
                    persist_session(session)
                    display_reply = _summarize_changes(
                        message, current_resume, updated,
                    )
                    # Hidden textarea for JS to pick up
                    resume_update_script = (
                        '<textarea id="resume-update-payload"'
                        ' style="display:none">'
                        f"{_escape(updated)}"
                        "</textarea>"
                    )
                else:
                    display_reply = (
                        "No changes were needed based on "
                        "your request."
                    )
        else:
            prep = session.get("interview_prep_result") or {}
            if not prep.get("prep_doc"):
                display_reply = (
                    "No interview prep loaded yet. Click "
                    "View or Generate on the tracker first."
                )
            else:
                old_prep = prep["prep_doc"]
                # Deep Research: retrieve relevant knowledge base chunks
                rag_chunks_prep: list[dict] = []
                if deep_search == "true":
                    try:
                        from src.rag.retriever import retrieve_experiences
                        rag_chunks_prep = retrieve_experiences(
                            [message], top_k=3,
                        )
                    except Exception:
                        pass  # Silently skip if ChromaDB unavailable
                updated_prep = _chat_edit_prep(
                    message,
                    old_prep,
                    prep.get("jd_data", {}),
                    history,
                    rag_chunks_prep,
                )
                if updated_prep and updated_prep != old_prep:
                    prep["prep_doc"] = updated_prep
                    session["interview_prep_result"] = prep
                    # Save to disk
                    spath = prep.get("saved_path", "")
                    if spath:
                        Path(spath).write_text(
                            updated_prep, encoding="utf-8",
                        )
                    persist_session(session)
                    display_reply = _summarize_changes(
                        message, old_prep, updated_prep,
                    )
                    # Update the prep display panel
                    resume_update_script = (
                        '<textarea id="prep-update-payload"'
                        ' style="display:none">'
                        f"{_escape(updated_prep)}"
                        "</textarea>"
                    )
                else:
                    display_reply = (
                        "No changes were needed based on "
                        "your request."
                    )

        history.append({"role": "user", "content": message})
        history.append(
            {"role": "assistant", "content": display_reply},
        )
        session["chat_history"] = history

    except Exception as e:
        display_reply = f"Error: {e}"

    return Response(
        content=(
            '<div class="flex justify-end">'
            '<div class="bg-teal-600 text-white rounded-xl '
            'px-3 py-2 max-w-[80%] text-sm">'
            f"{_escape(message)}</div></div>"
            '<div class="flex justify-start">'
            '<div class="bg-stone-100 text-stone-700 rounded-xl '
            'px-3 py-2 max-w-[80%] text-sm whitespace-pre-wrap">'
            f"{_escape(display_reply)}</div></div>"
            f"{resume_update_script}"
        ),
        media_type="text/html",
    )


@router.post("/reset")
def reset_chat(
    response: Response,
    sid: str | None = Cookie(default=None),
) -> Response:
    """Clear chat history."""
    sid_val = ensure_session_cookie(sid, response)
    session = get_session(sid_val)
    session["chat_history"] = []
    return Response(
        content=(
            '<p class="text-gray-400 text-xs text-center '
            'mt-8">Chat cleared.</p>'
        ),
        media_type="text/html",
    )


def _escape(text: str) -> str:
    """Basic HTML escaping for display content."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
