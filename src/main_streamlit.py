"""Streamlit frontend for the Career Prep Agent."""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'src' is importable regardless
# of how Streamlit is launched.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from src.llm import get_quality_llm  # noqa: E402

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_RESUME_DIR = Path("base_resume")
OUTPUT_DIR = Path("output")
TRACKER_CSV = OUTPUT_DIR / "tracker.csv"
TRACKER_STATUS_OPTIONS = ["applied", "interview", "rejected", "offer"]

st.set_page_config(
    page_title="Career Prep Agent",
    page_icon="💼",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_resume" not in st.session_state:
    st.session_state.current_resume = ""
if "interview_prep_result" not in st.session_state:
    st.session_state.interview_prep_result = None
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "resume"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_base_resumes() -> list[str]:
    """Return available base resume filenames from base_resume/."""
    if not BASE_RESUME_DIR.exists():
        return []
    return [f.name for f in BASE_RESUME_DIR.glob("*.md")]


def _fit_score_color(score: float) -> str:
    """Return a CSS color string for the given fit score."""
    if score >= 3.5:
        return "green"
    if score >= 2.5:
        return "orange"
    return "red"


def _fit_label(score: float, grade: str) -> str:
    """Format a human-readable fit label."""
    if score >= 3.5:
        return f"{score}/5 ({grade}) — Good fit"
    return f"{score}/5 ({grade}) — Low fit, consider skipping"


def _run_pipeline(
    jd_text: str,
    jd_url: str,
    base_resume_name: str,
    target_role: str = "",
    company_name: str = "",
) -> dict:
    """Load resume, build initial state, run the LangGraph pipeline.

    Args:
        jd_text: Raw job description text.
        jd_url: Optional JD URL string.
        base_resume_name: Filename of the selected base resume.

    Returns:
        Final GraphState dict from the pipeline.
    """
    from src.graph import app
    from src.state import GraphState

    resume_path = BASE_RESUME_DIR / base_resume_name
    base_content = resume_path.read_text(encoding="utf-8")

    initial: GraphState = {
        "jd_raw": jd_text,
        "jd_url": jd_url or None,
        "target_role": target_role,
        "company_name": company_name,
        "base_resume_path": str(resume_path),
        "base_resume_content": base_content,
        "jd_requirements": None,
        "matched_experiences": [],
        "gaps": [],
        "draft_content": "",
        "star_stories": [],
        "review_result": None,
        "revision_count": 0,
        "change_summary": {},
        "saved_resume_path": "",
        "saved_jd_path": "",
        "saved_stories_path": "",
        "tracker_updated": False,
        "final_output": "",
    }
    return app.invoke(initial)


def _chat_refine(
    user_message: str,
    current_resume: str,
    jd_requirements: dict,
    history: list[dict],
) -> str:
    """Send a chat message to refine the resume. Returns updated resume text.

    Args:
        user_message: User's refinement instruction.
        current_resume: Current resume Markdown.
        jd_requirements: Parsed JD requirements dict.
        history: List of {role, content} dicts (prior turns).

    Returns:
        Updated resume Markdown string.
    """
    system = (
        "You are a resume editor. The user will give you instructions to "
        "modify the resume below. Apply the requested changes and return "
        "ONLY the complete updated resume in Markdown. Never fabricate "
        "experiences, skills, or metrics not already in the resume.\n\n"
        f"JOB REQUIREMENTS:\n{json.dumps(jd_requirements, indent=2)}\n\n"
        f"CURRENT RESUME:\n{current_resume}"
    )

    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append(turn)
    messages.append({"role": "user", "content": user_message})

    llm = get_quality_llm(temperature=0.3)
    response = llm.invoke(messages)
    return (response.content or "").strip()


def _chat_interview_prep(
    user_message: str,
    prep_doc: str,
    jd_data: dict,
    history: list[dict],
) -> str:
    """Chat follow-up for interview prep. Returns answer text.

    Args:
        user_message: User's follow-up question.
        prep_doc: Current interview prep document.
        jd_data: JD requirements dict.
        history: Prior chat turns.

    Returns:
        Answer string.
    """
    system = (
        "You are an interview coach. The user has an interview prep "
        "document and is asking follow-up questions. Answer using "
        "the prep document and JD context below. Use STARL format "
        "for behavioral answers (Situation, Task, Action, Result, "
        "Learning). Be specific and practical.\n\n"
        f"JD REQUIREMENTS:\n{json.dumps(jd_data, indent=2)[:2000]}\n\n"
        f"INTERVIEW PREP DOCUMENT:\n{prep_doc[:4000]}"
    )

    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append(turn)
    messages.append({"role": "user", "content": user_message})

    llm = get_quality_llm(temperature=0.3)
    response = llm.invoke(messages)
    return (response.content or "").strip()


def _ingest_documents() -> str:
    """Run the RAG ingest pipeline and return a summary string."""
    from src.rag.ingest import ingest_documents
    files, chunks = ingest_documents("knowledge_base")
    return f"Ingested {files} file(s) → {chunks} chunks stored in ChromaDB."


def _run_interview_prep(row: "pd.Series") -> None:
    """Run the Phase 2 interview prep pipeline for a tracker row.

    Loads the saved resume and JD, runs the interview prep graph,
    and displays results in an expandable section.

    Args:
        row: A pandas Series from the tracker DataFrame.
    """
    from src.graph_interview import interview_app
    from src.state import InterviewPrepState

    resume_path = Path(f"output/resumes/{row['resume_filename']}")
    jd_path = Path(f"output/jds/{row.get('jd_filename', '')}")

    if not resume_path.exists():
        st.error(f"Resume not found: {resume_path}")
        return

    resume_content = resume_path.read_text(encoding="utf-8")
    jd_data = {}
    if jd_path.exists():
        try:
            jd_data = json.loads(jd_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    company = str(row.get("company", "Unknown"))
    role = str(row.get("role", ""))

    initial: InterviewPrepState = {
        "resume_content": resume_content,
        "jd_data": jd_data,
        "company": company,
        "role": role,
        "deep_experiences": [],
        "bq_templates": {},
        "interview_notes": "",
        "company_brief": {},
        "interview_questions": [],
        "interview_prep_output": "",
        "saved_prep_path": "",
    }

    with st.spinner(
        f"Preparing interview for {company} — {role}..."
    ):
        try:
            result = interview_app.invoke(initial)
        except Exception as e:
            st.error(f"Interview prep error: {e}")
            return

    prep_doc = result.get("interview_prep_output", "")
    saved_path = result.get("saved_prep_path", "")

    st.session_state.interview_prep_result = {
        "prep_doc": prep_doc,
        "saved_path": saved_path,
        "jd_data": jd_data,
        "company": company,
        "role": role,
    }

    st.success(f"Interview prep ready! Saved to {saved_path}")
    st.markdown(prep_doc)
    st.download_button(
        "Download Prep Document",
        data=prep_doc,
        file_name=Path(saved_path).name if saved_path else "prep.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("💼 Career Prep Agent")

tab_tailor, tab_tracker = st.tabs(
    ["Resume Tailoring", "Application Tracker"]
)

# ============================================================
# TAB 1 — Resume Tailoring
# ============================================================
with tab_tailor:

    # --- Sidebar ---
    with st.sidebar:
        st.header("Knowledge Base")
        st.caption(
            "Add or edit files in `knowledge_base/`, "
            "then click Re-ingest."
        )
        if st.button("Re-ingest Documents", use_container_width=True):
            with st.spinner("Ingesting..."):
                msg = _ingest_documents()
            st.success(msg)

        # Show current ChromaDB status
        if Path("chroma_data").exists():
            with st.expander("ChromaDB Status"):
                try:
                    from langchain_chroma import Chroma as _Chroma
                    from langchain_community.embeddings import (
                        HuggingFaceEmbeddings as _HFE,
                    )
                    _emb = _HFE(
                        model_name="sentence-transformers/"
                        "all-MiniLM-L6-v2",
                    )
                    _vs = _Chroma(
                        collection_name="knowledge_base",
                        embedding_function=_emb,
                        persist_directory="chroma_data",
                    )
                    _col = _vs._collection
                    _metas = _col.get(include=["metadatas"])
                    _stats = {}
                    for _m in _metas["metadatas"]:
                        _src = _m.get("source_file", "unknown")
                        _path = _m.get("source_path", "")
                        if _src not in _stats:
                            _stats[_src] = {
                                "folder": _path.rsplit("/", 1)[0]
                                if "/" in _path else "(root)",
                                "chunks": 0,
                            }
                        _stats[_src]["chunks"] += 1
                    st.metric("Total Chunks", _col.count())
                    st.metric("Files", len(_stats))
                    rows = [
                        {
                            "File": s,
                            "Folder": i["folder"],
                            "Chunks": i["chunks"],
                        }
                        for s, i in sorted(
                            _stats.items(),
                            key=lambda x: x[1]["folder"],
                        )
                    ]
                    st.dataframe(
                        rows, use_container_width=True,
                        hide_index=True,
                    )
                except Exception as _e:
                    st.caption(f"Could not load: {_e}")

        st.divider()
        st.header("Resume Template")
        resume_options = _list_base_resumes()
        if resume_options:
            selected_resume = st.selectbox(
                "Base resume", resume_options
            )
        else:
            st.warning("No .md files in base_resume/")
            selected_resume = None

    # --- Main area ---
    target_role_input = st.text_input(
        "Target Role / Position Title",
        placeholder="e.g. Software Engineer, Fund Accountant, Data Analyst",
    )
    company_name_input = st.text_input(
        "Company Name",
        placeholder="e.g. Planet Labs, Orbis Investments, Stripe",
    )
    jd_url_input = st.text_input(
        "Job posting URL (optional)", placeholder="https://..."
    )
    jd_input = st.text_area(
        "Paste job description here",
        height=250,
        placeholder="Copy and paste the full job description...",
    )

    run_btn = st.button(
        "Generate Tailored Resume",
        type="primary",
        disabled=not selected_resume,
    )

    if run_btn:
        if not jd_input.strip():
            st.error("Please paste a job description before generating.")
        else:
            with st.spinner("Running pipeline (Node 1 → 2 → 3)..."):
                try:
                    result = _run_pipeline(
                        jd_input, jd_url_input, selected_resume,
                        target_role_input, company_name_input,
                    )
                    st.session_state.pipeline_result = result
                    st.session_state.current_resume = result.get(
                        "final_output", ""
                    )
                    st.session_state.chat_history = []
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

    # --- Results ---
    result = st.session_state.pipeline_result
    if result:
        jd_req = result.get("jd_requirements", {})
        review = result.get("review_result", {})
        fit_score = jd_req.get("fit_score", 0)
        fit_grade = jd_req.get("fit_grade", "?")
        color = _fit_score_color(fit_score)
        label = _fit_label(fit_score, fit_grade)

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Fit Score", f"{fit_score}/5 ({fit_grade})")
        col2.metric(
            "Keyword Coverage",
            f"{review.get('keyword_coverage', 0):.0%}",
        )
        col3.metric("Revisions", result.get("revision_count", 0))

        if fit_score < 3.5:
            st.warning(label)
        else:
            st.success(label)

        # Editable resume
        st.subheader("Tailored Resume")
        # Use a dynamic key so the widget resets when content changes
        resume_hash = hash(st.session_state.current_resume)
        edited = st.text_area(
            "Edit as needed",
            value=st.session_state.current_resume,
            height=500,
            key=f"resume_editor_{resume_hash}",
        )
        st.session_state.current_resume = edited

        col_dl, col_save = st.columns(2)
        col_dl.download_button(
            "Download .md",
            data=edited,
            file_name=Path(
                result.get("saved_resume_path", "resume.md")
            ).name,
            mime="text/markdown",
        )
        if col_save.button("Save Final Version"):
            saved_path = result.get("saved_resume_path")
            if saved_path:
                Path(saved_path).write_text(edited, encoding="utf-8")
                st.success(f"Saved to {saved_path}")

        # Pipeline details expander
        with st.expander("Pipeline Details"):
            cs = result.get("change_summary", {})
            if cs:
                st.subheader("Change Summary")
                cs_col1, cs_col2, cs_col3 = st.columns(3)
                cs_col1.metric(
                    "Keywords Added",
                    len(cs.get("keywords_added", [])),
                )
                cs_col2.metric(
                    "Keywords Missing",
                    len(cs.get("keywords_missing", [])),
                )
                cs_col3.metric(
                    "Keyword Coverage",
                    f"{cs.get('keyword_coverage', 0):.0%}",
                )
                if cs.get("keywords_added"):
                    st.markdown(
                        "**Keywords added:** "
                        + ", ".join(
                            f"`{k}`" for k in cs["keywords_added"]
                        )
                    )
                if cs.get("keywords_missing"):
                    st.markdown(
                        "**Still missing:** "
                        + ", ".join(
                            f"`{k}`" for k in cs["keywords_missing"]
                        )
                    )
                bm = cs.get("bullets_modified", {})
                added_bullets = bm.get("added", [])
                removed_bullets = bm.get("removed", [])
                if added_bullets or removed_bullets:
                    with st.expander(
                        f"Bullet changes"
                        f" (+{len(added_bullets)} / -{len(removed_bullets)})"
                    ):
                        for b in added_bullets:
                            st.markdown(
                                f'<span style="color:green">+ {b}</span>',
                                unsafe_allow_html=True,
                            )
                        for b in removed_bullets:
                            st.markdown(
                                f'<span style="color:red">- {b}</span>',
                                unsafe_allow_html=True,
                            )
                st.divider()

            st.subheader("JD Analysis")
            st.json(jd_req)

            st.subheader("Matched Experiences")
            relevant_exps = [
                e for e in result.get("matched_experiences", [])
                if e.get("relevance_score", 0) > 0
            ]
            if not relevant_exps:
                st.caption("No relevant matches found.")
            for exp in sorted(
                relevant_exps,
                key=lambda e: e["relevance_score"],
                reverse=True,
            ):
                score = exp["relevance_score"]
                badge = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
                with st.expander(
                    f"{badge} [{score:.2f}] {exp['requirement'] or '—'}"
                    f"  ·  {exp['source_doc']}"
                ):
                    st.caption(exp["evidence"][:400])

            if result.get("gaps"):
                st.subheader("Gaps")
                st.warning(", ".join(result["gaps"]))

            st.subheader("Review Feedback")
            st.write(review.get("feedback") or "No issues found.")

            st.subheader("STAR Stories")
            for story in result.get("star_stories", []):
                with st.expander(story.get("experience_name", "Story")):
                    st.write(f"**Requirement:** {story.get('jd_requirement')}")
                    st.write(f"**S:** {story.get('situation')}")
                    st.write(f"**T:** {story.get('task')}")
                    st.write(f"**A:** {story.get('action')}")
                    st.write(f"**R:** {story.get('result')}")

# ============================================================
# TAB 2 — Application Tracker
# ============================================================
with tab_tracker:
    st.header("Application Tracker")

    if not TRACKER_CSV.exists():
        st.info(
            "No applications tracked yet. "
            "Generate a tailored resume to add your first entry."
        )
    else:
        df = pd.read_csv(TRACKER_CSV)
        for col in ["notes", "jd_url", "status"]:
            if col in df.columns:
                df[col] = df[col].fillna("").astype(str)

        st.caption(
            "Edit cells inline. Use the trash icon on each row to delete."
            " Click Save to persist."
        )
        edited_df = st.data_editor(
            df,
            column_config={
                "status": st.column_config.SelectboxColumn(
                    "Status",
                    options=TRACKER_STATUS_OPTIONS,
                ),
                "notes": st.column_config.TextColumn("Notes"),
                "jd_url": st.column_config.LinkColumn("JD URL"),
            },
            use_container_width=True,
            num_rows="dynamic",
            key="tracker_editor",
        )

        col_save, _ = st.columns([1, 4])
        if col_save.button("Save Changes"):
            edited_df.to_csv(TRACKER_CSV, index=False)
            st.success("Tracker saved.")
            st.rerun()

        st.divider()
        st.subheader("Interview Prep")
        interview_rows = edited_df[
            edited_df["status"] == "interview"
        ]
        if interview_rows.empty:
            st.info(
                "No applications at 'interview' status yet. "
                "Update a status above to unlock interview prep."
            )
        else:
            for _, row in interview_rows.iterrows():
                col_info, col_btn = st.columns([3, 1])
                col_info.write(
                    f"**{row['company']}** — {row['role']}"
                )
                prep_key = f"prep_{row['resume_filename']}"
                if col_btn.button("Prepare Interview", key=prep_key):
                    _run_interview_prep(row)

# ============================================================
# GLOBAL CHATBOT — below both tabs
# ============================================================
st.divider()
st.subheader("Chat Assistant")

# Mode selector
has_resume = bool(
    st.session_state.current_resume
    or st.session_state.pipeline_result
)
has_prep = st.session_state.interview_prep_result is not None

mode_options = []
if has_resume:
    mode_options.append("Refine Resume")
if has_prep:
    mode_options.append("Interview Prep")

# Restore current_resume from pipeline result if needed
if not st.session_state.current_resume and st.session_state.pipeline_result:
    st.session_state.current_resume = (
        st.session_state.pipeline_result.get("final_output", "")
    )

if not mode_options:
    st.caption(
        "Generate a tailored resume or run interview prep "
        "to enable the chat assistant."
    )
else:
    chat_mode = st.radio(
        "Chat mode", mode_options, horizontal=True,
        key="chat_mode_radio",
    )

    if chat_mode == "Refine Resume":
        st.caption(
            "Ask me to adjust the resume: "
            "'Make the second bullet more concise', "
            "'Add Kubernetes to skills', etc."
        )
    else:
        st.caption(
            "Ask follow-up interview questions: "
            "'Give me a STARL story for teamwork', "
            "'How should I answer why this company?', etc."
        )

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_msg = st.chat_input("Ask me anything...")
    if user_msg:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_msg}
        )
        with st.chat_message("user"):
            st.write(user_msg)

        with st.spinner("Thinking..."):
            try:
                if chat_mode == "Refine Resume":
                    result = st.session_state.pipeline_result or {}
                    jd_req = result.get("jd_requirements", {})
                    updated = _chat_refine(
                        user_msg,
                        st.session_state.current_resume,
                        jd_req,
                        st.session_state.chat_history[:-1],
                    )
                    st.session_state.current_resume = updated
                    reply = "Resume updated. Check the Tailored Resume above."
                else:
                    prep = st.session_state.interview_prep_result
                    reply = _chat_interview_prep(
                        user_msg,
                        prep.get("prep_doc", ""),
                        prep.get("jd_data", {}),
                        st.session_state.chat_history[:-1],
                    )

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": reply}
                )
                st.rerun()
            except Exception as e:
                st.error(f"Chat error: {e}")
