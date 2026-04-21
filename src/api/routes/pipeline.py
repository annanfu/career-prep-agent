"""Pipeline execution routes — run, poll status, get result, save, tex."""

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Cookie, Form, Request, Response

from src.api.app import templates
from src.api.dependencies import (
    BASE_RESUME_DIR,
    ensure_session_cookie,
    get_session,
    persist_session,
    task_manager,
)
from src.api.tasks import TaskStatus

router = APIRouter()


def _run_pipeline(
    jd_text: str,
    jd_url: str,
    base_resume_name: str,
    target_role: str,
    company_name: str,
    _task_state: Any = None,
) -> dict:
    """Execute the LangGraph pipeline with progress tracking.

    Args:
        jd_text: Raw job description text.
        jd_url: Optional JD URL string.
        base_resume_name: Filename of the selected base resume.
        target_role: Target job title.
        company_name: Target company.
        _task_state: TaskState for progress reporting.

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
        "persona_summary": "",
        "draft_content": "",
        "star_stories": [],
        "tailor_reasoning": "",
        "review_result": None,
        "revision_count": 0,
        "change_summary": {},
        "saved_resume_path": "",
        "saved_jd_path": "",
        "saved_stories_path": "",
        "tracker_updated": False,
        "final_output": "",
    }

    # Use stream() to track node progress in real time
    state = dict(initial)
    for event in app.stream(initial):
        node_name = list(event.keys())[0]
        state.update(event[node_name])
        if _task_state:
            _task_state.progress = node_name
    return state


@router.post("/run")
def run_pipeline(
    request: Request,
    response: Response,
    jd_text: str = Form(...),
    jd_url: str = Form(""),
    base_resume_name: str = Form(...),
    target_role: str = Form(""),
    company_name: str = Form(""),
    sid: str | None = Cookie(default=None),
) -> Response:
    """Submit the pipeline as a background task."""
    sid = ensure_session_cookie(sid, response)
    task_id = task_manager.submit(
        _run_pipeline,
        jd_text,
        jd_url,
        base_resume_name,
        target_role,
        company_name,
    )
    session = get_session(sid)
    session["active_task_id"] = task_id
    return templates.TemplateResponse(
        request,
        "partials/pipeline_status.html",
        {"task_id": task_id},
    )


@router.get("/status/{task_id}")
def pipeline_status(
    task_id: str,
    request: Request,
    response: Response,
    sid: str | None = Cookie(default=None),
) -> Response:
    """Poll endpoint — step indicator while running, result when done."""
    task = task_manager.get(task_id)
    if not task or task.status == TaskStatus.RUNNING:
        progress = task.progress if task else ""
        return templates.TemplateResponse(
            request,
            "partials/pipeline_status.html",
            {"task_id": task_id, "progress": progress},
        )

    if task.status == TaskStatus.FAILED:
        return templates.TemplateResponse(
            request,
            "partials/pipeline_error.html",
            {"error": task.error},
        )

    # Completed — save to session and render result
    from src.api.routes.pages import _build_pipeline_context

    result = task.result or {}
    sid_val = ensure_session_cookie(sid, response)
    session = get_session(sid_val)
    session["pipeline_result"] = result
    session["current_resume"] = result.get("final_output", "")
    session["saved_resume_path"] = result.get(
        "saved_resume_path", "",
    )
    session["saved_jd_path"] = result.get("saved_jd_path", "")
    session["base_resume_path"] = result.get(
        "base_resume_path", "",
    )
    session["chat_history"] = []

    # Persist to disk so session survives server restarts
    persist_session(session)

    task_manager.cleanup(task_id)

    ctx = _build_pipeline_context(session) or {}
    return templates.TemplateResponse(
        request, "partials/pipeline_result.html", ctx,
    )


@router.post("/save-resume")
def save_resume(
    resume_content: str = Form(...),
    saved_resume_path: str = Form(""),
    sid: str | None = Cookie(default=None),
) -> Response:
    """Save the edited resume content to disk."""
    if saved_resume_path:
        Path(saved_resume_path).write_text(
            resume_content, encoding="utf-8",
        )

    session = get_session(sid)
    session["current_resume"] = resume_content

    return Response(
        content=(
            '<p class="text-xs text-green-600">'
            "Saved successfully.</p>"
        ),
        media_type="text/html",
    )


@router.post("/mark-applied")
def mark_applied(
    response: Response,
    sid: str | None = Cookie(default=None),
) -> Response:
    """Add the current pipeline result to the tracker as 'applied'."""
    from datetime import date as dt_date

    from src.agents.save_and_track import (
        TRACKER_CSV,
        TRACKER_COLUMNS,
        _append_tracker,
    )

    sid_val = ensure_session_cookie(sid, response)
    session = get_session(sid_val)
    result = session.get("pipeline_result") or {}
    jd_req = result.get("jd_requirements") or {}

    if not jd_req:
        return Response(
            content=(
                '<p class="text-xs text-amber-600">'
                "No pipeline result to track.</p>"
            ),
            media_type="text/html",
        )

    # Check for duplicate (same company + role already in tracker)
    if TRACKER_CSV.exists():
        import pandas as pd_check
        df = pd_check.read_csv(TRACKER_CSV)
        company = (
            result.get("company_name")
            or jd_req.get("company", "")
        )
        role = (
            result.get("target_role")
            or jd_req.get("role", "")
        )
        dupes = df[
            (df["company"].str.lower() == company.lower())
            & (df["role"].str.lower() == role.lower())
        ]
        if not dupes.empty:
            return Response(
                content=(
                    '<p class="text-xs text-amber-600">'
                    "Already tracked.</p>"
                ),
                media_type="text/html",
            )
    else:
        company = (
            result.get("company_name")
            or jd_req.get("company", "")
        )
        role = (
            result.get("target_role")
            or jd_req.get("role", "")
        )

    saved_resume = session.get("saved_resume_path", "")
    saved_jd = session.get("saved_jd_path", "")

    tracker_row = {
        "company": company,
        "role": role,
        "fit_score": jd_req.get("fit_score", ""),
        "status": "applied",
        "notes": "",
        "resume_filename": (
            Path(saved_resume).name if saved_resume else ""
        ),
        "date": dt_date.today().isoformat(),
        "fit_grade": jd_req.get("fit_grade", ""),
        "jd_filename": (
            Path(saved_jd).name if saved_jd else ""
        ),
        "stories_filename": "",
        "jd_url": (
            jd_req.get("jd_url")
            or result.get("jd_url")
            or ""
        ),
    }
    _append_tracker(tracker_row)

    return Response(
        content=(
            '<div class="flex items-center gap-2">'
            '<span class="text-xs text-emerald-600">'
            "Added to tracker.</span>"
            '<span class="text-xs text-stone-400">'
            f"{company} — {role}</span></div>"
        ),
        media_type="text/html",
    )


_TEX_TEMPLATE_PATH = BASE_RESUME_DIR / "template.tex"
_TEX_PROMPT_PATH = Path("src/prompts/md_to_tex.txt")
_COVER_LETTER_PROMPT_PATH = Path("src/prompts/cover_letter.txt")


@router.post("/generate-tex")
def generate_tex(
    request: Request,
    response: Response,
    resume_content: str = Form(""),
    sid: str | None = Cookie(default=None),
) -> Response:
    """Convert current MD resume to LaTeX using the template."""
    from src.llm import get_quality_llm

    sid_val = ensure_session_cookie(sid, response)
    session = get_session(sid_val)

    md_resume = resume_content or session.get(
        "current_resume", "",
    )
    if not md_resume:
        return Response(
            content=(
                '<p class="text-sm text-amber-600">'
                "No resume to convert.</p>"
            ),
            media_type="text/html",
        )

    # Read template and prompt
    tex_template = _TEX_TEMPLATE_PATH.read_text(encoding="utf-8")
    prompt_template = _TEX_PROMPT_PATH.read_text(encoding="utf-8")

    # Get JD keywords so LLM knows what to bold
    result_data = session.get("pipeline_result") or {}
    jd_req = result_data.get("jd_requirements") or {}
    jd_skills = list(set(
        jd_req.get("required_skills", [])
        + jd_req.get("preferred_skills", [])
    ))
    # Also extract tech terms from raw JD to catch what parser missed
    jd_raw = result_data.get("jd_raw", "")
    if not jd_raw:
        # Try reading from saved JD file
        jd_path = session.get("saved_jd_path", "")
        if jd_path and Path(jd_path).exists():
            try:
                jd_archive = json.loads(
                    Path(jd_path).read_text(encoding="utf-8"),
                )
                jd_raw = jd_archive.get("jd_raw", "")
            except (json.JSONDecodeError, OSError):
                pass
    # Common tech terms to scan for in JD text
    _TECH_TERMS = [
        "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL",
        "Redis", "Kafka", "RabbitMQ", "Elasticsearch",
        "LangChain", "LangGraph", "LlamaIndex", "ChromaDB",
        "Hugging Face", "PyTorch", "scikit-learn", "TensorFlow",
        "React", "Node.js", "Express", "FastAPI", "Django",
        "Flask", "Spring Boot", "Kubernetes", "EKS", "Docker",
        "Terraform", "Ansible", "Jenkins", "Harness",
        "CI/CD", "Linux", "Unix", "Git", "GitHub",
        "AWS", "Azure", "GCP", "Google Cloud",
        "Agile", "TDD", "REST API", "microservices",
        "vector database", "knowledge graph",
    ]
    jd_lower = jd_raw.lower()
    for term in _TECH_TERMS:
        if term.lower() in jd_lower and term not in jd_skills:
            jd_skills.append(term)
    jd_keywords_str = ", ".join(sorted(set(jd_skills)))
    print(f"[generate-tex] JD keywords for bolding: {jd_keywords_str[:200]}")

    prompt = (
        prompt_template
        .replace("{resume_md}", md_resume)
        .replace("{jd_keywords}", jd_keywords_str)
    )

    # LLM call to convert MD → LaTeX sections
    llm = get_quality_llm(temperature=0)
    resp = llm.invoke(prompt)
    raw = (resp.content or "").strip()

    # Parse JSON response
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        sections = json.loads(raw)
    except json.JSONDecodeError:
        return Response(
            content=(
                '<p class="text-sm text-rose-600">'
                "Failed to parse LLM output. Try again.</p>"
            ),
            media_type="text/html",
        )

    # Fill template
    filled = tex_template
    filled = filled.replace(
        "%%EXPERIENCE%%",
        sections.get("experience", ""),
    )
    filled = filled.replace(
        "%%PROJECTS%%",
        sections.get("projects", ""),
    )
    filled = filled.replace(
        "%%EDUCATION%%",
        sections.get("education", ""),
    )
    filled = filled.replace(
        "%%SKILLS%%",
        sections.get("skills", ""),
    )

    # Save to output
    saved_path = session.get("saved_resume_path", "")
    if saved_path:
        tex_path = Path(saved_path).with_suffix(".tex")
    else:
        tex_path = Path("output/resumes/resume.tex")
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(filled, encoding="utf-8")

    fn = tex_path.name
    return Response(
        content=(
            '<div class="flex items-center gap-3">'
            '<p class="text-sm text-emerald-600">'
            "TeX generated successfully.</p>"
            f'<a href="/api/download/tex/{fn}" hx-boost="false" '
            f'download="{fn}" '
            'class="text-sm bg-stone-50 hover:bg-stone-100 '
            "text-stone-600 py-1.5 px-3 rounded-lg border "
            'border-stone-200 transition">'
            "Download .tex</a></div>"
        ),
        media_type="text/html",
    )


@router.post("/generate-cover-letter")
def generate_cover_letter(
    response: Response,
    resume_content: str = Form(""),
    sid: str | None = Cookie(default=None),
) -> Response:
    """Generate a 3-paragraph cover letter from resume + JD."""
    from src.llm import get_quality_llm

    sid_val = ensure_session_cookie(sid, response)
    session = get_session(sid_val)

    md_resume = resume_content or session.get(
        "current_resume", "",
    )
    if not md_resume:
        return Response(
            content=(
                '<p class="text-sm text-amber-600">'
                "Generate a tailored resume first.</p>"
            ),
            media_type="text/html",
        )

    result_data = session.get("pipeline_result") or {}
    jd_req = result_data.get("jd_requirements") or {}
    company = (
        result_data.get("company_name")
        or jd_req.get("company", "")
    )
    role = (
        result_data.get("target_role")
        or jd_req.get("role", "")
    )

    prompt_template = _COVER_LETTER_PROMPT_PATH.read_text(
        encoding="utf-8",
    )
    prompt = (
        prompt_template
        .replace("{target_role}", role)
        .replace("{company_name}", company)
        .replace(
            "{jd_requirements}",
            json.dumps(jd_req, indent=2),
        )
        .replace("{resume_content}", md_resume)
    )

    llm = get_quality_llm(temperature=0.4)
    resp = llm.invoke(prompt)
    letter = (resp.content or "").strip()

    # Strip markdown fences if present
    letter = re.sub(r"^```(?:\w+)?\s*", "", letter)
    letter = re.sub(r"\s*```$", "", letter)

    # Save to disk
    saved_path = session.get("saved_resume_path", "")
    if saved_path:
        cl_path = (
            Path(saved_path)
            .with_name(
                Path(saved_path).stem + "_cover_letter.txt",
            )
        )
    else:
        cl_path = Path("output/resumes/cover_letter.txt")
    cl_path.parent.mkdir(parents=True, exist_ok=True)
    cl_path.write_text(letter, encoding="utf-8")
    fn = cl_path.name

    # Return editable textarea + download link
    escaped = (
        letter.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return Response(
        content=(
            '<div class="bg-white border border-stone-200 '
            'rounded-xl p-5 mb-5">'
            '<div class="flex items-center justify-between '
            'mb-3">'
            '<h3 class="text-sm font-medium text-stone-700">'
            "Cover Letter</h3>"
            '<a href="/api/download/cover-letter/'
            f'{fn}" hx-boost="false" download="{fn}" '
            'class="text-sm bg-stone-50 hover:bg-stone-100 '
            "text-stone-600 py-1.5 px-3 rounded-lg border "
            'border-stone-200 transition">'
            "Download</a></div>"
            '<textarea rows="12" '
            'class="w-full font-mono text-sm">'
            f"{escaped}</textarea></div>"
        ),
        media_type="text/html",
    )
