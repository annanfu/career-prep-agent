"""Pipeline execution routes — run, poll status, get result, save."""

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
    result = task.result or {}
    sid_val = ensure_session_cookie(sid, response)
    session = get_session(sid_val)
    session["pipeline_result"] = result
    session["current_resume"] = result.get("final_output", "")
    session["saved_resume_path"] = result.get(
        "saved_resume_path", "",
    )
    session["saved_jd_path"] = result.get("saved_jd_path", "")
    session["chat_history"] = []

    # Persist to disk so session survives server restarts
    persist_session(session)

    jd_req = result.get("jd_requirements") or {}
    review = result.get("review_result") or {}
    change_summary = result.get("change_summary") or {}

    matched = [
        e for e in result.get("matched_experiences", [])
        if e.get("relevance_score", 0) > 0
    ]
    matched.sort(
        key=lambda e: e["relevance_score"], reverse=True,
    )

    saved_path = result.get("saved_resume_path", "")
    resume_fn = Path(saved_path).name if saved_path else "resume.md"

    task_manager.cleanup(task_id)

    return templates.TemplateResponse(
        request,
        "partials/pipeline_result.html",
        {
            "fit_score": jd_req.get("fit_score", 0),
            "fit_grade": jd_req.get("fit_grade", "?"),
            "keyword_coverage": review.get(
                "keyword_coverage", 0,
            ),
            "revision_count": result.get("revision_count", 0),
            "resume_content": result.get("final_output", ""),
            "resume_filename": resume_fn,
            "saved_resume_path": saved_path,
            "change_summary": change_summary,
            "jd_requirements": jd_req,
            "matched_experiences": matched,
            "gaps": result.get("gaps", []),
            "review_feedback": review.get("feedback", ""),
            "star_stories": result.get("star_stories", []),
        },
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
