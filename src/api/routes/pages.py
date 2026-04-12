"""HTML page routes — serves full-page templates."""

from pathlib import Path

from fastapi import APIRouter, Cookie, Request, Response

from src.api.app import templates
from src.api.dependencies import (
    ensure_session_cookie,
    get_session,
    list_base_resumes,
)

router = APIRouter()


def _build_pipeline_context(session: dict) -> dict | None:
    """Build template context for pipeline_result partial from session.

    Args:
        session: Server-side session dict.

    Returns:
        Template context dict, or None if no result exists.
    """
    result = session.get("pipeline_result") or {}
    jd_req = result.get("jd_requirements") or {}
    if not jd_req:
        return None

    review = result.get("review_result") or {}
    change_summary = result.get("change_summary") or {}

    matched = [
        e for e in result.get("matched_experiences", [])
        if e.get("relevance_score", 0) > 0
    ]
    matched.sort(
        key=lambda e: e["relevance_score"], reverse=True,
    )

    saved_path = session.get("saved_resume_path", "")
    resume_fn = (
        Path(saved_path).name if saved_path else "resume.md"
    )

    return {
        "fit_score": jd_req.get("fit_score", 0),
        "fit_grade": jd_req.get("fit_grade", "?"),
        "keyword_coverage": review.get(
            "keyword_coverage", 0,
        ),
        "revision_count": result.get("revision_count", 0),
        "resume_content": session.get("current_resume", ""),
        "resume_filename": resume_fn,
        "saved_resume_path": saved_path,
        "change_summary": change_summary,
        "jd_requirements": jd_req,
        "matched_experiences": matched,
        "gaps": result.get("gaps", []),
        "review_feedback": review.get("feedback", ""),
        "star_stories": result.get("star_stories", []),
        "tailor_reasoning": result.get(
            "tailor_reasoning", "",
        ),
    }


@router.get("/")
def index_page(
    request: Request,
    response: Response,
    sid: str | None = Cookie(default=None),
) -> Response:
    """Render the Resume Tailoring page."""
    sid_val = ensure_session_cookie(sid, response)
    session = get_session(sid_val)

    ctx: dict = {
        "active_tab": "tailor",
        "resumes": list_base_resumes(),
        "pipeline_ctx": _build_pipeline_context(session),
    }
    return templates.TemplateResponse(
        request, "index.html", ctx,
    )


@router.get("/tracker")
def tracker_page(
    request: Request,
    response: Response,
    sid: str | None = Cookie(default=None),
) -> Response:
    """Render the Application Tracker page."""
    ensure_session_cookie(sid, response)
    return templates.TemplateResponse(
        request,
        "tracker.html",
        {"active_tab": "tracker"},
    )
