"""HTML page routes — serves full-page templates."""

from fastapi import APIRouter, Cookie, Request, Response

from src.api.app import templates
from src.api.dependencies import (
    ensure_session_cookie,
    list_base_resumes,
)

router = APIRouter()


@router.get("/")
def index_page(
    request: Request,
    response: Response,
    sid: str | None = Cookie(default=None),
) -> Response:
    """Render the Resume Tailoring page."""
    ensure_session_cookie(sid, response)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "active_tab": "tailor",
            "resumes": list_base_resumes(),
        },
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
