"""Application Tracker routes — CRUD + interview prep."""

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Cookie, Form, Request, Response

from src.api.app import templates
from src.api.dependencies import (
    TRACKER_CSV,
    TRACKER_STATUS_OPTIONS,
    ensure_session_cookie,
    get_session,
    persist_session,
    task_manager,
)
from src.api.tasks import TaskStatus

router = APIRouter()


def _load_tracker() -> list[dict]:
    """Load tracker.csv into a list of dicts."""
    if not TRACKER_CSV.exists():
        return []
    df = pd.read_csv(TRACKER_CSV)
    for col in df.columns:
        df[col] = df[col].fillna("").astype(str)
    return df.to_dict(orient="records")


@router.get("/data")
def tracker_data(request: Request) -> Response:
    """Return the tracker table partial with prep status."""
    import re

    rows = _load_tracker()
    prep_dir = Path("output/interview_prep")
    for row in rows:
        company = str(row.get("company", ""))
        role = str(row.get("role", ""))
        safe_co = re.sub(
            r"[^\w\s-]", "", company,
        ).strip().replace(" ", "_")
        safe_ro = re.sub(
            r"[^\w\s-]", "", role,
        ).strip().replace(" ", "_")
        fn = f"{safe_co}_{safe_ro}_prep.md"
        row["has_prep"] = (prep_dir / fn).exists()

    # Compute summary stats
    total = len(rows)
    status_counts = {}
    for r in rows:
        s = r.get("status", "")
        status_counts[s] = status_counts.get(s, 0) + 1

    return templates.TemplateResponse(
        request,
        "partials/tracker_table.html",
        {
            "rows": rows,
            "status_options": TRACKER_STATUS_OPTIONS,
            "total": total,
            "status_counts": status_counts,
        },
    )


@router.post("/save")
def tracker_save() -> Response:
    """Placeholder — use /save-json instead."""
    return Response(
        content=(
            '<p class="text-xs text-gray-500">'
            "Use the Save button (JS handler).</p>"
        ),
        media_type="text/html",
    )


@router.post("/save-json")
async def tracker_save_json(request: Request) -> Response:
    """Save tracker data from JSON body."""
    body = await request.json()
    rows = body.get("rows", [])
    if rows:
        df = pd.DataFrame(rows)
        TRACKER_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(TRACKER_CSV, index=False)
    return Response(
        content=(
            '<p class="text-xs text-green-600">'
            "Tracker saved.</p>"
        ),
        media_type="text/html",
    )


@router.get("/interview-rows")
def interview_rows(request: Request) -> Response:
    """Return the interview prep section partial."""
    import re

    all_rows = _load_tracker()
    interviews = [
        r for r in all_rows if r.get("status") == "interview"
    ]

    prep_dir = Path("output/interview_prep")
    for row in interviews:
        company = str(row.get("company", ""))
        role = str(row.get("role", ""))
        safe_company = re.sub(
            r"[^\w\s-]", "", company,
        ).strip().replace(" ", "_")
        safe_role = re.sub(
            r"[^\w\s-]", "", role,
        ).strip().replace(" ", "_")
        fn = f"{safe_company}_{safe_role}_prep.md"
        prep_path = prep_dir / fn
        if prep_path.exists():
            row["prep_filename"] = fn
            row["prep_content"] = prep_path.read_text(
                encoding="utf-8",
            )
        else:
            row["prep_filename"] = ""
            row["prep_content"] = ""

    return templates.TemplateResponse(
        request,
        "partials/interview_section.html",
        {"interview_rows": interviews},
    )


@router.get("/view-prep")
def view_prep(
    company: str,
    role: str,
    request: Request,
    response: Response,
    sid: str | None = Cookie(default=None),
) -> Response:
    """Load existing prep from disk and return as HTML."""
    import re

    safe_co = re.sub(
        r"[^\w\s-]", "", company,
    ).strip().replace(" ", "_")
    safe_ro = re.sub(
        r"[^\w\s-]", "", role,
    ).strip().replace(" ", "_")
    fn = f"{safe_co}_{safe_ro}_prep.md"
    prep_path = Path("output/interview_prep") / fn

    if not prep_path.exists():
        return Response(
            content=(
                '<p class="text-sm text-gray-400 p-4">'
                "No prep found. Click Generate first.</p>"
            ),
            media_type="text/html",
        )

    prep_doc = prep_path.read_text(encoding="utf-8")

    # Update session so chat knows the current prep
    sid_val = ensure_session_cookie(sid, response)
    session = get_session(sid_val)

    jd_data: dict = {}
    # Try to load JD data from tracker
    rows = _load_tracker()
    for r in rows:
        if (str(r.get("company", "")) == company
                and str(r.get("role", "")) == role):
            jd_fn = r.get("jd_filename", "")
            jd_path = Path(f"output/jds/{jd_fn}")
            if jd_fn and jd_path.exists():
                try:
                    jd_data = json.loads(
                        jd_path.read_text(encoding="utf-8"),
                    )
                except json.JSONDecodeError:
                    pass
            break

    session["interview_prep_result"] = {
        "prep_doc": prep_doc,
        "saved_path": str(prep_path),
        "jd_data": jd_data,
        "company": company,
        "role": role,
    }
    persist_session(session)

    import re
    safe_co = re.sub(
        r"[^\w\s-]", "", company,
    ).strip().replace(" ", "_")
    safe_ro = re.sub(
        r"[^\w\s-]", "", role,
    ).strip().replace(" ", "_")
    fn = f"{safe_co}_{safe_ro}_prep.md"

    return Response(
        content=(
            '<div class="prose prose-sm max-w-none text-sm '
            'max-h-[600px] overflow-y-auto">'
            f'<pre class="whitespace-pre-wrap font-sans">'
            f'{_escape(prep_doc)}</pre></div>'
            f'<script>onPrepLoaded("{_escape_js(company)}",'
            f'"{_escape_js(role)}","{_escape_js(fn)}")</script>'
        ),
        media_type="text/html",
    )


def _run_interview_prep(
    resume_filename: str,
    jd_filename: str,
    company: str,
    role: str,
    _task_state: object = None,
) -> dict:
    """Run Phase 2 interview prep pipeline (background thread).

    Args:
        resume_filename: Filename in output/resumes/.
        jd_filename: Filename in output/jds/.
        company: Company name.
        role: Role title.

    Returns:
        Final InterviewPrepState dict.
    """
    from src.graph_interview import interview_app
    from src.state import InterviewPrepState

    resume_path = Path(f"output/resumes/{resume_filename}")
    jd_path = Path(f"output/jds/{jd_filename}")

    resume_content = ""
    if resume_path.exists():
        resume_content = resume_path.read_text(encoding="utf-8")

    jd_data: dict = {}
    if jd_path.exists():
        try:
            jd_data = json.loads(
                jd_path.read_text(encoding="utf-8"),
            )
        except json.JSONDecodeError:
            pass

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
    # Use stream() to track node progress in real time
    state = dict(initial)
    for event in interview_app.stream(initial):
        node_name = list(event.keys())[0]
        state.update(event[node_name])
        if _task_state and hasattr(_task_state, "progress"):
            _task_state.progress = node_name
    return state


@router.post("/interview-prep")
def start_interview_prep(
    request: Request,
    response: Response,
    resume_filename: str = Form(...),
    jd_filename: str = Form(""),
    company: str = Form(""),
    role: str = Form(""),
    sid: str | None = Cookie(default=None),
) -> Response:
    """Start interview prep as a background task."""
    ensure_session_cookie(sid, response)
    task_id = task_manager.submit(
        _run_interview_prep,
        resume_filename,
        jd_filename,
        company,
        role,
    )
    poll_url = f"/api/tracker/interview-status/{task_id}"
    return templates.TemplateResponse(
        request,
        "partials/interview_status.html",
        {
            "task_id": task_id,
            "poll_url": poll_url,
            "progress": "",
        },
    )


@router.get("/interview-status/{task_id}")
def interview_status(
    task_id: str,
    request: Request,
    response: Response,
    sid: str | None = Cookie(default=None),
) -> Response:
    """Poll interview prep task status."""
    task = task_manager.get(task_id)
    poll_url = f"/api/tracker/interview-status/{task_id}"
    if not task or task.status == TaskStatus.RUNNING:
        progress = task.progress if task else ""
        return templates.TemplateResponse(
            request,
            "partials/interview_status.html",
            {
                "task_id": task_id,
                "poll_url": poll_url,
                "progress": progress,
            },
        )

    if task.status == TaskStatus.FAILED:
        return templates.TemplateResponse(
            request,
            "partials/pipeline_error.html",
            {"error": task.error},
        )

    # Completed
    result = task.result or {}
    prep_doc = result.get("interview_prep_output", "")
    saved_path = result.get("saved_prep_path", "")

    sid_val = ensure_session_cookie(sid, response)
    session = get_session(sid_val)
    session["interview_prep_result"] = {
        "prep_doc": prep_doc,
        "saved_path": saved_path,
        "jd_data": result.get("jd_data", {}),
        "company": result.get("company", ""),
        "role": result.get("role", ""),
    }
    persist_session(session)

    task_manager.cleanup(task_id)
    fn = Path(saved_path).name if saved_path else "prep.md"
    company = result.get("company", "")
    role = result.get("role", "")

    return Response(
        content=(
            '<div class="prose prose-sm max-w-none text-sm '
            'max-h-[600px] overflow-y-auto">'
            '<pre class="whitespace-pre-wrap font-sans">'
            f'{_escape(prep_doc)}</pre></div>'
            f'<script>onPrepLoaded("{_escape_js(company)}",'
            f'"{_escape_js(role)}","{_escape_js(fn)}")</script>'
        ),
        media_type="text/html",
    )


def _escape(text: str) -> str:
    """Basic HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _escape_js(text: str) -> str:
    """Escape for use in JS string literals."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("'", "\\'")
        .replace("\n", "\\n")
    )
