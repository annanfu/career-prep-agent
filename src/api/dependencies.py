"""Shared FastAPI dependencies: session store, task manager, helpers."""

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import Cookie, Response

from src.api.tasks import TaskManager

# ---------------------------------------------------------------------------
# Singletons (created once, shared across all requests)
# ---------------------------------------------------------------------------
task_manager = TaskManager(max_workers=2)

BASE_RESUME_DIR = Path("base_resume")
OUTPUT_DIR = Path("output")
TRACKER_CSV = OUTPUT_DIR / "tracker.csv"
TRACKER_STATUS_OPTIONS = ["applied", "interview", "rejected", "offer"]

# ---------------------------------------------------------------------------
# Server-side session store (replaces st.session_state)
# ---------------------------------------------------------------------------
_sessions: dict[str, dict[str, Any]] = {}

_SESSION_DEFAULTS: dict[str, Any] = {
    "pipeline_result": None,
    "current_resume": "",
    "saved_resume_path": "",
    "saved_jd_path": "",
    "chat_history": [],
    "chat_mode": "resume",
    "interview_prep_result": None,
}

# File-based persistence so sessions survive server restarts
_SESSION_PERSIST_FILE = OUTPUT_DIR / ".session.json"


def get_session(
    session_id: str | None = Cookie(
        default=None, alias="sid",
    ),
) -> dict[str, Any]:
    """Return (or create) the server-side session dict.

    Args:
        session_id: Session cookie value.

    Returns:
        Mutable session dict.
    """
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    return _SESSION_DEFAULTS.copy()


def ensure_session_cookie(
    session_id: str | None,
    response: Response,
) -> str:
    """Ensure a session cookie is set, creating one if needed.

    If the in-memory session was lost (e.g. server restart),
    automatically restores from disk.

    Args:
        session_id: Existing cookie value or None.
        response: FastAPI response to set cookie on.

    Returns:
        The (possibly new) session id.
    """
    if session_id and session_id in _sessions:
        return session_id

    # Try to restore from disk before creating a blank session
    new_id = session_id or str(uuid.uuid4())
    restored = _restore_from_disk()
    _sessions[new_id] = restored
    response.set_cookie(
        "sid", new_id, httponly=True, samesite="lax",
    )
    return new_id


def persist_session(session: dict[str, Any]) -> None:
    """Persist essential session data to disk.

    Stores file paths (not large content) so the session
    can be restored after a server restart.

    Args:
        session: The session dict to persist.
    """
    # Store paths + small data, not full content
    prep = session.get("interview_prep_result") or {}
    data = {
        "saved_resume_path": session.get(
            "saved_resume_path", "",
        ),
        "saved_jd_path": session.get("saved_jd_path", ""),
        "base_resume_path": session.get("base_resume_path", ""),
        "chat_history": session.get("chat_history", []),
        "interview_prep_saved_path": prep.get(
            "saved_path", "",
        ),
        "interview_prep_jd_data": prep.get("jd_data", {}),
        "interview_prep_company": prep.get("company", ""),
        "interview_prep_role": prep.get("role", ""),
    }
    _SESSION_PERSIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SESSION_PERSIST_FILE.write_text(
        json.dumps(data, default=str, ensure_ascii=False),
    )


def _restore_from_disk() -> dict[str, Any]:
    """Restore session from disk, re-reading files as needed.

    Returns:
        A populated session dict, or defaults if nothing saved.
    """
    session = _SESSION_DEFAULTS.copy()

    if not _SESSION_PERSIST_FILE.exists():
        return session

    try:
        data = json.loads(_SESSION_PERSIST_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return session

    # Restore resume content from saved file
    resume_path = data.get("saved_resume_path", "")
    if resume_path and Path(resume_path).exists():
        session["current_resume"] = Path(resume_path).read_text(
            encoding="utf-8",
        )
        session["saved_resume_path"] = resume_path

    # Restore JD requirements from saved file
    jd_path = data.get("saved_jd_path", "")
    if jd_path and Path(jd_path).exists():
        try:
            jd_archive = json.loads(
                Path(jd_path).read_text(encoding="utf-8"),
            )
            session["pipeline_result"] = {
                "jd_requirements": jd_archive.get(
                    "parsed", jd_archive,
                ),
            }
            session["saved_jd_path"] = jd_path
        except (json.JSONDecodeError, OSError):
            pass

    # Restore base resume path (always readable from disk — source file)
    base_resume_path = data.get("base_resume_path", "")
    if base_resume_path:
        session["base_resume_path"] = base_resume_path

    # Restore chat history
    session["chat_history"] = data.get("chat_history", [])

    # Restore interview prep from saved file
    prep_path = data.get("interview_prep_saved_path", "")
    if prep_path and Path(prep_path).exists():
        prep_doc = Path(prep_path).read_text(encoding="utf-8")
        session["interview_prep_result"] = {
            "prep_doc": prep_doc,
            "saved_path": prep_path,
            "jd_data": data.get(
                "interview_prep_jd_data", {},
            ),
            "company": data.get(
                "interview_prep_company", "",
            ),
            "role": data.get("interview_prep_role", ""),
        }

    return session


def list_base_resumes() -> list[str]:
    """Return available base resume filenames from base_resume/."""
    if not BASE_RESUME_DIR.exists():
        return []
    return sorted(f.name for f in BASE_RESUME_DIR.glob("*.md"))
