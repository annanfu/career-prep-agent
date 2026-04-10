"""File download routes for resumes and interview prep documents."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/resume/{filename}")
def download_resume(filename: str) -> FileResponse:
    """Download a tailored resume from output/resumes/.

    Args:
        filename: Resume filename.

    Returns:
        FileResponse forcing a download.
    """
    path = Path("output/resumes") / filename
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Resume not found: {filename}",
        )
    return FileResponse(
        path=str(path),
        filename=filename,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.get("/prep/{filename}")
def download_prep(filename: str) -> FileResponse:
    """Download an interview prep document.

    Args:
        filename: Prep document filename.

    Returns:
        FileResponse forcing a download.
    """
    path = Path("output/interview_prep") / filename
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Prep doc not found: {filename}",
        )
    return FileResponse(
        path=str(path),
        filename=filename,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
