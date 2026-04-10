"""Knowledge base routes — ingest documents and ChromaDB status."""

from pathlib import Path

from fastapi import APIRouter, Request, Response

from src.api.app import templates
from src.api.dependencies import task_manager
from src.api.tasks import TaskStatus

router = APIRouter()

_SPINNER_HTML = """
<svg class="animate-spin h-3 w-3"
     xmlns="http://www.w3.org/2000/svg"
     fill="none" viewBox="0 0 24 24">
  <circle class="opacity-25" cx="12" cy="12" r="10"
          stroke="currentColor" stroke-width="4"></circle>
  <path class="opacity-75" fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z">
  </path>
</svg>
"""


def _ingest(_task_state: object = None) -> dict:
    """Run the RAG ingest pipeline (background thread).

    Returns:
        Dict with files and chunks counts.
    """
    from src.rag.ingest import ingest_documents
    files, chunks = ingest_documents("knowledge_base")
    return {"files": files, "chunks": chunks}


@router.post("/ingest")
def start_ingest() -> Response:
    """Start document ingestion as a background task."""
    task_id = task_manager.submit(_ingest)
    return Response(
        content=(
            f'<div hx-get="/api/knowledge/ingest-status/'
            f'{task_id}" hx-trigger="every 2s" '
            f'hx-swap="outerHTML" '
            f'class="flex items-center gap-2 text-xs '
            f'text-blue-600">{_SPINNER_HTML} '
            f"Ingesting...</div>"
        ),
        media_type="text/html",
    )


@router.get("/ingest-status/{task_id}")
def ingest_status(task_id: str) -> Response:
    """Poll ingest task status."""
    task = task_manager.get(task_id)
    if not task or task.status == TaskStatus.RUNNING:
        return Response(
            content=(
                f'<div hx-get="/api/knowledge/ingest-status/'
                f'{task_id}" hx-trigger="every 2s" '
                f'hx-swap="outerHTML" '
                f'class="flex items-center gap-2 text-xs '
                f'text-blue-600">{_SPINNER_HTML} '
                f"Ingesting...</div>"
            ),
            media_type="text/html",
        )

    if task.status == TaskStatus.FAILED:
        task_manager.cleanup(task_id)
        return Response(
            content=(
                '<p class="text-xs text-red-600">'
                f"Ingest failed: {task.error}</p>"
            ),
            media_type="text/html",
        )

    result = task.result or {}
    task_manager.cleanup(task_id)
    files = result.get("files", 0)
    chunks = result.get("chunks", 0)
    return Response(
        content=(
            '<p class="text-xs text-green-600">'
            f"Ingested {files} file(s) "
            f"&rarr; {chunks} chunks.</p>"
        ),
        media_type="text/html",
    )


@router.get("/chromadb-status")
def chromadb_status(request: Request) -> Response:
    """Return ChromaDB stats partial."""
    if not Path("chroma_data").exists():
        return templates.TemplateResponse(
            request,
            "partials/chromadb_status.html",
            {
                "stats": None,
                "error": None,
                "total_chunks": 0,
                "num_files": 0,
            },
        )

    try:
        from langchain_chroma import Chroma
        from langchain_community.embeddings import (
            HuggingFaceEmbeddings,
        )

        emb = HuggingFaceEmbeddings(
            model_name=(
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
        )
        vs = Chroma(
            collection_name="knowledge_base",
            embedding_function=emb,
            persist_directory="chroma_data",
        )
        col = vs._collection
        metas = col.get(include=["metadatas"])

        file_stats: dict[str, int] = {}
        for m in metas["metadatas"]:
            src = m.get("source_file", "unknown")
            file_stats[src] = file_stats.get(src, 0) + 1

        stats = [
            {"file": f, "chunks": c}
            for f, c in sorted(file_stats.items())
        ]
        return templates.TemplateResponse(
            request,
            "partials/chromadb_status.html",
            {
                "stats": stats,
                "total_chunks": col.count(),
                "num_files": len(file_stats),
                "error": None,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "partials/chromadb_status.html",
            {
                "stats": None,
                "error": str(e),
                "total_chunks": 0,
                "num_files": 0,
            },
        )
