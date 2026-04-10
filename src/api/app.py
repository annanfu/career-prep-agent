"""FastAPI application factory for the Career Prep Agent."""

import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Career Prep Agent")

# Static files & templates
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(_TEMPLATE_DIR))

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
from src.api.routes.chat import router as chat_router  # noqa: E402
from src.api.routes.download import router as download_router  # noqa: E402
from src.api.routes.knowledge import router as knowledge_router  # noqa: E402
from src.api.routes.pages import router as pages_router  # noqa: E402
from src.api.routes.pipeline import router as pipeline_router  # noqa: E402
from src.api.routes.tracker import router as tracker_router  # noqa: E402

app.include_router(pages_router)
app.include_router(pipeline_router, prefix="/api/pipeline")
app.include_router(tracker_router, prefix="/api/tracker")
app.include_router(chat_router, prefix="/api/chat")
app.include_router(knowledge_router, prefix="/api/knowledge")
app.include_router(download_router, prefix="/api/download")
