# Career Prep Agent

An AI-powered job application pipeline that automates resume tailoring, application tracking, and interview preparation — built with a multi-agent architecture using LangGraph, RAG, and ChromaDB.

---

## What It Does

1. **Paste a job description** → the agent parses requirements, retrieves your relevant experiences from a personal knowledge base, and rewrites your resume to match.
2. **Scores your fit** (1–5) before you spend time applying, so you can prioritize.
3. **Quality review loop** — checks keyword coverage and faithfulness (no hallucinated skills) before saving.
4. **Saves everything**: tailored resume (`.md`), archived JD (`.json`), and a running tracker (`.csv`).
5. **Conversational refinement** — chat with the agent to tweak bullets, tone, or emphasis. Changes update the editor and save to disk in real time. Toggle **Deep Research** to pull specific facts from your knowledge base via RAG, or restore deleted experiences from the master resume.
6. **Interview prep** — when you land an interview, the agent deep-retrieves your stories, researches the company via web search, surfaces likely questions, and generates STARL-format answers. Chat can edit the prep document directly (Deep Research available here too).
7. **LaTeX export** — generate a `.tex` file from the tailored resume with JD keywords auto-bolded, ready for PDF compilation.

---

## Architecture

### Phase 1 — Resume Tailoring Pipeline

```
User inputs: Target Role + Company Name + JD text
      │
      ▼
┌──────────────────────────┐
│ Node 1: JD Analyzer      │  LLM: parse JD → JSON
│   • RAG retrieval        │  ChromaDB semantic search
│   • Relevance scoring    │  Single batched LLM call for all chunks
│   • Fit score (1-5)      │  Rule-based formula
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Node 2: Resume Tailor    │  LLM: rewrite resume (base resume + JD)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Node 3: Quality Reviewer │  Rule-based: keyword coverage ≥ 70%
│                          │  LLM: faithfulness check vs base resume
└──────────┬───────────────┘
           │
      pass? ── Yes ──→ save_and_track() ──→ Done
           │                ├── Save resume .md
           No               └── Save JD .json
           │                (tracker entry deferred until user
           └→ Node 2         clicks "Mark as Applied")
              (max 2 loops)
```

### Phase 2 — Interview Prep (Orchestrator + Parallel Sub-agents)

```
User selects application (status = "interview")
      │
      ▼
┌─────────────────────────────────┐
│ Orchestrator                     │
│ Loads saved resume + archived JD │
└──────────┬──────────────────────┘
      ┌────┼────┐
      ▼    ▼    ▼
   Sub-A Sub-B Sub-C         (parallel fan-out)
   Deep  Company  Interview
   RAG   Research  Questions
   +BQ   (Tavily)  (Tavily+LLM)
   templates
      └────┼────┘
           ▼                       (fan-in)
┌─────────────────────────────────┐
│ Content Generator                │
│ • Self-introduction              │
│ • "What do you know about X?"   │
│ • "Why this company?"            │
│ • STARL stories per bullet       │
│ • Likely Q&A with answers        │
│ • Gap warnings                   │
└─────────────────────────────────┘
```

---

## Tech Stack

| Category | Technology | Why |
|---|---|---|
| Language | Python 3.11+ | LangChain/LangGraph ecosystem |
| Agent Framework | LangGraph | Supports cycles (review loop) and parallel fan-out/fan-in |
| LLM | Provider-agnostic via `src/llm.py` | Supports OpenAI, Gemini, Groq — switch with one env var |
| RAG | LangChain loaders + RecursiveCharacterTextSplitter | Custom semantic separators for STAR stories |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Local, free, no API key needed |
| Vector DB | ChromaDB (local persistent) | Zero infra, fully rebuildable from source docs |
| Web Search | Tavily API | Company research + interview question mining |
| Backend | FastAPI + Jinja2 | REST API with server-rendered templates, async task management |
| Frontend | Tailwind CSS + HTMX | Modern, lightweight UI with partial page updates — no JS framework needed |
| Containerization | Docker + docker-compose | One-command deployment |

---

## Quick Start — Local

### Prerequisites

- Python 3.11+
- An API key for at least one LLM provider:
  - [OpenAI](https://platform.openai.com/api-keys) (recommended: `gpt-4o-mini` or newer)
  - [Google AI Studio](https://aistudio.google.com/app/apikey) (`gemini-2.5-flash`)
  - [Groq](https://console.groq.com/keys) (free tier, limited)
- (Optional) [Tavily API key](https://tavily.com) for interview prep web search (free: 1000/month)

### Setup

```bash
# 1. Clone
git clone https://github.com/annanfu/career-prep-agent.git
cd career-prep-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env
# Edit .env → fill in your API key(s) and set LLM_PROVIDER
```

### Prepare Your Data

The app needs two directories with your personal content:

```bash
# 4. Create directories
mkdir -p knowledge_base base_resume

# 5. Add your base resume (Markdown format)
#    This is the "master" resume with ALL your experiences — the app selects
#    and tailors from it per job. See base_resume/README.md for format guide.
cp your_resume.md base_resume/resume_master.md

# 6. Add knowledge base documents
#    These are personal docs the RAG system uses for evidence:
#    - Work experience details, STAR stories, project write-ups
#    - Organize into subdirectories:
mkdir -p knowledge_base/{work_experience,projects,interview_prep}
cp your_work_details.md knowledge_base/work_experience/
cp your_project_writeup.md knowledge_base/projects/
```

### Run

```bash
# 7. Ingest knowledge base into ChromaDB (run once, re-run when docs change)
python -m src.rag.ingest

# 8. Launch
uvicorn src.api.app:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

---

## Quick Start — Docker

```bash
# 1. Clone and configure .env (steps 1-3 above)
# 2. Create knowledge_base/ and base_resume/ with your content (steps 4-6 above)

# 3. Build and run
docker-compose up --build
```

Open [http://localhost:8000](http://localhost:8000)

Docker mounts `knowledge_base/`, `base_resume/`, `output/`, and `chroma_data/` as volumes, so your data persists outside the container. Edit files on your host machine, then click "Re-ingest documents" in the sidebar.

---

## Usage

1. **Sidebar → Re-ingest documents**: builds the ChromaDB vector store from your knowledge base. Run once, then again whenever you add/edit files.
2. **Fill in Target Role + Company Name** + paste the full JD text.
3. **Generate Tailored Resume**: pipeline runs with real-time progress tracking (Analyze JD → Tailor Resume → Quality Review → Save).
4. **Check the fit score**: ≥3.5 = good fit, <3.5 = consider skipping.
5. **Review Pipeline Details**: expand to see JD keywords, matched experiences, gaps, keyword changes, tailoring reasoning, and review feedback.
6. **Refine via Chat**: open the Chat Assistant (bottom-right), select "Refine Resume" mode, and request changes. The resume editor updates in real time. Check **Deep Research** to retrieve evidence from your knowledge base via RAG, or ask the chatbot to restore experiences deleted during tailoring (it has access to the master resume).
7. **Download, Save, or Generate .tex** — the LaTeX export auto-bolds JD keywords and quantitative metrics.
8. **Mark as Applied**: click the button to add the application to the tracker. The pipeline no longer auto-tracks — you decide which applications to record.
9. **Tracker tab**: view summary stats (total, by status), update application status; changing to `interview` auto-saves and shows a Generate button.
10. **Interview Prep**: click Generate/View in the Prep column. The prep document appears in a panel below the tracker with its own progress indicator.
11. **Interview Prep Chat**: switch chat mode to "Interview Prep" to edit the prep document — changes save to disk automatically. Deep Research is available here too.

---

## Switching LLM Providers

Edit `.env`:

```bash
LLM_PROVIDER=openai    # or gemini, groq
```

Model names are configured in `src/llm.py`:

```python
_OPENAI_FAST = "gpt-4o-mini"       # JD parsing, review
_OPENAI_QUALITY = "gpt-4.1-mini"   # Resume tailoring, chatbot
```

The app is provider-agnostic — switching providers requires changing only `.env` and optionally the model constants in `src/llm.py`. No pipeline code changes needed.

---

## Project Structure

```
career-prep-agent/
├── src/
│   ├── api/
│   │   ├── app.py                   # FastAPI app, router registration
│   │   ├── tasks.py                 # Background task manager (ThreadPool + progress)
│   │   ├── dependencies.py          # Session store (file-backed), shared state
│   │   └── routes/
│   │       ├── pages.py             # HTML page routes (/, /tracker)
│   │       ├── pipeline.py          # Pipeline run/poll/save endpoints
│   │       ├── tracker.py           # Tracker CRUD + interview prep endpoints
│   │       ├── chat.py              # Chat edit (resume + interview prep)
│   │       ├── knowledge.py         # RAG ingest + ChromaDB status
│   │       └── download.py          # File download endpoints
│   ├── templates/
│   │   ├── base.html                # Layout (Tailwind + HTMX + Inter font)
│   │   ├── index.html               # Resume tailoring page
│   │   ├── tracker.html             # Application tracker page
│   │   └── partials/                # HTMX partial templates
│   ├── static/
│   │   ├── css/app.css              # Custom styles
│   │   └── js/app.js                # Chat, tracker, interview prep JS
│   ├── main_streamlit.py            # Legacy Streamlit frontend (preserved)
│   ├── llm.py                       # Central LLM factory (provider-agnostic)
│   ├── state.py                     # GraphState + InterviewPrepState TypedDicts
│   ├── graph.py                     # LangGraph Phase 1 pipeline
│   ├── graph_interview.py           # LangGraph Phase 2 (fan-out/fan-in)
│   ├── agents/
│   │   ├── jd_analyzer_matcher.py   # Node 1: JD parse + RAG + batched scoring
│   │   ├── resume_tailor.py         # Node 2: tailor resume
│   │   ├── quality_reviewer.py      # Node 3: keyword + faithfulness check
│   │   ├── save_and_track.py        # File I/O, zero LLM calls
│   │   └── interview/
│   │       ├── deep_retriever.py    # Sub-agent A: deep RAG + BQ templates
│   │       ├── company_researcher.py # Sub-agent B: Tavily web search
│   │       └── question_researcher.py # Sub-agent C: interview Q mining
│   ├── rag/
│   │   ├── ingest.py                # Load → chunk → embed → ChromaDB
│   │   └── retriever.py             # Semantic search + cosine dedup
│   ├── prompts/                     # All prompt templates (.txt files)
│   └── utils/
│       └── web_search.py            # Tavily API wrapper
├── knowledge_base/                  # Your personal docs (gitignored)
├── base_resume/                     # Resume templates (gitignored)
├── output/                          # Generated files (gitignored)
├── chroma_data/                     # Vector store (gitignored, rebuildable)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```
