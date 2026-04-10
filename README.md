# Career Prep Agent

An AI-powered job application pipeline that automates resume tailoring, application tracking, and interview preparation — built with a multi-agent architecture using LangGraph, RAG, and ChromaDB.

---

## What It Does

1. **Paste a job description** → the agent parses requirements, retrieves your relevant experiences from a personal knowledge base, and rewrites your resume to match.
2. **Scores your fit** (1–5) before you spend time applying, so you can prioritize.
3. **Quality review loop** — checks keyword coverage and faithfulness (no hallucinated skills) before saving.
4. **Saves everything**: tailored resume (`.md`), archived JD (`.json`), STAR stories (`.json`), and a running tracker (`.csv`).
5. **Conversational refinement** — chat with the agent to tweak bullets, tone, or emphasis.
6. **Interview prep** — when you land an interview, the agent deep-retrieves your stories, researches the company via web search, surfaces likely questions, and generates STARL-format answers.

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
│   • Relevance scoring    │  LLM per chunk
│   • Fit score (1-5)      │  Rule-based formula
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Node 2: Resume Tailor    │  LLM call 1: tailor resume (base resume + JD only)
│   • STAR story generator │  LLM call 2: generate STAR stories (RAG evidence)
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│ Node 3: Quality Reviewer │  Rule-based: keyword coverage ≥ 70%
│                          │  LLM: faithfulness check vs base resume
└──────────┬───────────────┘
           │
      pass? ── Yes ──→ save_and_track() ──→ Done
           │                ├── Save resume .md
           No               ├── Save JD .json
           │                ├── Save STAR stories .json
           └→ Node 2        └── Append tracker.csv
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
| Frontend | Streamlit | Rapid prototyping with session state |
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
streamlit run src/main.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## Quick Start — Docker

```bash
# 1. Clone and configure .env (steps 1-3 above)
# 2. Create knowledge_base/ and base_resume/ with your content (steps 4-6 above)

# 3. Build and run
docker-compose up --build
```

Open [http://localhost:8501](http://localhost:8501)

Docker mounts `knowledge_base/`, `base_resume/`, `output/`, and `chroma_data/` as volumes, so your data persists outside the container. Edit files on your host machine, then click "Re-ingest" in the sidebar.

---

## Usage

1. **Sidebar → Re-ingest Documents**: builds the ChromaDB vector store from your knowledge base. Run once, then again whenever you add/edit files.
2. **Fill in Target Role + Company Name** + paste the full JD text.
3. **Generate Tailored Resume**: pipeline runs in ~30–60 seconds depending on your LLM provider.
4. **Check the fit score**: ≥3.5 = good fit, <3.5 = consider skipping.
5. **Review Pipeline Details**: expand to see matched experiences, STAR stories, gaps, and review feedback.
6. **Refine via Chat**: use the chat assistant at the bottom (mode: "Refine Resume") to iteratively adjust.
7. **Download or Save** the final version.
8. **Tracker tab**: update application status; set to `interview` and click "Prepare Interview" to generate interview prep.
9. **Interview Prep Chat**: switch chat mode to "Interview Prep" for follow-up questions and sample answers.

---

## Switching LLM Providers

Edit `.env`:

```bash
LLM_PROVIDER=openai    # or gemini, groq
```

Model names are configured in `src/llm.py`:

```python
_OPENAI_FAST = "gpt-4o-mini"       # JD parsing, review
_OPENAI_QUALITY = "gpt-4o"         # Resume tailoring
```

The app is provider-agnostic — switching providers requires changing only `.env` and optionally the model constants in `src/llm.py`. No pipeline code changes needed.

---

## Project Structure

```
career-prep-agent/
├── src/
│   ├── main.py                      # Streamlit app (UI + global chatbot)
│   ├── llm.py                       # Central LLM factory (provider-agnostic)
│   ├── state.py                     # GraphState + InterviewPrepState TypedDicts
│   ├── graph.py                     # LangGraph Phase 1 pipeline
│   ├── graph_interview.py           # LangGraph Phase 2 (fan-out/fan-in)
│   ├── agents/
│   │   ├── jd_analyzer_matcher.py   # Node 1: JD parse + RAG + fit score
│   │   ├── resume_tailor.py         # Node 2: tailor resume + STAR stories
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

---

## Design Decisions

**Why separate resume tailoring from STAR story generation?**
Combining them in one LLM call caused hallucination — RAG chunks overwhelmed the base resume context. Splitting into two calls (resume = base resume + JD only, STAR = RAG evidence + JD) eliminates this.

**Why LangGraph instead of LangChain LCEL?**
The quality review feedback loop requires conditional routing back to Node 2 — a cycle that LCEL chains cannot express. LangGraph's `StateGraph` + `add_conditional_edges` handles this natively.

**Why hybrid rule-based + LLM in the reviewer?**
Keyword coverage is deterministic and cheap (string matching). Faithfulness requires understanding — that's where LLM adds value. Using LLM only where needed keeps costs low.

**Why fan-out/fan-in for interview prep?**
The three sub-agents (deep RAG, company research, question mining) are independent. Running them in parallel cuts latency by ~3x vs sequential.

**Why custom chunk separators for RAG?**
Default `RecursiveCharacterTextSplitter` splits at 1000 chars, cutting STAR stories in half. Custom separators (`\nStory`, `\nSituation`, `\n## `) with 2000-char chunks keep stories intact.

**Why ChromaDB (local)?**
Zero infrastructure, no cost, fully rebuildable. If corrupted: `python -m src.rag.ingest`.
