"""save_and_track: pure Python file I/O — zero LLM calls."""

import csv
import difflib
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.state import GraphState

OUTPUT_DIR = Path("output")
RESUMES_DIR = OUTPUT_DIR / "resumes"
JDS_DIR = OUTPUT_DIR / "jds"
STORIES_DIR = OUTPUT_DIR / "stories"
TRACKER_CSV = OUTPUT_DIR / "tracker.csv"

TRACKER_COLUMNS = [
    "company",
    "role",
    "fit_score",
    "status",
    "notes",
    "resume_filename",
    "date",
    "fit_grade",
    "jd_filename",
    "stories_filename",
    "jd_url",
]


def _safe_filename(text: str) -> str:
    """Convert arbitrary text to a filesystem-safe string.

    Replaces spaces with underscores and removes characters that are
    not alphanumeric, underscore, or hyphen.

    Args:
        text: Raw string (e.g. company name or role title).

    Returns:
        Sanitized string safe for use in filenames.
    """
    text = text.strip().replace(" ", "_")
    return re.sub(r"[^\w\-]", "", text)


def _build_filename(jd_requirements: dict) -> str:
    """Build the base filename from date, company, and role.

    Args:
        jd_requirements: Parsed JD dict with 'company' and 'role' keys.

    Returns:
        Filename stem like '2026-04-08_Stripe_Software_Engineer'.
    """
    today = date.today().isoformat()
    company = _safe_filename(
        jd_requirements.get("company", "Unknown")
    )
    role = _safe_filename(jd_requirements.get("role", "Role"))
    return f"{today}_{company}_{role}"


def _compute_change_summary(
    base_resume: str,
    draft: str,
    jd_requirements: dict,
    keyword_coverage: float,
) -> dict:
    """Compute a rule-based diff summary between base resume and draft.

    No LLM calls. Uses string matching and difflib.

    Args:
        base_resume: Original base resume Markdown text.
        draft: Tailored resume draft Markdown text.
        jd_requirements: Parsed JD dict with required_skills and
            preferred_skills.
        keyword_coverage: Float from Node 3 ReviewResult.

    Returns:
        Dict with keys:
            keywords_added (list[str]),
            keywords_missing (list[str]),
            bullets_modified (dict with 'removed' and 'added' lists),
            keyword_coverage (float).
    """
    required = jd_requirements.get("required_skills", [])
    preferred = jd_requirements.get("preferred_skills", [])
    all_skills = required + preferred

    base_lower = base_resume.lower()
    draft_lower = draft.lower()

    keywords_added = [
        s for s in all_skills
        if s.lower() not in base_lower and s.lower() in draft_lower
    ]
    keywords_missing = [
        s for s in all_skills
        if s.lower() not in draft_lower
    ]

    # Diff non-empty lines only
    base_lines = [l for l in base_resume.splitlines() if l.strip()]
    draft_lines = [l for l in draft.splitlines() if l.strip()]

    removed, added = [], []
    for line in difflib.unified_diff(
        base_lines, draft_lines, lineterm="", n=0
    ):
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("@@"):
            continue
        if line.startswith("-"):
            removed.append(line[1:].strip())
        elif line.startswith("+"):
            added.append(line[1:].strip())

    return {
        "keywords_added": keywords_added,
        "keywords_missing": keywords_missing,
        "bullets_modified": {"removed": removed, "added": added},
        "keyword_coverage": keyword_coverage,
    }


def _ensure_dirs() -> None:
    """Create all required output directories if they don't exist."""
    for d in (RESUMES_DIR, JDS_DIR, STORIES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _append_tracker(row: dict) -> None:
    """Append one row to tracker.csv, creating it with headers if needed.

    Args:
        row: Dict with keys matching TRACKER_COLUMNS.
    """
    file_exists = TRACKER_CSV.exists()

    with open(TRACKER_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACKER_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in TRACKER_COLUMNS})


def save_and_track_node(state: GraphState) -> dict:
    """LangGraph save node: persist resume, JD, stories, tracker.

    Zero LLM calls — pure Python file I/O only.

    Args:
        state: Current GraphState. Requires jd_requirements,
            final_output, star_stories.

    Returns:
        Dict with keys: saved_resume_path, saved_jd_path,
            saved_stories_path, tracker_updated.
    """
    print("\n[save_and_track] Saving outputs...")

    _ensure_dirs()

    jd_req = state["jd_requirements"]
    filename = _build_filename(jd_req)

    # 1. Save tailored resume
    resume_path = RESUMES_DIR / f"{filename}.md"
    resume_path.write_text(state["final_output"], encoding="utf-8")
    print(f"  Resume saved  : {resume_path}")

    # 2. Save JD archive (raw + parsed + fit score)
    jd_path = JDS_DIR / f"{filename}_jd.json"
    jd_archive = {
        "jd_raw": state.get("jd_raw", ""),
        "jd_url": state.get("jd_url"),
        "parsed": jd_req,
        "date_saved": date.today().isoformat(),
    }
    jd_path.write_text(
        json.dumps(jd_archive, indent=2), encoding="utf-8"
    )
    print(f"  JD archive    : {jd_path}")

    # 3. Save STAR stories
    stories_path = STORIES_DIR / f"{filename}_stories.json"
    stories_path.write_text(
        json.dumps(state.get("star_stories", []), indent=2),
        encoding="utf-8",
    )
    print(f"  STAR stories  : {stories_path}")

    # 4. Tracker append is deferred — user clicks "Mark as Applied"
    #    in the UI to add the row (see pipeline routes).

    # 5. Compute change summary (rule-based, no LLM)
    keyword_coverage = (
        (state.get("review_result") or {}).get("keyword_coverage", 0.0)
    )
    change_summary = _compute_change_summary(
        base_resume=state.get("base_resume_content", ""),
        draft=state["final_output"],
        jd_requirements=jd_req,
        keyword_coverage=keyword_coverage,
    )
    print(
        f"  Change summary: +{len(change_summary['keywords_added'])} keywords"
        f", {len(change_summary['keywords_missing'])} missing"
        f", {len(change_summary['bullets_modified']['added'])} bullets added"
    )

    return {
        "saved_resume_path": str(resume_path),
        "saved_jd_path": str(jd_path),
        "saved_stories_path": str(stories_path),
        "tracker_updated": False,
        "change_summary": change_summary,
    }


if __name__ == "__main__":
    from src.state import JDRequirements

    sample_jd: JDRequirements = {
        "company": "Stripe",
        "role": "Software Engineer Data Platform",
        "jd_url": "https://stripe.com/jobs/123",
        "required_skills": ["Python", "ETL", "SQL"],
        "preferred_skills": ["Spark"],
        "key_responsibilities": ["Build data pipelines"],
        "soft_skills": [],
        "domain": "fintech",
        "experience_level": "mid-level",
        "fit_score": 3.8,
        "fit_grade": "B",
    }

    test_state: GraphState = {
        "jd_raw": "Sample JD text",
        "jd_url": "https://stripe.com/jobs/123",
        "base_resume_path": "base_resume/resume_master.md",
        "base_resume_content": "",
        "jd_requirements": sample_jd,
        "matched_experiences": [],
        "gaps": [],
        "draft_content": "# Tailored Resume\n\nSample content.",
        "star_stories": [
            {
                "experience_name": "ETL Pipeline",
                "jd_requirement": "Python ETL",
                "situation": "Needed to process 40K APKs",
                "task": "Build ingestion pipeline",
                "action": "Wrote ETL in Python with multiprocessing",
                "result": "98% detection accuracy",
                "source_docs": ["KNOWLEDGE.md"],
            }
        ],
        "review_result": {
            "passed": True,
            "feedback": "",
            "keyword_coverage": 0.83,
            "faithfulness_issues": [],
        },
        "revision_count": 1,
        "change_summary": {},
        "saved_resume_path": "",
        "saved_jd_path": "",
        "saved_stories_path": "",
        "tracker_updated": False,
        "final_output": (
            "# Tailored Resume\n\n"
            "- Built ETL pipeline in Python and SQL for 40K apps.\n"
            "- Deployed on AWS with Docker and CI/CD.\n"
        ),
    }

    result = save_and_track_node(test_state)
    print("\n" + "=" * 60)
    print(f"Resume path  : {result['saved_resume_path']}")
    print(f"JD path      : {result['saved_jd_path']}")
    print(f"Stories path : {result['saved_stories_path']}")
    print(f"Tracker      : {result['tracker_updated']}")
    print(f"Change summary: {json.dumps(result['change_summary'], indent=2)}")

    # Verify tracker row
    df = pd.read_csv(TRACKER_CSV)
    print("\nTracker CSV (last row):")
    print(df.tail(1).to_string(index=False))
