"""Shared LangGraph state schemas for the career prep agent pipeline."""

from typing import List, Optional, TypedDict


class JDRequirements(TypedDict):
    """Parsed job description fields plus fit scoring."""

    company: str
    role: str
    jd_url: Optional[str]
    required_skills: List[str]
    preferred_skills: List[str]
    key_responsibilities: List[str]
    soft_skills: List[str]
    domain: str
    experience_level: str
    fit_score: float   # 1–5, computed rule-based after RAG matching
    fit_grade: str     # A–F


class MatchedExperience(TypedDict):
    """A single RAG-retrieved experience chunk matched to a JD requirement."""

    requirement: str
    evidence: str
    source_doc: str
    relevance_score: float
    related_chunks: List[str]


class ReviewResult(TypedDict):
    """Output of the quality reviewer node."""

    passed: bool
    feedback: str
    keyword_coverage: float
    faithfulness_issues: List[str]


class InterviewPrepState(TypedDict):
    """Shared state for the Phase 2 interview prep pipeline."""

    # --- Input (loaded from Phase 1 outputs) ---
    resume_content: str          # saved tailored resume markdown
    jd_data: dict                # archived JD JSON (raw + parsed + fit score)
    company: str
    role: str

    # --- Sub-agent A output ---
    deep_experiences: List[dict]  # enriched STAR-ready evidence per bullet
    bq_templates: dict            # self-intro, why-company, etc. from KB
    interview_notes: str          # past interview notes and reflections

    # --- Sub-agent B output ---
    company_brief: dict           # structured company research

    # --- Sub-agent C output ---
    interview_questions: List[dict]  # categorized likely questions

    # --- Content generator output ---
    interview_prep_output: str    # final markdown prep document
    saved_prep_path: str          # file path where prep was saved


class GraphState(TypedDict):
    """Full shared state passed between all LangGraph nodes."""

    # --- Input ---
    jd_raw: str
    jd_url: Optional[str]
    target_role: str          # e.g. "Software Engineer", "Fund Accountant"
    company_name: str         # user-provided, authoritative for search
    base_resume_path: str
    base_resume_content: str

    # --- Node 1 output ---
    jd_requirements: Optional[JDRequirements]
    matched_experiences: List[MatchedExperience]
    gaps: List[str]

    # --- Node 2 output ---
    draft_content: str
    star_stories: List[dict]

    # --- Node 3 output ---
    review_result: Optional[ReviewResult]
    revision_count: int

    # --- Change summary (rule-based, computed in save_and_track) ---
    change_summary: dict  # keywords_added/missing, bullets_modified, coverage

    # --- Save output ---
    saved_resume_path: str
    saved_jd_path: str
    saved_stories_path: str
    tracker_updated: bool

    # --- Final ---
    final_output: str
