"""Node 1: JD Analyzer & Matcher — parse JD, retrieve, score fit."""

import json
import re
from pathlib import Path

from dotenv import load_dotenv

from src.llm import get_fast_llm
from src.rag.retriever import retrieve_experiences
from src.state import GraphState, MatchedExperience

load_dotenv()

PROMPT_PATH = Path("src/prompts/jd_analyzer.txt")

RELEVANCE_THRESHOLD = 0.5
TOP_K_PER_QUERY = 5
GAP_PENALTY = 0.2


# ---------------------------------------------------------------------------
# Step 1: LLM — parse JD into structured JSON
# ---------------------------------------------------------------------------

def _parse_jd(jd_text: str) -> dict:
    """Call Gemini Flash to extract structured JD fields.

    Args:
        jd_text: Raw job description text.

    Returns:
        Dict matching JDRequirements schema (without fit_score/fit_grade).
    """
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{jd_text}", jd_text)

    llm = get_fast_llm(temperature=0)
    response = llm.invoke(prompt)
    raw = (response.content or "").strip()

    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return json.loads(raw)


# ---------------------------------------------------------------------------
# Step 2: RAG — retrieve and score experience chunks
# ---------------------------------------------------------------------------

def _score_chunk_vs_requirements(
    chunk: str,
    requirements: list[str],
) -> tuple[str, float]:
    """Score a chunk against all requirements in one LLM call.

    Returns the best-matching requirement and its relevance score.
    Using a single batched call instead of one call per requirement
    keeps total LLM calls equal to the number of chunks (not N×M).

    Args:
        chunk: Retrieved knowledge base text.
        requirements: List of JD required skills / responsibilities.

    Returns:
        Tuple of (best_requirement, score 0.0–1.0).
    """
    if not requirements:
        return "", 0.0

    llm = get_fast_llm(temperature=0)
    req_list = "\n".join(
        f"{i + 1}. {r}" for i, r in enumerate(requirements)
    )
    # Truncate chunk to avoid token limits
    chunk_trimmed = chunk[:600]

    # Step 1: ask which requirement number best matches + score
    prompt = (
        "You are a resume screener.\n\n"
        f"Requirements (numbered):\n{req_list}\n\n"
        f"Experience snippet:\n{chunk_trimmed}\n\n"
        "Reply with TWO lines only, no extra text:\n"
        "LINE 1: the requirement number that best matches (e.g. 2)\n"
        "LINE 2: relevance score 0.0-1.0 (e.g. 0.8)"
    )
    response = llm.invoke(prompt)
    raw = (response.content or "").strip()

    try:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        req_idx = int(lines[0]) - 1
        score = float(lines[1])
        best_req = (
            requirements[req_idx]
            if 0 <= req_idx < len(requirements) else ""
        )
        return best_req, max(0.0, min(1.0, score))
    except (ValueError, IndexError):
        pass

    # Fallback: extract any float from the response
    floats = re.findall(r"\b0\.\d+\b|\b1\.0\b", raw)
    if floats:
        score = float(floats[-1])
        nums = re.findall(r"\b([1-9])\b", raw)
        req_idx = int(nums[0]) - 1 if nums else 0
        best_req = (
            requirements[req_idx]
            if 0 <= req_idx < len(requirements) else ""
        )
        return best_req, max(0.0, min(1.0, score))

    return "", 0.0


def _retrieve_and_score(
    jd_requirements: dict,
) -> tuple[list[MatchedExperience], list[str]]:
    """Build queries from JD, retrieve chunks, score relevance, flag gaps.

    Args:
        jd_requirements: Parsed JD dict with required_skills and
            key_responsibilities.

    Returns:
        Tuple of (matched_experiences, gaps).
    """
    required_skills: list[str] = jd_requirements.get(
        "required_skills", []
    )
    responsibilities: list[str] = jd_requirements.get(
        "key_responsibilities", []
    )

    queries = required_skills + responsibilities
    if not queries:
        return [], []

    chunks = retrieve_experiences(queries, top_k=TOP_K_PER_QUERY)

    matched: list[MatchedExperience] = []

    # One LLM call per chunk (batch all requirements) — not per skill
    for chunk_data in chunks:
        content = chunk_data["content"]
        source = chunk_data["source_doc"]
        related = chunk_data.get("related_chunks", [])

        best_req, best_score = _score_chunk_vs_requirements(
            content, required_skills
        )
        matched.append(
            MatchedExperience(
                requirement=best_req,
                evidence=content,
                source_doc=source,
                relevance_score=best_score,
                related_chunks=related,
            )
        )

    # Drop chunks that scored 0 — pure noise, don't send to Node 2
    matched = [m for m in matched if m["relevance_score"] > 0]

    # Flag gaps: required skills with no matched evidence above threshold
    covered = {
        m["requirement"]
        for m in matched
        if m["relevance_score"] >= RELEVANCE_THRESHOLD
    }
    gaps = [s for s in required_skills if s not in covered]

    return matched, gaps


# ---------------------------------------------------------------------------
# Step 3: Rule-based fit score (no LLM)
# ---------------------------------------------------------------------------

def _compute_fit_score(
    jd_requirements: dict,
    matched_experiences: list[MatchedExperience],
    gaps: list[str],
) -> tuple[float, str]:
    """Compute fit score (1–5) and grade (A–F) — rule-based, no LLM.

    Formula (per SPEC.md):
        skills_match = covered_skills / total_required_skills
        responsibility_coverage = covered_resp / total_responsibilities
        fit_score = (skills_match * 2.5 + responsibility_coverage * 2.5)
                    - (len(gaps) * 0.2), clamped to [1, 5]

    Args:
        jd_requirements: Parsed JD dict.
        matched_experiences: List of matched experience dicts.
        gaps: List of unmet required skills.

    Returns:
        Tuple of (fit_score, fit_grade).
    """
    required_skills = jd_requirements.get("required_skills", [])
    responsibilities = jd_requirements.get("key_responsibilities", [])

    if required_skills:
        covered_skills = {
            m["requirement"]
            for m in matched_experiences
            if m["relevance_score"] >= RELEVANCE_THRESHOLD
        }
        skills_match = len(covered_skills) / len(required_skills)
    else:
        skills_match = 1.0

    if responsibilities:
        covered_resp = {
            m["requirement"]
            for m in matched_experiences
            if m["relevance_score"] >= RELEVANCE_THRESHOLD
            and m["requirement"] in responsibilities
        }
        responsibility_coverage = (
            len(covered_resp) / len(responsibilities)
        )
    else:
        responsibility_coverage = 1.0

    raw = (
        skills_match * 2.5
        + responsibility_coverage * 2.5
        - len(gaps) * GAP_PENALTY
    )
    fit_score = round(max(1.0, min(5.0, raw)), 2)

    if fit_score >= 4.5:
        grade = "A"
    elif fit_score >= 3.5:
        grade = "B"
    elif fit_score >= 2.5:
        grade = "C"
    elif fit_score >= 1.5:
        grade = "D"
    else:
        grade = "F"

    return fit_score, grade


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def jd_analyzer_matcher_node(state: GraphState) -> dict:
    """LangGraph Node 1: parse JD, retrieve experiences, compute fit score.

    Args:
        state: Current GraphState with jd_raw populated.

    Returns:
        Dict with keys: jd_requirements, matched_experiences, gaps.
    """
    print("\n[Node 1] Parsing job description...")
    parsed = _parse_jd(state["jd_raw"])

    if state.get("jd_url") and not parsed.get("jd_url"):
        parsed["jd_url"] = state["jd_url"]

    # Override with user-provided values if available
    if state.get("company_name"):
        parsed["company"] = state["company_name"]
    if state.get("target_role"):
        parsed["role"] = state["target_role"]

    print(
        f"  Parsed: {parsed['company']} — {parsed['role']}\n"
        f"  Required skills: {parsed.get('required_skills', [])}"
    )

    print("\n[Node 1] Retrieving + scoring experiences...")
    matched, gaps = _retrieve_and_score(parsed)
    print(
        f"  Matched {len(matched)} chunks | "
        f"{len(gaps)} gap(s): {gaps}"
    )

    print("\n[Node 1] Computing fit score (rule-based)...")
    fit_score, fit_grade = _compute_fit_score(parsed, matched, gaps)
    parsed["fit_score"] = fit_score
    parsed["fit_grade"] = fit_grade
    print(f"  Fit score: {fit_score}/5 ({fit_grade})")

    return {
        "jd_requirements": parsed,
        "matched_experiences": matched,
        "gaps": gaps,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_JD = """
    Software Engineer — Data Platform
    Stripe | San Francisco, CA (Remote OK)

    We're looking for a Software Engineer to join our Data Platform team.
    You'll build and maintain the infrastructure that powers Stripe's
    internal data products used by hundreds of engineers.

    Requirements:
    - 3+ years of software engineering experience
    - Strong Python skills and experience with ETL pipelines
    - Experience with distributed systems and large-scale data processing
    - Proficiency in SQL and relational databases
    - Familiarity with AWS (S3, Lambda, RDS) or equivalent cloud platforms
    - Experience with Docker and CI/CD pipelines

    Nice to have:
    - Experience with Spark or Kafka
    - Kubernetes experience
    - Go programming language

    Responsibilities:
    - Design and build scalable data pipelines
    - Maintain and improve data infrastructure reliability
    - Collaborate with cross-functional teams to deliver data products
    - Write clean, well-tested code with high coverage
    """

    test_state: GraphState = {
        "jd_raw": SAMPLE_JD,
        "jd_url": None,
        "base_resume_path": "",
        "base_resume_content": "",
        "jd_requirements": None,
        "matched_experiences": [],
        "gaps": [],
        "draft_content": "",
        "star_stories": [],
        "review_result": None,
        "revision_count": 0,
        "saved_resume_path": "",
        "saved_jd_path": "",
        "saved_stories_path": "",
        "tracker_updated": False,
        "final_output": "",
    }

    result = jd_analyzer_matcher_node(test_state)

    print("\n" + "=" * 60)
    print("JD REQUIREMENTS:")
    print(json.dumps(result["jd_requirements"], indent=2))
    print(f"\nMATCHED EXPERIENCES: {len(result['matched_experiences'])}")
    for exp in result["matched_experiences"][:3]:
        print(
            f"  [{exp['relevance_score']:.2f}]"
            f" {exp['requirement'][:40]}"
            f" | {exp['source_doc']}"
        )
    print(f"\nGAPS: {result['gaps']}")
    print(
        f"\nFIT SCORE: {result['jd_requirements']['fit_score']}/5"
        f" ({result['jd_requirements']['fit_grade']})"
    )
