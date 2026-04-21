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

def _score_all_chunks_batch(
    chunks: list[dict],
    requirements: list[str],
) -> list[tuple[str, float]]:
    """Score ALL chunks against requirements in a SINGLE LLM call.

    Args:
        chunks: List of chunk dicts with 'content' field.
        requirements: List of JD required skills.

    Returns:
        List of (best_requirement, score) tuples, one per chunk.
    """
    if not requirements or not chunks:
        return [("", 0.0)] * len(chunks)

    req_list = "\n".join(
        f"{i + 1}. {r}" for i, r in enumerate(requirements)
    )

    chunk_list = "\n\n".join(
        f"[CHUNK {i + 1}]\n{c['content'][:400]}"
        for i, c in enumerate(chunks[:20])  # Cap at 20 chunks
    )

    prompt = (
        "You are a resume screener. Score each experience chunk "
        "against the job requirements.\n\n"
        f"REQUIREMENTS (numbered):\n{req_list}\n\n"
        f"EXPERIENCE CHUNKS:\n{chunk_list}\n\n"
        f"For each chunk (1 to {min(len(chunks), 20)}), reply "
        "with one line in this exact format:\n"
        "CHUNK_NUMBER REQUIREMENT_NUMBER SCORE\n\n"
        "Example:\n1 3 0.8\n2 1 0.6\n3 5 0.2\n\n"
        "Rules:\n"
        "- REQUIREMENT_NUMBER = which requirement best matches\n"
        "- SCORE = relevance 0.0 to 1.0\n"
        "- One line per chunk, no extra text"
    )

    llm = get_fast_llm(temperature=0)
    response = llm.invoke(prompt)
    raw = (response.content or "").strip()

    # Parse the batch response
    results: list[tuple[str, float]] = []
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    for i in range(min(len(chunks), 20)):
        try:
            parts = lines[i].split()
            req_idx = int(parts[1]) - 1
            score = float(parts[2])
            best_req = (
                requirements[req_idx]
                if 0 <= req_idx < len(requirements) else ""
            )
            results.append(
                (best_req, max(0.0, min(1.0, score))),
            )
        except (ValueError, IndexError):
            # Fallback: try to extract any numbers from the line
            if i < len(lines):
                floats = re.findall(
                    r"\b0\.\d+\b|\b1\.0\b", lines[i],
                )
                if floats:
                    results.append(("", float(floats[-1])))
                    continue
            results.append(("", 0.0))

    # Pad if we got fewer results than chunks
    while len(results) < len(chunks):
        results.append(("", 0.0))

    return results


def _retrieve_and_score(
    jd_requirements: dict,
) -> tuple[list[MatchedExperience], list[str]]:
    """Build queries from JD, retrieve chunks, score in one batch.

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

    # Single LLM call to score ALL chunks at once
    scores = _score_all_chunks_batch(chunks, required_skills)

    matched: list[MatchedExperience] = []
    for chunk_data, (best_req, best_score) in zip(
        chunks, scores,
    ):
        matched.append(
            MatchedExperience(
                requirement=best_req,
                evidence=chunk_data["content"],
                source_doc=chunk_data["source_doc"],
                relevance_score=best_score,
                related_chunks=chunk_data.get(
                    "related_chunks", [],
                ),
            )
        )

    matched = [m for m in matched if m["relevance_score"] > 0]

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
        "persona_summary": "",
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
