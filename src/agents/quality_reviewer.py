"""Node 3: Quality Reviewer — keyword coverage + faithfulness check."""

import json
import re
from pathlib import Path

from dotenv import load_dotenv

from src.llm import get_fast_llm
from src.state import GraphState, ReviewResult

load_dotenv()
PROMPT_PATH = Path("src/prompts/quality_reviewer.txt")

KEYWORD_COVERAGE_THRESHOLD = 0.7


def _rule_based_keyword_coverage(
    draft: str,
    required_skills: list[str],
) -> float:
    """Count how many required skills appear in the draft (case-insensitive).

    Args:
        draft: Tailored resume markdown text.
        required_skills: List of required skill strings from the JD.

    Returns:
        Coverage ratio between 0.0 and 1.0.
    """
    if not required_skills:
        return 1.0
    draft_lower = draft.lower()
    found = sum(
        1 for skill in required_skills if skill.lower() in draft_lower
    )
    return found / len(required_skills)


def _llm_faithfulness_check(
    draft: str,
    base_resume_content: str,
) -> tuple[list[str], str]:
    """Ask LLM to flag any claims in the resume not present in the base resume.

    Args:
        draft: Tailored resume markdown text.
        base_resume_content: The original base resume (source of truth).

    Returns:
        Tuple of (faithfulness_issues list, feedback string).
    """
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    prompt = prompt_template.replace(
        "{draft_content}", draft
    ).replace(
        "{base_resume_content}", base_resume_content,
    )

    llm = get_fast_llm(temperature=0)
    response = llm.invoke(prompt)
    raw = (response.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
        issues = data.get("faithfulness_issues", [])
        feedback = data.get("feedback", "")
        return issues, feedback
    except json.JSONDecodeError:
        return [], ""


def quality_reviewer_node(state: GraphState) -> dict:
    """LangGraph Node 3: hybrid keyword + faithfulness review.

    Pass criteria (per SPEC.md):
        - faithfulness_issues is empty AND
        - keyword_coverage >= 0.7

    If failing AND revision_count < 2, returns feedback for Node 2.
    If passing OR revision_count >= 2, pipeline proceeds to save.

    Args:
        state: Current GraphState with draft_content populated.

    Returns:
        Dict with keys: review_result (ReviewResult),
            revision_count (int).
    """
    print("\n[Node 3] Running quality review...")

    draft = state["draft_content"]
    required_skills = (state.get("jd_requirements") or {}).get(
        "required_skills", []
    )
    matched = state.get("matched_experiences", [])
    revision_count = state.get("revision_count", 0) + 1

    # Part A: rule-based keyword coverage (no LLM)
    coverage = _rule_based_keyword_coverage(draft, required_skills)
    print(f"  Keyword coverage: {coverage:.0%}")

    # Part B: LLM faithfulness check (against base resume, not RAG chunks)
    base_resume = state.get("base_resume_content", "")
    issues, feedback = _llm_faithfulness_check(draft, base_resume)
    print(f"  Faithfulness issues: {len(issues)}")

    passed = (not issues) and (coverage >= KEYWORD_COVERAGE_THRESHOLD)

    # Build combined feedback for Node 2 if failing
    if not passed:
        parts = []
        if coverage < KEYWORD_COVERAGE_THRESHOLD:
            missing = [
                s for s in required_skills
                if s.lower() not in draft.lower()
            ]
            parts.append(
                f"Keyword coverage is {coverage:.0%} "
                f"(need ≥{KEYWORD_COVERAGE_THRESHOLD:.0%}). "
                f"Missing: {missing}"
            )
        if feedback:
            parts.append(feedback)
        feedback = " | ".join(parts)

    result = ReviewResult(
        passed=passed,
        feedback=feedback,
        keyword_coverage=coverage,
        faithfulness_issues=issues,
    )

    status = "PASS" if passed else "FAIL"
    print(f"  Review result: {status} (revision #{revision_count})")

    return {
        "review_result": result,
        "revision_count": revision_count,
    }


if __name__ == "__main__":
    # Test with a synthetic draft
    SAMPLE_DRAFT = """
    # Sample Candidate

    ## Technical Skills
    Python, SQL, ETL, AWS, Docker, distributed systems

    ## Work Experience
    **Northeastern University** — Research Assistant
    - Built an end-to-end ETL pipeline in Python for 40K Android APKs,
      achieving 98% malware detection accuracy using scikit-learn.
    - Deployed workloads on AWS (S3, Lambda, RDS) with Docker containers.
    """

    SAMPLE_EVIDENCE = [
        {
            "requirement": "Python ETL pipelines",
            "evidence": (
                "Built an end-to-end ML-based Android malware detection "
                "system in Python, achieving 98% detection accuracy across "
                "40K applications through ETL workflows."
            ),
            "source_doc": "KNOWLEDGE.md",
            "relevance_score": 0.9,
            "related_chunks": [],
        }
    ]

    SAMPLE_JD_REQS = {
        "required_skills": ["Python", "ETL", "SQL", "AWS", "Docker"],
    }

    test_state: GraphState = {
        "jd_raw": "",
        "jd_url": None,
        "base_resume_path": "",
        "base_resume_content": "",
        "jd_requirements": SAMPLE_JD_REQS,
        "matched_experiences": SAMPLE_EVIDENCE,
        "gaps": [],
        "draft_content": SAMPLE_DRAFT,
        "star_stories": [],
        "review_result": None,
        "revision_count": 0,
        "saved_resume_path": "",
        "saved_jd_path": "",
        "saved_stories_path": "",
        "tracker_updated": False,
        "final_output": "",
    }

    result = quality_reviewer_node(test_state)
    rv = result["review_result"]
    print("\n" + "=" * 60)
    print(f"Passed         : {rv['passed']}")
    print(f"Coverage       : {rv['keyword_coverage']:.0%}")
    print(f"Issues         : {rv['faithfulness_issues']}")
    print(f"Feedback       : {rv['feedback']}")
    print(f"Revision count : {result['revision_count']}")
