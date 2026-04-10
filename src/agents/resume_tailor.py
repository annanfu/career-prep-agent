"""Node 2: Resume Tailor — two-step: tailor resume, then generate STAR stories."""

import json
import re
from pathlib import Path

from dotenv import load_dotenv

from src.llm import get_quality_llm
from src.state import GraphState

load_dotenv()

TAILOR_PROMPT_PATH = Path("src/prompts/resume_tailor.txt")
STAR_PROMPT_PATH = Path("src/prompts/star_story.txt")


def _tailor_resume(
    base_resume: str,
    jd_requirements: dict,
    review_feedback: str,
    target_role: str,
) -> str:
    """Call LLM to tailor the resume. No RAG chunks — base resume only.

    Args:
        base_resume: Full base resume markdown text.
        jd_requirements: Parsed JD dict.
        review_feedback: Feedback from quality reviewer, or empty string.
        target_role: Target job title (e.g. "Software Engineer").

    Returns:
        Tailored resume as a Markdown string.
    """
    # Derive industry context from role + JD domain for persona prompt
    domain = jd_requirements.get("domain", "")
    target_role_industry = (
        f"{domain} — {target_role}" if domain else target_role
    )

    prompt_template = TAILOR_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (
        prompt_template
        .replace("{base_resume}", base_resume)
        .replace("{jd_requirements}", json.dumps(jd_requirements, indent=2))
        .replace("{review_feedback}", review_feedback or "None")
        .replace("{target_role}", target_role)
        .replace("{target_role_industry}", target_role_industry)
    )

    llm = get_quality_llm(temperature=0.2)
    response = llm.invoke(prompt)
    raw = (response.content or "").strip()

    # Strip markdown code fences if model wraps output
    raw = re.sub(r"^```(?:markdown)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    return raw


def _generate_star_stories(
    matched_experiences: list[dict],
    jd_requirements: dict,
) -> list[dict]:
    """Call LLM to generate STAR stories from RAG matched experiences.

    Args:
        matched_experiences: List of MatchedExperience dicts from Node 1.
        jd_requirements: Parsed JD dict.

    Returns:
        List of STAR story dicts.
    """
    relevant = [
        m for m in matched_experiences if m.get("relevance_score", 0) >= 0.5
    ]
    if not relevant:
        return []

    prompt_template = STAR_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (
        prompt_template
        .replace("{jd_requirements}", json.dumps(jd_requirements, indent=2))
        .replace("{matched_experiences}", json.dumps(relevant, indent=2))
    )

    llm = get_quality_llm(temperature=0.2)
    response = llm.invoke(prompt)
    raw = (response.content or "").strip()

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        stories = json.loads(raw)
        return stories if isinstance(stories, list) else []
    except json.JSONDecodeError:
        return []


def resume_tailor_node(state: GraphState) -> dict:
    """LangGraph Node 2: tailor the resume and generate STAR stories.

    Two separate LLM calls:
    1. Tailor resume — base_resume + JD only (no RAG chunks, avoids hallucination)
    2. Generate STAR stories — matched_experiences + JD (RAG evidence used here)

    Args:
        state: Current GraphState. Requires base_resume_content,
            matched_experiences, jd_requirements. Optionally uses
            review_result for revision feedback.

    Returns:
        Dict with keys: draft_content (str), star_stories (list[dict]).
    """
    print("\n[Node 2] Tailoring resume (base resume + JD only)...")
    print(f"  base_resume_content length: {len(state.get('base_resume_content', ''))}")
    print(f"  base_resume first 100 chars: {state.get('base_resume_content', '')[:100]!r}")

    review_feedback = ""
    if state.get("review_result") and state["review_result"]:
        review_feedback = state["review_result"].get("feedback", "")
        issues = state["review_result"].get("faithfulness_issues", [])
        if issues:
            review_feedback += (
                "\nFaithfulness issues to fix:\n"
                + "\n".join(f"- {i}" for i in issues)
            )

    target_role = state.get("target_role") or (
        state.get("jd_requirements") or {}
    ).get("role", "the target role")

    tailored = _tailor_resume(
        state["base_resume_content"],
        state["jd_requirements"],
        review_feedback,
        target_role,
    )
    print(f"  Draft length: {len(tailored)} chars")

    # STAR stories skipped for speed — can be generated on-demand
    # via interview prep pipeline instead
    return {
        "draft_content": tailored,
        "star_stories": [],
    }


if __name__ == "__main__":
    from src.state import MatchedExperience, JDRequirements

    sample_jd: JDRequirements = {
        "company": "IREN",
        "role": "Software Engineer",
        "jd_url": None,
        "required_skills": ["Python", "Go", "Kubernetes", "Terraform", "AWS"],
        "preferred_skills": ["Grafana", "Prometheus"],
        "key_responsibilities": [
            "Design, implement, and operate production software systems",
            "Deploy services on Kubernetes",
            "Define infrastructure using Terraform",
        ],
        "soft_skills": ["ownership", "communication"],
        "domain": "AI Cloud / Infrastructure",
        "experience_level": "mid-level",
        "fit_score": 3.8,
        "fit_grade": "B",
    }

    sample_experience: MatchedExperience = {
        "requirement": "Python ETL pipelines",
        "evidence": (
            "Built an end-to-end, time-aware ML system for Android malware "
            "detection over 40K applications using Python, covering data "
            "collection, feature engineering, self-supervised pre-training, "
            "and supervised classification, achieving 98% detection "
            "accuracy."
        ),
        "source_doc": "Self_Supervised_Android_Malware_Detection_KNOWLEDGE.md",
        "relevance_score": 0.85,
        "related_chunks": [],
    }

    base_resume_path = Path("base_resume/resume_master.md")
    if not base_resume_path.exists():
        print("base_resume/resume_master.md not found — skipping test.")
    else:
        base_content = base_resume_path.read_text(encoding="utf-8")

        test_state: GraphState = {
            "jd_raw": "",
            "jd_url": None,
            "base_resume_path": str(base_resume_path),
            "base_resume_content": base_content,
            "jd_requirements": sample_jd,
            "matched_experiences": [sample_experience],
            "gaps": [],
            "draft_content": "",
            "star_stories": [],
            "review_result": None,
            "revision_count": 0,
            "change_summary": {},
            "saved_resume_path": "",
            "saved_jd_path": "",
            "saved_stories_path": "",
            "tracker_updated": False,
            "final_output": "",
        }

        result = resume_tailor_node(test_state)
        print("\n" + "=" * 60)
        print("TAILORED RESUME (first 1200 chars):")
        print(result["draft_content"][:1200])
        print(f"\nSTAR STORIES ({len(result['star_stories'])}):")
        for s in result["star_stories"]:
            print(f"  - {s.get('experience_name')} | {s.get('jd_requirement')}")
