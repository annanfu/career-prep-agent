"""Sub-agent A: Deep Experience Retriever.

Enrich resume bullets with KB detail + retrieve BQ templates.
"""

from typing import List

from src.rag.retriever import retrieve_experiences
from src.state import InterviewPrepState


def _extract_bullets(resume_md: str) -> List[str]:
    """Extract bullet-point lines from a Markdown resume.

    Args:
        resume_md: Resume in Markdown format.

    Returns:
        List of bullet text strings (without leading '- ').
    """
    bullets = []
    for line in resume_md.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            text = stripped[2:].strip()
            if len(text) > 20:
                bullets.append(text)
    return bullets


def _retrieve_templates() -> dict:
    """Retrieve BQ answer templates from knowledge base.

    Queries ChromaDB for self-intro, why-company, why-transition,
    weakness, work-style, and open-questions templates.

    Returns:
        Dict with template keys and retrieved text.
    """
    template_queries = {
        "self_intro": "self introduction career history "
                      "Master Computer Science fund accountant",
        "why_company": "why do you want to work company "
                       "values culture what value I bring",
        "why_transition": "why transition software engineering "
                          "career switch motivation",
        "weakness": "weakness get too deep into details "
                    "take time build confidence",
        "work_style": "work style ownership proactive "
                      "responsive team collaboration",
        "open_questions": "encounter something don't know "
                          "break down problem documentation",
    }

    templates = {}
    for key, query in template_queries.items():
        try:
            chunks = retrieve_experiences([query], top_k=2)
            if chunks:
                # Combine top 2 results for richer template
                combined = "\n\n".join(
                    c["content"][:800] for c in chunks
                )
                templates[key] = combined
            else:
                templates[key] = ""
        except RuntimeError:
            templates[key] = ""

    return templates


def _retrieve_interview_notes(
    company: str,
    role: str,
) -> str:
    """Retrieve past interview notes for this or similar companies.

    Args:
        company: Target company name.
        role: Target role title.

    Returns:
        Retrieved interview notes text, or empty string.
    """
    queries = [
        f"interview notes {company} {role}",
        "interview reflection takeaways improvement",
    ]
    try:
        chunks = retrieve_experiences(queries, top_k=3)
        if chunks:
            return "\n\n".join(c["content"][:600] for c in chunks)
    except RuntimeError:
        pass
    return ""


def deep_retriever_node(state: InterviewPrepState) -> dict:
    """Retrieve deep evidence for resume bullets + BQ templates.

    Three retrieval tasks:
    1. For each resume bullet, find detailed KB evidence
    2. Retrieve BQ answer templates (self-intro, why-company, etc.)
    3. Retrieve past interview notes and reflections

    Args:
        state: InterviewPrepState with resume_content populated.

    Returns:
        Dict with 'deep_experiences' key containing all results.
    """
    print("\n[Sub-agent A] Retrieving deep experience evidence...")

    resume = state.get("resume_content", "")
    company = state.get("company", "")
    role = state.get("role", "")
    bullets = _extract_bullets(resume)
    print(f"  Found {len(bullets)} resume bullets to enrich")

    # 1. Bullet-level evidence
    deep_results: List[dict] = []
    for bullet in bullets:
        query = bullet[:200]
        try:
            chunks = retrieve_experiences([query], top_k=3)
        except RuntimeError:
            chunks = []

        if chunks:
            best = chunks[0]
            deep_results.append({
                "bullet": bullet,
                "evidence": best["content"][:500],
                "source_doc": best["source_doc"],
                "has_evidence": True,
            })
        else:
            deep_results.append({
                "bullet": bullet,
                "evidence": "",
                "source_doc": "",
                "has_evidence": False,
            })

    print(f"  Enriched {len(deep_results)} bullets")

    # 2. BQ templates
    print("  Retrieving BQ answer templates...")
    templates = _retrieve_templates()
    print(f"  Found templates for: "
          f"{[k for k, v in templates.items() if v]}")

    # 3. Interview notes
    print("  Retrieving past interview notes...")
    notes = _retrieve_interview_notes(company, role)
    print(f"  Interview notes: {len(notes)} chars")

    return {
        "deep_experiences": deep_results,
        "bq_templates": templates,
        "interview_notes": notes,
    }


if __name__ == "__main__":
    test_state: InterviewPrepState = {
        "resume_content": (
            "## Work Experience\n\n"
            "- Built an end-to-end ML system for malware detection\n"
            "- Designed a scalable ETL pipeline\n"
        ),
        "jd_data": {},
        "company": "Orbis",
        "role": "Software Developer",
        "deep_experiences": [],
        "company_brief": {},
        "interview_questions": [],
        "interview_prep_output": "",
        "saved_prep_path": "",
    }

    result = deep_retriever_node(test_state)
    print(f"\nDeep experiences: {len(result['deep_experiences'])}")
    templates = result.get("bq_templates", {})
    for k, v in templates.items():
        print(f"  {k}: {len(v)} chars")
    print(f"  Interview notes: "
          f"{len(result.get('interview_notes', ''))} chars")
