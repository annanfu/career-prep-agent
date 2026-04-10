"""Sub-agent B: Company Researcher — web search for company intelligence."""

from typing import List

from src.state import InterviewPrepState
from src.utils.web_search import search_web


def company_researcher_node(state: InterviewPrepState) -> dict:
    """Research the target company via web search.

    Searches for recent news, tech blog posts, engineering culture,
    products, and mission. Compiles into a structured company brief.

    Args:
        state: InterviewPrepState with company and role populated.

    Returns:
        Dict with 'company_brief' key containing structured research.
    """
    company = state.get("company", "Unknown")
    role = state.get("role", "")
    print(f"\n[Sub-agent B] Researching {company}...")

    # Use role context to disambiguate common company names
    queries = [
        f'"{company}" company about us products services {role}',
        f'"{company}" company culture values mission careers',
        f'"{company}" Glassdoor reviews work culture',
        f'"{company}" recent news 2025 2026',
        f'"{company}" engineering blog tech stack',
        f'"{company}" {role} jobs careers',
    ]

    all_results: List[dict] = []
    for query in queries:
        hits = search_web(query, max_results=3)
        all_results.extend(hits)

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique: List[dict] = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique.append(r)

    brief = {
        "company": company,
        "role": role,
        "research_results": unique,
        "num_sources": len(unique),
    }

    print(f"  Found {len(unique)} unique sources about {company}")
    return {"company_brief": brief}


if __name__ == "__main__":
    test_state: InterviewPrepState = {
        "resume_content": "",
        "jd_data": {},
        "company": "Planet Labs",
        "role": "Software Engineer",
        "deep_experiences": [],
        "company_brief": {},
        "interview_questions": [],
        "interview_prep_output": "",
        "saved_prep_path": "",
    }

    result = company_researcher_node(test_state)
    brief = result["company_brief"]
    print(f"\nCompany brief for {brief['company']}:")
    print(f"  Sources: {brief['num_sources']}")
    for r in brief["research_results"][:3]:
        print(f"  - {r['title'][:60]}")
