"""Sub-agent C: Interview Question Researcher — find likely interview Qs."""

import json
import re
from typing import List

from src.llm import get_fast_llm
from src.state import InterviewPrepState
from src.utils.web_search import search_web


def question_researcher_node(state: InterviewPrepState) -> dict:
    """Research likely interview questions for the target company + role.

    Searches Glassdoor, Blind, LeetCode Discuss for interview questions,
    then uses LLM to categorize them into behavioral, technical, and
    system design buckets.

    Args:
        state: InterviewPrepState with company and role populated.

    Returns:
        Dict with 'interview_questions' key.
    """
    company = state.get("company", "Unknown")
    role = state.get("role", "")
    print(f"\n[Sub-agent C] Researching interview questions for "
          f"{company} {role}...")

    queries = [
        f"{company} {role} interview questions Glassdoor",
        f"{company} {role} interview experience Blind",
        f"{company} software engineer interview LeetCode",
        f"{company} {role} behavioral interview questions",
    ]

    all_results: List[dict] = []
    for query in queries:
        hits = search_web(query, max_results=3)
        all_results.extend(hits)

    if not all_results:
        print("  No web results found, generating generic questions")
        return {"interview_questions": _generate_generic(role)}

    # Combine snippets and ask LLM to extract + categorize questions
    snippets = "\n\n".join(
        f"Source: {r['title']}\n{r['content'][:500]}"
        for r in all_results[:10]
    )

    prompt = (
        f"Based on these interview experience snippets for {company} "
        f"({role}), extract specific interview questions that were asked."
        f"\n\nSnippets:\n{snippets}\n\n"
        "Return ONLY a JSON array. Each element:\n"
        '{"question": "...", "category": "behavioral|technical|system_design"}\n'
        "If a snippet doesn't contain a clear question, skip it.\n"
        "Include 10-15 questions total. No markdown fences."
    )

    llm = get_fast_llm(temperature=0)
    response = llm.invoke(prompt)
    raw = (response.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        questions = json.loads(raw)
        if not isinstance(questions, list):
            questions = []
    except json.JSONDecodeError:
        questions = _generate_generic(role)

    print(f"  Found {len(questions)} interview questions")
    return {"interview_questions": questions}


def _generate_generic(role: str) -> List[dict]:
    """Return generic interview questions when web search fails.

    Args:
        role: Target role title.

    Returns:
        List of question dicts with category.
    """
    return [
        {"question": "Tell me about yourself.", "category": "behavioral"},
        {"question": "Why do you want to work here?", "category": "behavioral"},
        {"question": f"Describe a challenging {role} project.", "category": "behavioral"},
        {"question": "Tell me about a time you disagreed with a teammate.", "category": "behavioral"},
        {"question": "How do you handle tight deadlines?", "category": "behavioral"},
        {"question": "Design a URL shortening service.", "category": "system_design"},
        {"question": "How would you design a rate limiter?", "category": "system_design"},
        {"question": "Explain the trade-offs between SQL and NoSQL.", "category": "technical"},
        {"question": "What is your debugging process?", "category": "technical"},
        {"question": "Explain REST vs gRPC.", "category": "technical"},
    ]


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

    result = question_researcher_node(test_state)
    qs = result["interview_questions"]
    print(f"\nQuestions found: {len(qs)}")
    for q in qs[:5]:
        print(f"  [{q['category']}] {q['question']}")
