"""LangGraph StateGraph for Phase 2: Interview Prep pipeline.

Architecture: Orchestrator -> fan-out to 3 parallel sub-agents
-> fan-in -> Content Generator.
"""

import json
import re
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src.agents.interview.company_researcher import company_researcher_node
from src.agents.interview.deep_retriever import deep_retriever_node
from src.agents.interview.question_researcher import question_researcher_node
from src.llm import get_quality_llm
from src.state import InterviewPrepState

PREP_PROMPT_PATH = Path("src/prompts/interview_prep.txt")
OUTPUT_DIR = Path("output/interview_prep")


def _content_generator_node(state: InterviewPrepState) -> dict:
    """Generate the final interview prep document from all sub-agent outputs.

    Args:
        state: InterviewPrepState with all sub-agent fields populated.

    Returns:
        Dict with 'interview_prep_output' and 'saved_prep_path'.
    """
    print("\n[Content Generator] Building interview prep document...")

    company = state.get("company", "Unknown")
    role = state.get("role", "")
    jd_data = state.get("jd_data", {})

    # Format BQ templates as readable text
    bq_templates = state.get("bq_templates", {})
    bq_text = "\n\n".join(
        f"### {k.replace('_', ' ').title()}\n{v}"
        for k, v in bq_templates.items() if v
    ) or "No templates found in knowledge base."

    interview_notes = state.get("interview_notes", "") or "None."

    prompt_template = PREP_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (
        prompt_template
        .replace("{company}", company)
        .replace("{role}", role)
        .replace(
            "{resume_content}",
            state.get("resume_content", ""),
        )
        .replace(
            "{jd_requirements}",
            json.dumps(jd_data, indent=2)[:3000],
        )
        .replace(
            "{deep_experiences}",
            json.dumps(
                state.get("deep_experiences", []), indent=2
            )[:12000],
        )
        .replace("{bq_templates}", bq_text[:5000])
        .replace("{interview_notes}", interview_notes[:3000])
        .replace(
            "{company_brief}",
            json.dumps(
                state.get("company_brief", {}), indent=2
            )[:4000],
        )
        .replace(
            "{interview_questions}",
            json.dumps(
                state.get("interview_questions", []), indent=2
            )[:3000],
        )
    )

    llm = get_quality_llm(temperature=0.3)
    response = llm.invoke(prompt)
    prep_doc = (response.content or "").strip()

    # Strip markdown fences if present
    prep_doc = re.sub(r"^```(?:markdown)?\s*", "", prep_doc)
    prep_doc = re.sub(r"\s*```$", "", prep_doc)

    # Save to file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_company = re.sub(r"[^\w\s-]", "", company).strip().replace(" ", "_")
    safe_role = re.sub(r"[^\w\s-]", "", role).strip().replace(" ", "_")
    filename = f"{safe_company}_{safe_role}_prep.md"
    save_path = OUTPUT_DIR / filename
    save_path.write_text(prep_doc, encoding="utf-8")

    print(f"  Prep document: {len(prep_doc)} chars")
    print(f"  Saved to: {save_path}")

    return {
        "interview_prep_output": prep_doc,
        "saved_prep_path": str(save_path),
    }


def build_interview_graph() -> StateGraph:
    """Construct and compile the Phase 2 interview prep LangGraph.

    Architecture:
        START → [deep_retriever, company_researcher, question_researcher]
              → content_generator → END

    Returns:
        Compiled LangGraph app ready for .invoke().
    """
    graph = StateGraph(InterviewPrepState)

    # Register nodes
    graph.add_node("deep_retriever", deep_retriever_node)
    graph.add_node("company_researcher", company_researcher_node)
    graph.add_node("question_researcher", question_researcher_node)
    graph.add_node("content_generator", _content_generator_node)

    # Fan-out: START → 3 parallel sub-agents
    graph.add_edge(START, "deep_retriever")
    graph.add_edge(START, "company_researcher")
    graph.add_edge(START, "question_researcher")

    # Fan-in: all 3 → content_generator
    graph.add_edge("deep_retriever", "content_generator")
    graph.add_edge("company_researcher", "content_generator")
    graph.add_edge("question_researcher", "content_generator")

    # content_generator → END
    graph.add_edge("content_generator", END)

    return graph.compile()


# Compiled graph — importable by main.py
interview_app = build_interview_graph()


if __name__ == "__main__":
    sample_resume = """
# Sample Candidate

## Work Experience

**Northeastern University** — Vancouver, BC | Jan 2025 – Dec 2025
*Software Engineer (Research Assistant)*

- Built an end-to-end ML system for Android malware detection using Python
- Designed a scalable ETL pipeline to ingest raw Android APKs

## Projects

**Cloud-based Distributed Album Store System** | AWS, Java

- Built a distributed album store with microservices and RESTful APIs
- Implemented async event-driven processing with AWS SQS/SNS
"""

    sample_jd = {
        "company": "Planet Labs",
        "role": "Software Engineer",
        "required_skills": ["Python", "API design", "PostgreSQL",
                            "Docker", "Linux"],
        "key_responsibilities": [
            "Maintain satellite operation services",
            "Implement HTTP APIs",
            "Deploy critical infrastructure",
        ],
    }

    initial: InterviewPrepState = {
        "resume_content": sample_resume,
        "jd_data": sample_jd,
        "company": "Planet Labs",
        "role": "Software Engineer",
        "deep_experiences": [],
        "bq_templates": {},
        "interview_notes": "",
        "company_brief": {},
        "interview_questions": [],
        "interview_prep_output": "",
        "saved_prep_path": "",
    }

    print("Running interview prep pipeline...\n")
    final = interview_app.invoke(initial)

    print("\n" + "=" * 60)
    print(f"Company: {final.get('company')}")
    print(f"Role: {final.get('role')}")
    print(f"Deep experiences: {len(final.get('deep_experiences', []))}")
    print(f"Company brief sources: "
          f"{final.get('company_brief', {}).get('num_sources', 0)}")
    print(f"Interview questions: "
          f"{len(final.get('interview_questions', []))}")
    print(f"Prep saved: {final.get('saved_prep_path')}")
    print("\nPREP DOCUMENT (first 1000 chars):")
    print(final.get("interview_prep_output", "")[:1000])
