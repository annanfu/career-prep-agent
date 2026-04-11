"""LangGraph StateGraph orchestrating the Phase 1 resume tailoring pipeline."""

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src.agents.jd_analyzer_matcher import jd_analyzer_matcher_node
from src.agents.quality_reviewer import quality_reviewer_node
from src.agents.resume_tailor import resume_tailor_node
from src.agents.save_and_track import save_and_track_node
from src.state import GraphState

MAX_REVISIONS = 2


def _route_after_review(state: GraphState) -> str:
    """Conditional edge: proceed to save or loop back to tailor.

    Passes when review_result.passed is True OR revision_count >= MAX.

    Args:
        state: Current GraphState after quality_reviewer_node runs.

    Returns:
        'save_and_track' or 'resume_tailor'.
    """
    review = state.get("review_result")
    revision_count = state.get("revision_count", 0)

    if (review and review["passed"]) or revision_count >= MAX_REVISIONS:
        return "save_and_track"
    return "resume_tailor"


def _set_final_output(state: GraphState) -> dict:
    """Intermediate node: copy draft_content → final_output before saving.

    Args:
        state: Current GraphState.

    Returns:
        Dict with final_output set.
    """
    return {"final_output": state["draft_content"]}


def build_graph() -> StateGraph:
    """Construct and compile the Phase 1 LangGraph StateGraph.

    Returns:
        Compiled LangGraph app ready for .invoke().
    """
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("jd_analyzer_matcher", jd_analyzer_matcher_node)
    graph.add_node("resume_tailor", resume_tailor_node)
    graph.add_node("quality_reviewer", quality_reviewer_node)
    graph.add_node("set_final_output", _set_final_output)
    graph.add_node("save_and_track", save_and_track_node)

    # Edges
    graph.add_edge(START, "jd_analyzer_matcher")
    graph.add_edge("jd_analyzer_matcher", "resume_tailor")
    graph.add_edge("resume_tailor", "quality_reviewer")

    # Conditional: pass → set_final_output → save | fail → resume_tailor
    graph.add_conditional_edges(
        "quality_reviewer",
        _route_after_review,
        {
            "save_and_track": "set_final_output",
            "resume_tailor": "resume_tailor",
        },
    )
    graph.add_edge("set_final_output", "save_and_track")
    graph.add_edge("save_and_track", END)

    return graph.compile()


# Compiled graph — importable by main.py and tests
app = build_graph()


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

    # Select base resume
    resume_path = Path("base_resume/resume_master.md")
    if not resume_path.exists():
        print(f"Base resume not found at {resume_path}. Exiting.")
        raise SystemExit(1)

    base_content = resume_path.read_text(encoding="utf-8")

    initial_state: GraphState = {
        "jd_raw": SAMPLE_JD,
        "jd_url": None,
        "target_role": "Software Engineer",
        "company_name": "Stripe",
        "base_resume_path": str(resume_path),
        "base_resume_content": base_content,
        "jd_requirements": None,
        "matched_experiences": [],
        "gaps": [],
        "draft_content": "",
        "star_stories": [],
        "tailor_reasoning": "",
        "review_result": None,
        "revision_count": 0,
        "change_summary": {},
        "saved_resume_path": "",
        "saved_jd_path": "",
        "saved_stories_path": "",
        "tracker_updated": False,
        "final_output": "",
    }

    print("Running full pipeline...\n")
    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    jd_req = final_state.get("jd_requirements", {})
    review = final_state.get("review_result", {})

    print(f"Company       : {jd_req.get('company')}")
    print(f"Role          : {jd_req.get('role')}")
    print(
        f"Fit score     : {jd_req.get('fit_score')}/5"
        f" ({jd_req.get('fit_grade')})"
    )
    print(
        f"Keyword cov.  : "
        f"{review.get('keyword_coverage', 0):.0%}"
    )
    print(f"Review passed : {review.get('passed')}")
    print(f"Revisions     : {final_state.get('revision_count')}")
    print(f"Resume saved  : {final_state.get('saved_resume_path')}")
    print(f"JD saved      : {final_state.get('saved_jd_path')}")
    print(
        f"STAR stories  : "
        f"{len(final_state.get('star_stories', []))} generated"
    )
    print(f"Tracker       : {final_state.get('tracker_updated')}")
    print("\nFINAL RESUME (first 800 chars):")
    print(final_state.get("final_output", "")[:800])
