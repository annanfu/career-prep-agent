"""Node 2: Resume Tailor — three-step: persona, tailor resume, generate STAR stories."""

import json
import re
from pathlib import Path

from dotenv import load_dotenv

from src.llm import get_fast_llm, get_quality_llm
from src.state import GraphState

load_dotenv()

TAILOR_PROMPT_PATH = Path("src/prompts/resume_tailor.txt")
STAR_PROMPT_PATH = Path("src/prompts/star_story.txt")
PERSONA_PROMPT_PATH = Path("src/prompts/persona.txt")
PERSONA_KB_PATH = Path("knowledge_base/persona/persona.md")


def _fix_combined_titles(base_resume: str, target_role: str) -> str:
    """Replace combined job titles with a single title closest to target.

    Finds lines like '*X / Y / Z (Context)*' and picks the title
    that best matches target_role, keeping the parenthetical.

    Args:
        base_resume: Full resume markdown text.
        target_role: Target job title from user input.

    Returns:
        Resume with combined titles replaced.
    """
    lines = base_resume.split("\n")
    result: list[str] = []
    target_lower = target_role.lower()

    for line in lines:
        # Match italic job title lines with slashes: *Title1 / Title2 (Context)*
        if line.startswith("*") and "/" in line and line.endswith("*"):
            inner = line.strip("*").strip()
            # Extract parenthetical if present
            paren = ""
            if "(" in inner and inner.endswith(")"):
                paren_start = inner.rfind("(")
                paren = inner[paren_start:]
                inner = inner[:paren_start].strip()

            # Split titles by /
            titles = [t.strip() for t in inner.split("/")]

            # Pick best match: find title with most word overlap to target
            best = titles[0]
            best_score = 0
            target_words = set(target_lower.split())
            for title in titles:
                title_words = set(title.lower().split())
                score = len(target_words & title_words)
                if score > best_score:
                    best_score = score
                    best = title
            # If no word overlap, pick first title
            result.append(
                f"*{best} {paren}*" if paren else f"*{best}*",
            )
        else:
            result.append(line)

    return "\n".join(result)


def _filter_projects(base_resume: str, jd_requirements: dict) -> str:
    """Pre-filter projects by keyword overlap before sending to LLM.

    Keeps a project if ANY JD keyword appears in its header or bullets.
    This ensures the LLM cannot accidentally drop relevant projects.

    Args:
        base_resume: Full base resume markdown text.
        jd_requirements: Parsed JD dict.

    Returns:
        Resume with only matching projects kept.
    """
    # Collect all JD keywords (lowercase)
    _STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "for", "of",
        "to", "in", "on", "at", "by", "with", "from", "into",
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will",
        "can", "may", "that", "this", "these", "those",
        "not", "all", "any", "each", "both", "such", "new",
        "our", "you", "your", "they", "them", "their", "its",
        "who", "what", "which", "how", "when", "where", "as",
        "well", "also", "more", "very", "real", "best",
    }
    # Short tech terms that should never be filtered by length
    _SHORT_TECH = {
        "ai", "ml", "go", "c", "r", "sql", "aws", "gcp",
        "api", "ci", "cd", "ci/cd", "ux", "ui", "qa",
        "llm", "rag", "nlp", "etl", "sdk", "jvm",
    }
    jd_keywords: set[str] = set()
    for field in (
        "required_skills", "preferred_skills",
    ):
        for item in jd_requirements.get(field, []):
            # Keep full multi-word skills as-is
            jd_keywords.add(item.lower())
            # Split and add individual words
            for word in item.lower().split():
                if word in _SHORT_TECH:
                    jd_keywords.add(word)
                elif len(word) >= 4 and word not in _STOP_WORDS:
                    jd_keywords.add(word)

    # Also add common variations
    extras = set()
    for kw in jd_keywords:
        extras.add(kw.replace("-", " "))
        extras.add(kw.replace(" ", "-"))
    jd_keywords.update(extras)

    lines = base_resume.split("\n")
    result_lines: list[str] = []
    in_projects = False
    current_project: list[str] = []
    current_header = ""

    for line in lines:
        if line.strip().startswith("## Projects"):
            in_projects = True
            result_lines.append(line)
            continue

        if in_projects and line.strip().startswith("## "):
            # End of projects section — flush last project
            if current_project:
                block = "\n".join(current_project).lower()
                if any(kw in block for kw in jd_keywords):
                    result_lines.extend(current_project)
            in_projects = False
            result_lines.append(line)
            continue

        if in_projects:
            if line.strip().startswith("**") and "|" in line:
                # New project header — flush previous
                if current_project:
                    block = "\n".join(current_project).lower()
                    if any(kw in block for kw in jd_keywords):
                        result_lines.extend(current_project)
                current_project = [line]
            elif line.strip() == "---" and current_project:
                current_project.append(line)
            elif current_project:
                current_project.append(line)
            else:
                result_lines.append(line)
        else:
            result_lines.append(line)

    # Flush last project if still in projects section
    if current_project:
        block = "\n".join(current_project).lower()
        if any(kw in block for kw in jd_keywords):
            result_lines.extend(current_project)

    return "\n".join(result_lines)


def _generate_persona(
    jd_requirements: dict,
    target_role: str,
) -> str:
    """Generate a 1-2 sentence persona summary tailored to the JD.

    Reads the base persona from knowledge_base/persona/persona.md and
    uses an LLM to craft a professional summary aligned with the target role.

    Args:
        jd_requirements: Parsed JD dict.
        target_role: Target job title.

    Returns:
        Persona summary string (1-2 sentences), or empty string on failure.
    """
    if not PERSONA_KB_PATH.exists():
        print("  [Persona] persona.md not found — skipping.")
        return ""

    base_persona = PERSONA_KB_PATH.read_text(encoding="utf-8")
    prompt_template = PERSONA_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = (
        prompt_template
        .replace("{base_persona}", base_persona)
        .replace("{jd_requirements}", json.dumps(jd_requirements, indent=2))
        .replace("{target_role}", target_role)
    )

    llm = get_fast_llm(temperature=0.3)
    response = llm.invoke(prompt)
    summary = (response.content or "").strip()
    # Strip quotes if the model wraps the output
    summary = summary.strip('"').strip("'")
    return summary


def _inject_persona_into_resume(resume: str, persona_summary: str) -> str:
    """Insert a persona summary line at the top of the tailored resume.

    Places the summary right after the name/contact header (first two lines),
    before the first section heading.

    Args:
        resume: Tailored resume markdown text.
        persona_summary: 1-2 sentence professional summary.

    Returns:
        Resume with persona summary injected.
    """
    if not persona_summary:
        return resume

    lines = resume.split("\n")
    insert_idx = 0

    # Find the first section heading (## ...) after the name/contact block
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("## ") and i > 0:
            insert_idx = i
            break

    if insert_idx == 0:
        # Fallback: insert after the first non-empty line
        for i, line in enumerate(lines):
            if line.strip() and i > 0:
                insert_idx = i + 1
                break

    # Insert persona summary as italic text with a blank line before and after
    persona_block = [f"*{persona_summary}*", ""]
    lines = lines[:insert_idx] + persona_block + lines[insert_idx:]
    return "\n".join(lines)


def _tailor_resume(
    base_resume: str,
    jd_requirements: dict,
    review_feedback: str,
    target_role: str,
) -> tuple[str, str]:
    """Call LLM to tailor the resume. No RAG chunks — base resume only.

    Args:
        base_resume: Full base resume markdown text.
        jd_requirements: Parsed JD dict.
        review_feedback: Feedback from quality reviewer, or empty string.
        target_role: Target job title (e.g. "Software Engineer").

    Returns:
        Tuple of (tailored_resume, reasoning).
    """
    # Pre-process: fix combined job titles in base resume
    base_resume = _fix_combined_titles(base_resume, target_role)
    # Pre-filter projects by JD keyword overlap
    base_resume = _filter_projects(base_resume, jd_requirements)

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

    llm = get_quality_llm(temperature=0)
    response = llm.invoke(prompt)
    raw = (response.content or "").strip()

    # Parse reasoning and resume
    separator = "---RESUME_START---"
    reasoning = ""
    if separator in raw:
        parts = raw.split(separator, 1)
        reasoning = parts[0].strip()
        raw = parts[1].strip()
    else:
        # Fallback: split at the first Markdown heading that looks like
        # the resume start (e.g. "# Annan Fu" or "# Name")
        heading_match = re.search(r"^(# .+)$", raw, re.MULTILINE)
        if heading_match and heading_match.start() > 0:
            reasoning = raw[: heading_match.start()].strip()
            raw = raw[heading_match.start():]

    # Strip markdown code fences if model wraps output
    raw = re.sub(r"^```(?:markdown)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    # Strip trailing notes/reasoning that leak into the resume.
    # Matches patterns like "**Notes:**", "Notes:", "---\nNotes",
    # "**Reasoning:**", etc. at the end of the output.
    raw = re.sub(
        r"\n+[-*\s]*\*{0,2}(?:Notes?|Reasoning|Key Changes)"
        r"(?:\s*:|\*{0,2}\s*:).*",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    ).rstrip()

    # Post-process: fix combined titles if LLM re-introduced them
    raw = _fix_combined_titles(raw, target_role)

    return raw, reasoning


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
    """LangGraph Node 2: persona generation, resume tailoring, STAR stories.

    Three LLM calls:
    1. Generate persona summary — base persona + JD (fast LLM, first pass only)
    2. Tailor resume — base_resume + JD only (no RAG chunks, avoids hallucination)
    3. Generate STAR stories — matched_experiences + JD (RAG evidence used here)

    Args:
        state: Current GraphState. Requires base_resume_content,
            matched_experiences, jd_requirements. Optionally uses
            review_result for revision feedback.

    Returns:
        Dict with keys: persona_summary, draft_content, star_stories, tailor_reasoning.
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

    # Step 1: Generate persona summary (only on first pass, not revisions)
    persona_summary = state.get("persona_summary", "")
    if not persona_summary:
        print("  [Persona] Generating persona summary...")
        persona_summary = _generate_persona(
            state["jd_requirements"],
            target_role,
        )
        if persona_summary:
            print(f"  [Persona] Summary: {persona_summary}")
        else:
            print("  [Persona] No summary generated.")

    # Step 2: Tailor resume
    tailored, reasoning = _tailor_resume(
        state["base_resume_content"],
        state["jd_requirements"],
        review_feedback,
        target_role,
    )
    if reasoning:
        print(f"  Reasoning:\n{reasoning}")

    # Step 3: Inject persona summary at top of tailored resume
    if persona_summary:
        tailored = _inject_persona_into_resume(tailored, persona_summary)

    print(f"  Draft length: {len(tailored)} chars")

    return {
        "persona_summary": persona_summary,
        "draft_content": tailored,
        "star_stories": [],
        "tailor_reasoning": reasoning,
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
            "persona_summary": "",
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
