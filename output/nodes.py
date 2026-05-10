from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from final_practice_guided.output.state import QAState

_llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)


def _call(system: str, user: str) -> str:
    return _llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content


# ---------------------------------------------------------------------------
# Agent 1 — Context Agent
# ---------------------------------------------------------------------------

def context_agent_node(state: QAState, config: dict) -> dict:
    """Agent 1 — owns: user_story"""
    fs_tools = config["configurable"]["fs_tools"]

    # Find the read_file tool
    read_tool = None
    for tool in fs_tools:
        if "read_file" in tool.name.lower() or tool.name.lower() == "read_file":
            read_tool = tool
            break

    # Broader fallback: any tool with "read" in its name
    if read_tool is None:
        for tool in fs_tools:
            if "read" in tool.name.lower():
                read_tool = tool
                break

    if read_tool is None:
        raise ValueError(
            f"Could not find a read_file tool. Available: {[t.name for t in fs_tools]}"
        )

    result = read_tool.invoke({"path": state["file_path"]})

    # Normalise result to a plain string
    if hasattr(result, "content"):
        user_story = str(result.content)
    else:
        user_story = str(result)

    return {"user_story": user_story}


# ---------------------------------------------------------------------------
# Agent 2 — QA Agent
# ---------------------------------------------------------------------------

def qa_agent_node(state: QAState) -> dict:
    """Agent 2 — owns: draft_test_cases"""
    system = (
        "You are a senior QA engineer. Write thorough test cases for the user story below.\n"
        "Use this format for every test case:\n\n"
        "### TC-001: <Title>\n"
        "- **Type**: Manual | Automated\n"
        "- **Priority**: High | Medium | Low\n"
        "- **Preconditions**: <list or \"None\">\n"
        "- **Steps**:\n"
        "  1. ...\n"
        "- **Expected Result**: <what should happen>\n\n"
        "If quality feedback or human feedback is present, address every point explicitly."
    )

    user_parts = [f"User Story:\n{state['user_story']}"]

    if state.get("quality_feedback"):
        user_parts.append(f"\nQuality Feedback to Address:\n{state['quality_feedback']}")

    if state.get("human_feedback"):
        user_parts.append(f"\nHuman Feedback to Address:\n{state['human_feedback']}")

    user_message = "\n".join(user_parts)
    draft = _call(system, user_message)
    return {"draft_test_cases": draft}


# ---------------------------------------------------------------------------
# Agent 3 — Quality Agent
# ---------------------------------------------------------------------------

def quality_agent_node(state: QAState) -> dict:
    """Agent 3 — owns: quality_passed, quality_feedback, reviewed_test_cases, retry_count"""
    system = (
        "You are a QA lead performing a quality gate review.\n"
        "Check the test cases for:\n"
        "- Duplicate or missing TC IDs\n"
        "- Vague or incomplete steps\n"
        "- Missing expected results\n"
        "- Edge cases that are clearly implied by the user story but not covered\n\n"
        "If the test cases pass all checks, respond with exactly:\n"
        "PASSED\n\n"
        "If they do not pass, respond with:\n"
        "FAILED\n"
        "<bullet list of specific issues to fix>\n\n"
        "Do not rewrite the test cases. Only judge and explain."
    )

    response = _call(system, state["draft_test_cases"])
    new_retry_count = state["retry_count"] + 1

    if response.strip().startswith("PASSED"):
        return {
            "quality_passed": True,
            "quality_feedback": "",
            "reviewed_test_cases": state["draft_test_cases"],
            "retry_count": new_retry_count,
        }
    else:
        issues = response.strip()
        if issues.upper().startswith("FAILED"):
            issues = issues[len("FAILED"):].strip()
        return {
            "quality_passed": False,
            "quality_feedback": issues,
            "reviewed_test_cases": state.get("reviewed_test_cases", ""),
            "retry_count": new_retry_count,
        }


# ---------------------------------------------------------------------------
# Human Review Node (HITL)
# ---------------------------------------------------------------------------

def human_review_node(state: QAState) -> dict:
    """HITL gate — owns: human_approved, human_feedback"""
    print("\n" + "=" * 60)
    print("REVIEWED TEST CASES:")
    print("=" * 60)
    print(state["reviewed_test_cases"])
    print("=" * 60 + "\n")

    response = interrupt("Approve these test cases? Type 'approve' or describe changes: ")

    normalized = response.lower().strip() if response else ""

    if normalized in ("approve", "ok", "yes", ""):
        return {"human_approved": True, "human_feedback": ""}
    else:
        return {"human_approved": False, "human_feedback": response}


# ---------------------------------------------------------------------------
# Publish Node
# ---------------------------------------------------------------------------

def publish_node(state: QAState, config: dict) -> dict:
    """Writes the approved test cases to a local output file."""
    fs_tools = config["configurable"]["fs_tools"]

    print(f"[publish_node] Available tools: {[t.name for t in fs_tools]}")

    # Find the write_file tool
    write_tool = None
    for tool in fs_tools:
        if "write_file" in tool.name.lower() or tool.name.lower() == "write_file":
            write_tool = tool
            break

    # Broader fallback: any tool with "write" in its name
    if write_tool is None:
        for tool in fs_tools:
            if "write" in tool.name.lower():
                write_tool = tool
                break

    if write_tool is None:
        raise ValueError(
            f"Could not find a write_file tool. Available: {[t.name for t in fs_tools]}"
        )

    write_tool.invoke(
        {
            "path": "output_test_cases.md",
            "content": state["reviewed_test_cases"],
        }
    )

    return {"published": True}
