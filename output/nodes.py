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
    notion_tools = config["configurable"]["notion_tools"]

    # Find the page-fetch tool: prefer names containing "retrieve" and "page",
    # but exclude tools whose names imply creation or listing.
    fetch_tool = None
    for tool in notion_tools:
        name = tool.name.lower()
        if ("retrieve" in name or "get" in name) and "page" in name:
            fetch_tool = tool
            break

    # Broader fallback: any tool with "retrieve" in its name
    if fetch_tool is None:
        for tool in notion_tools:
            if "retrieve" in tool.name.lower():
                fetch_tool = tool
                break

    if fetch_tool is None:
        raise ValueError(
            f"Could not find a page-fetch tool. Available: {[t.name for t in notion_tools]}"
        )

    result = fetch_tool.invoke({"page_id": state["page_id"]})

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

def _chunk_text(text: str, size: int = 2000) -> list[dict]:
    """Split text into Notion paragraph blocks (max 2000 chars each)."""
    blocks = []
    for i in range(0, len(text), size):
        chunk = text[i : i + size]
        blocks.append(
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": chunk}}]
                },
            }
        )
    return blocks


def publish_node(state: QAState, config: dict) -> dict:
    """Creates a child Notion page with the approved test cases."""
    notion_tools = config["configurable"]["notion_tools"]

    print(f"[publish_node] Available tools: {[t.name for t in notion_tools]}")

    # Find the page-create tool
    create_tool = None
    for tool in notion_tools:
        if "create" in tool.name.lower() and "page" in tool.name.lower():
            create_tool = tool
            break

    # Broader fallback: any tool with "create" in its name
    if create_tool is None:
        for tool in notion_tools:
            if "create" in tool.name.lower():
                create_tool = tool
                break

    if create_tool is None:
        raise ValueError(
            f"Could not find a page-create tool. Available: {[t.name for t in notion_tools]}"
        )

    content = state["reviewed_test_cases"]
    children = _chunk_text(content)

    create_tool.invoke(
        {
            "parent": {"page_id": state["page_id"]},
            "properties": {
                "title": [{"text": {"content": "QA Test Cases"}}]
            },
            "children": children,
        }
    )

    return {"published": True}
