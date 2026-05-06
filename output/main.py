import sys
import uuid
from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from langchain_mcp_adapters.client import MultiServerMCPClient

from final_practice_guided.output.graph import build_graph


def load_notion_tools() -> list:
    import json
    import asyncio

    with open("mcp_config.json") as f:
        cfg = json.load(f)

    async def _load():
        async with MultiServerMCPClient(cfg) as client:
            return client.get_tools()

    return asyncio.run(_load())


def run(page_id: str) -> None:
    notion_tools = load_notion_tools()
    graph = build_graph()
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "notion_tools": notion_tools,
        }
    }

    initial = {
        "page_id": page_id,
        "user_story": "",
        "draft_test_cases": "",
        "reviewed_test_cases": "",
        "quality_passed": False,
        "quality_feedback": "",
        "human_approved": False,
        "human_feedback": "",
        "published": False,
        "retry_count": 0,
    }

    print(f"\n{'='*60}\nProcessing Notion page: {page_id}\n{'='*60}")

    try:
        for step in graph.stream(initial, config=thread, stream_mode="updates"):
            node_name, updates = next(iter(step.items()))
            print(f"  ✓ {node_name} wrote: {list(updates.keys())}")
    except GraphInterrupt as exc:
        while True:
            response = input(f"\n[REVIEW] {exc.args[0]}\n> ").strip()
            try:
                for step in graph.stream(
                    Command(resume=response), config=thread, stream_mode="updates"
                ):
                    node_name, updates = next(iter(step.items()))
                    print(f"  ✓ {node_name} wrote: {list(updates.keys())}")
                break
            except GraphInterrupt as next_exc:
                exc = next_exc

    final = graph.get_state(thread).values
    print(f"\n  published = {final.get('published')}")


if __name__ == "__main__":
    page_id = sys.argv[1] if len(sys.argv) > 1 else ""
    if not page_id:
        print("Usage: python main.py <notion-page-id>")
        sys.exit(1)
    run(page_id)
