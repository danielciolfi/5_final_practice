# Final Practice - Open Challenge: Design Your Own Multi-Agent LangGraph Pipeline

In this task you'll design and build a LangGraph pipeline **of your own choosing**. Unlike the guided QA agent exercise, this challenge starts with a blank canvas — you pick the problem domain, decompose the work into agents, and wire everything together yourself.

**Minimum requirements for your pipeline:**

| Constraint | Requirement |
| --- | --- |
| Agent nodes | At least **2 specialist agent nodes**, each with a clearly bounded responsibility |
| Human-in-the-loop | Exactly **1 `interrupt()`-based review node** that can loop back |
| MCP server | At least **1 MCP server** providing external context (Notion, GitHub, Slack, filesystem, etc.) |
| State | A single `TypedDict` state shared across all nodes |
| Checkpointer | `MemorySaver` so the interrupt/resume loop persists across calls |

---

## Step 1 — Choose your challenge

Pick one of the example domains below, or propose your own.

<details>
<summary>📋 Example domains (click to expand)</summary>

| Domain | What it does |
| --- | --- |
| **Content pipeline** | Fetches a GitHub issue, searches Notion for related docs, drafts a blog post, human approves, publishes as a Notion page |
| **Incident responder** | Pulls an on-call alert, searches a runbook MCP server, drafts a post-mortem, human confirms severity, posts to Slack |
| **Code review assistant** | Reads a PR diff, queries an internal style-guide MCP, generates review comments, human approves/edits, posts comments to GitHub |
| **Onboarding generator** | Reads a new-hire Jira ticket, fetches team docs via Notion MCP, drafts a personalised onboarding plan, human refines it, saves to Notion |
| **Meeting summariser** | Reads a transcript file, queries a Confluence MCP for project context, writes action items, human edits, sends to email or Slack |
| **Your own idea** | Anything that has a multi-step information-gathering phase, a generation phase, and a human gate |

</details>

> **Think before you continue.**  
> Write down (on paper or in a comment block) a one-paragraph description of:
> - What problem this pipeline solves
> - Who the end user is
> - What "done" looks like — what artifact is produced and where it lands

---

## Step 2 — Define your agent role decomposition

Before writing any code, answer these questions. There is no single right answer — the goal is intentional design.

### 2.1 Identify responsibilities

List every distinct responsibility your pipeline needs to carry out. Think in verbs: *fetch*, *search*, *draft*, *review*, *validate*, *publish*.

> **Checkpoint — ask yourself:**
> - Which responsibilities require external data (API calls, MCP lookups)?
> - Which responsibilities require LLM reasoning?
> - Which responsibilities require a human decision?

### 2.2 Group into agents

Group the responsibilities from 2.1 into **at least two agents**. Each agent must have a single, coherent job that could be described in one sentence.

Use this template for each agent:

```
Agent name  : _______________
One-sentence responsibility : _______________
Inputs from state : _______________
Outputs written to state : _______________
LLM calls it makes : ___ (count and brief description)
External tool calls it makes : ___
```

> **Common mistake to avoid:** do not put data-fetching and generation in the same agent. Keep gathering and reasoning separate so each agent stays testable and replaceable.

### 2.3 Decide the human gate

Answer:
- After which agent does the human review happen, and why at that point?
- What does the human see (which state field is printed)?
- What are the two outcomes — what does "approve" mean and what does "request changes" mean?
- Where does the graph route on each outcome?

### 2.4 Sketch the graph (on paper or ASCII)

Draw the nodes, edges, and the conditional branch before touching code. Example structure:

```
START → [Agent A] → [Agent B] → [Human Review]
                                      │
                    ┌─────────────────┴─────────────────┐
                    │ approved                           │ needs changes
                    ▼                                    ▼
               [Publish] → END                     [Agent B]  (loop)
```

Your graph does not have to match this shape — it just needs to be intentional.

---

## Step 3 — Set up your environment

Once you have your design:

1. Create a new folder for your project.
2. Copy `requirements.txt` as a starting point and add any extra packages your domain needs.
3. Create a `.env` file with the API keys your pipeline will use.

Minimum `requirements.txt`:

```
langchain>=0.3
langchain-anthropic>=0.3
langchain-core>=0.3
langchain-mcp-adapters>=0.1
langgraph>=0.2
python-dotenv>=1.0
httpx>=0.27
```

4. Confirm Node.js is available for MCP servers that use `npx`:

```bash
node --version   # should print v18 or higher
```

---

## Step 4 — Create `state.py`

Define your `TypedDict`. Use your decomposition notes from Step 2 to decide which fields belong here.

Guidelines:
- Every piece of data passed between nodes lives in state — nothing is passed as function arguments.
- Give each field a name that makes its owner obvious (e.g. `context_agent` writes `raw_docs`; `draft_agent` writes `draft_output`).
- Add `retry_count: int` to support the loop safety exit.
- Add `human_approved: bool` and `human_feedback: str` for the HITL node.

```python
from typing import TypedDict

class YourState(TypedDict):
    # inputs
    ...
    # per-agent outputs
    ...
    # HITL fields
    human_approved: bool
    human_feedback: str
    # control
    retry_count: int
    published: bool
```

> **Checkpoint:** Does every field have exactly one node that writes it? If two nodes write the same field, consider splitting them or using `Annotated[list, operator.add]` for accumulation.

---

## Step 5 — Configure your MCP server

### 5.1 Choose your MCP server

Decide which MCP server gives your pipeline the external context it needs.

| Server | npm package | Use case |
| --- | --- | --- |
| Notion | `@notionhq/notion-mcp-server` | Docs, wikis, project pages |
| GitHub | `@modelcontextprotocol/server-github` | Issues, PRs, file contents |
| Filesystem | `@modelcontextprotocol/server-filesystem` | Local files, runbooks |
| Slack | `@modelcontextprotocol/server-slack` | Channel history, threads |
| Brave Search | `@modelcontextprotocol/server-brave-search` | Web search |

### 5.2 Create `mcp_config.json`

```json
{
  "your_server_name": {
    "transport": "stdio",
    "command": "npx",
    "args": ["-y", "<npm-package-name>"],
    "env": {
      "YOUR_API_KEY_ENV_VAR": "your_key_here"
    }
  }
}
```

> **Think:** Which tools does your MCP server expose? Which tool names will your agents call to search and which to fetch full content? (Use `client.get_tools()` in a scratch script to print them out.)

---

## Step 6 — Implement agent nodes in `nodes.py`

### Shared setup

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from state import YourState

_llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)

def _call(system: str, user: str) -> str:
    return _llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content
```

### For each agent node

Follow this pattern:

```python
def your_agent_node(state: YourState, config: dict) -> dict:
    """One-sentence description of this agent's job."""
    # 1. Read the fields this agent needs from `state`
    # 2. Make LLM / tool calls
    # 3. Return only the fields this agent writes
    return {"field_a": ..., "field_b": ...}
```

> **Important constraints:**
> - An agent node must return **only** the state fields it owns (see your decomposition notes).
> - Never read from a field your agent does not own without a clear reason.
> - If your agent calls MCP tools, receive `notion_tools` (or equivalent) from `config["configurable"]`.

### MCP tool call pattern

```python
notion_tools = config["configurable"]["notion_tools"]

# find the right tool by name
search_tool = next(t for t in notion_tools if "search" in t.name)
results = search_tool.invoke({"query": "your query"})

# parse page IDs from results, then fetch each page
fetch_tool = next(t for t in notion_tools if "retrieve" in t.name or "page" in t.name)
page = fetch_tool.invoke({"page_id": page_id})
```

### Human review node

```python
from langgraph.types import interrupt

def human_review_node(state: YourState) -> dict:
    # 1. Print the content the human needs to review
    print(state["the_field_to_review"])
    # 2. Pause and collect feedback
    response = interrupt("Your prompt here — approve or describe changes: ")
    # 3. Route based on response
    approved = response.strip().lower() in {"approve", "ok", "yes", ""}
    return {
        "human_approved": approved,
        "human_feedback": "" if approved else response.strip(),
    }
```

### Publish node

```python
def publish_node(state: YourState) -> dict:
    # Post, save, or send the final artifact
    ...
    return {"published": True}
```

---

## Step 7 — Wire the graph in `graph.py`

```python
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from nodes import agent_a_node, agent_b_node, human_review_node, publish_node
from state import YourState

def review_router(state: YourState) -> str:
    if state.get("human_approved") or state.get("retry_count", 0) >= 3:
        return "publish"
    return "agent_b"   # ← replace with the node that should re-run

def build_graph():
    checkpointer = MemorySaver()
    graph = StateGraph(YourState)

    # Register nodes
    graph.add_node("agent_a", agent_a_node)
    graph.add_node("agent_b", agent_b_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("publish", publish_node)

    # Edges
    graph.set_entry_point("agent_a")
    graph.add_edge("agent_a", "agent_b")
    graph.add_edge("agent_b", "human_review")
    graph.add_conditional_edges("human_review", review_router, {"publish": "publish", "agent_b": "agent_b"})
    graph.add_edge("publish", END)

    return graph.compile(checkpointer=checkpointer)
```

> **Checklist before moving on:**
> - [ ] Every node registered with `add_node`?
> - [ ] Entry point set?
> - [ ] No node is an island (every node has at least one outgoing edge)?
> - [ ] Conditional edge covers all possible return values of the router?
> - [ ] Safety exit via `retry_count >= 3`?

---

## Step 8 — Implement `main.py`

```python
import sys, uuid, json, asyncio
from dotenv import load_dotenv
load_dotenv()

from langgraph.types import Command
from langgraph.errors import GraphInterrupt
from langchain_mcp_adapters.client import MultiServerMCPClient
from graph import build_graph


def load_mcp_tools() -> list:
    with open("mcp_config.json") as f:
        cfg = json.load(f)
    async def _load():
        async with MultiServerMCPClient(cfg) as client:
            return client.get_tools()
    return asyncio.run(_load())


def run(your_input: str) -> None:
    mcp_tools = load_mcp_tools()
    graph = build_graph()
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "notion_tools": mcp_tools,   # rename key to match your agent node
        }
    }

    initial = {
        # populate all fields from your TypedDict
        "human_approved": False,
        "human_feedback": "",
        "published": False,
        "retry_count": 0,
    }

    print(f"\n{'='*60}\nRunning pipeline for: {your_input}\n{'='*60}")

    try:
        for step in graph.stream(initial, config=thread, stream_mode="updates"):
            node_name, updates = next(iter(step.items()))
            print(f"  ✓ {node_name} wrote: {list(updates.keys())}")
    except GraphInterrupt as exc:
        while True:
            response = input(f"\n[REVIEW] {exc.args[0]}\n> ").strip()
            try:
                for step in graph.stream(Command(resume=response), config=thread, stream_mode="updates"):
                    node_name, updates = next(iter(step.items()))
                    print(f"  ✓ {node_name} wrote: {list(updates.keys())}")
                break
            except GraphInterrupt as next_exc:
                exc = next_exc

    final = graph.get_state(thread).values
    print(f"\n  published = {final.get('published')}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "default-input")
```

---

## Step 9 — Run and validate

```bash
python main.py <your-input>
```

Work through this checklist as you test:

| Check | What to look for |
| --- | --- |
| MCP tools load | No import error; `load_mcp_tools()` returns a non-empty list |
| Agent A runs | Prints `✓ agent_a wrote: [...]` with the expected fields |
| Agent B runs | Prints `✓ agent_b wrote: [...]` with the expected fields |
| Interrupt fires | Prompt appears in the terminal; execution pauses |
| Feedback loop | Entering feedback re-runs the generation agent; `retry_count` increments |
| Approval | Entering "approve" routes to the publish node |
| Publish | `published = True` printed at the end |
| LangSmith trace | Open smith.langchain.com and confirm the full run is visible |

---

## Submission checklist

- [ ] `.env` contains all required API keys
- [ ] `mcp_config.json` configured with your chosen MCP server
- [ ] `state.py` — `TypedDict` with all fields, including `human_approved`, `human_feedback`, `retry_count`, `published`
- [ ] At least **2 agent nodes**, each with a single clearly bounded responsibility
- [ ] Agent nodes only write the state fields they own
- [ ] **1 human review node** that calls `interrupt()` and returns both `human_approved` and `human_feedback`
- [ ] **1 publish node** that produces the final artifact
- [ ] `graph.py` — all nodes registered, edges complete, conditional edge covers all router return values
- [ ] `retry_count >= 3` safety exit present
- [ ] Compiled with `MemorySaver`
- [ ] `main.py` — MCP tools passed via `config["configurable"]`; interrupt/resume loop works
- [ ] Full end-to-end run completes without errors
- [ ] Human feedback loop works: pause → feedback → agent re-runs → approve
- [ ] Final artifact is published after approval
- [ ] Code committed and pushed to `main`
