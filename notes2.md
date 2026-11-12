# Module 2 - Notes

## Overview
Module 2 moves from single agents (Module 1) to agents that connect to external tools through MCP, carry custom state and runtime context, delegate work to sub-agents, and combine everything in a multi-agent wedding planner. Two bonus notebooks cover RAG and SQL agents.

| Notebook | Topic |
|---|---|
| `2.1_mcp.ipynb` | Connecting agents to local and online MCP servers |
| `2.1_travel_agent.ipynb` | Travel agent using a remote MCP server (Kiwi flights) |
| `2.2_state.ipynb` | Custom agent state and writing/reading it from tools |
| `2.2_runtime_context.ipynb` | Passing static runtime context into an agent and its tools |
| `2.3_multi_agent.ipynb` | Sub-agents exposed to a main agent as tools |
| `2.4_wedding_planners.ipynb` | Full multi-agent wedding planner with state, retries and a SQL playlist agent |
| `bonus_rag.ipynb` | Semantic search over an employee handbook PDF, used by an agent |
| `bonus_sql.ipynb` | SQL agent over the Chinook sample database |

---

## Common Setup (all notebooks)
Every notebook starts by loading the API key from the project-level `.env` file:

```python
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path.cwd().parents[1] / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENROUTER_API_KEY")
print("API key exists:", api_key is not None)
```

- `Path.cwd().parents[1]` assumes the notebook runs from `notebooks/module-2`, so `parents[1]` is the project root.
- Models are called through OpenRouter with the string form `"openrouter:openai/gpt-5-nano"`.
- Several notebooks use `await` (`ainvoke`, `get_tools`), so run them in a Jupyter cell that supports top-level `await`.

---

## 2.1 MCP (Model Context Protocol)

### Key Concepts
- **MCP** is a standard protocol for exposing tools, resources and prompts from a server to any compatible agent.
- `langchain_mcp_adapters` provides `MultiServerMCPClient`, which turns MCP tools into LangChain tools.
- Transports used: `stdio` (server runs as a local subprocess) and `streamable_http` (remote URL).
- An MCP server can also expose **prompts** and **resources**, not just tools.

### Code Examples

#### Local MCP server over stdio
```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "local_server": {
            "transport": "stdio",
            "command": "python",
            "args": ["resources/2.1_mcp_server.py"],
        }
    }
)

tools = await client.get_tools()
resources = await client.get_resources("local_server")
prompt = await client.get_prompt("local_server", "prompt")
prompt = prompt[0].content
```

#### Using the MCP tools and prompt in an agent
```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=tools,
    system_prompt=prompt
)

response = await agent.ainvoke(
    {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]}
)
```

#### Online MCP server (time server run through `uv`)
```python
client = MultiServerMCPClient(
    {
        "time": {
            "transport": "stdio",
            "command": "uv",
            "args": [
                "run", "python", "-m", "mcp_server_time",
                "--local-timezone=America/New_York"
            ]
        }
    }
)
tools = await client.get_tools()
agent = create_agent(model="openrouter:openai/gpt-5-nano", tools=tools)
response = await agent.ainvoke({"messages": [HumanMessage(content="What time is it?")]})
```

### Windows Notes (from the notebook)
- The first cells add `pywin32_system32` to the DLL search path with `os.add_dll_directory(...)`, so `import win32api` works in the `.venv312` environment.
- A cell resets `sys.stderr` on Windows Jupyter to avoid a stderr issue.

### Key Takeaways
- MCP lets you plug in tools from any server without writing the tool code yourself.
- `get_tools()` returns LangChain tools that you pass straight to `create_agent`.
- A server's prompt can be reused as the agent's `system_prompt`.

---

## 2.1 Travel Agent (remote MCP server)

### Key Concepts
- Uses a **remote** MCP server over `streamable_http` instead of a local subprocess.
- The agent gets its tools from the server, so no flight-search code is written locally.
- `InMemorySaver` plus a `thread_id` keeps the conversation across calls.
- The system prompt forbids follow-up questions, so the agent must act on the first request.

### Code Example
```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage

client = MultiServerMCPClient(
    {
        "travel_server": {
            "transport": "streamable_http",
            "url": "https://mcp.kiwi.com"
        }
    }
)
tools = await client.get_tools()

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=tools,
    checkpointer=InMemorySaver(),
    system_prompt="You are a travel agent. No follow up questions."
)

config = {"configurable": {"thread_id": "1"}}
response = await agent.ainvoke(
    {"messages": [HumanMessage(content="Get me a direct flight from San Francisco to Tokyo on March 31st")]},
    config
)
print(response["messages"][-1].content)
```

### Key Takeaways
- Remote MCP servers are connected the same way as local ones, only the transport changes.
- A system prompt can constrain behaviour (here: no clarifying questions).

---

## 2.2 State

### Key Concepts
- `AgentState` can be subclassed to add your own fields (state beyond `messages`).
- A tool can return a `Command` that **writes** to that state, using `ToolRuntime` to get the `tool_call_id`.
- A tool can **read** state through `runtime.state`.
- The checkpointer stores state per `thread_id`, so values persist across turns.

### Code Examples

#### Define custom state
```python
from langchain.agents import AgentState

class CustomState(AgentState):
    favourite_colour: str
```

#### Write to state from a tool
```python
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

@tool
def update_favourite_colour(favourite_colour: str, runtime: ToolRuntime) -> Command:
    """Update the favourite colour of the user in the state once they've revealed it."""
    return Command(update={
        "favourite_colour": favourite_colour,
        "messages": [ToolMessage("Successfully updated favourite colour", tool_call_id=runtime.tool_call_id)]}
    )

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=[update_favourite_colour],
    checkpointer=InMemorySaver(),
    state_schema=CustomState
)
```

#### Read from state in a tool
```python
@tool
def read_favourite_colour(runtime: ToolRuntime) -> str:
    """Read the favourite colour of the user from the state."""
    try:
        return runtime.state["favourite_colour"]
    except KeyError:
        return "No favourite colour found in state"
```

The notebook shows that, on the same `thread_id`, a later question ("What's my favourite colour?") is answered from the stored state.

### Key Takeaways
- Use `Command(update=...)` to change custom state from a tool.
- Use `runtime.state` to read it; combine with a checkpointer to keep it between turns.
- Passing a state value directly in `invoke` (e.g. `"favourite_colour": "green"`) also works.

---

## 2.2 Runtime Context

### Key Concepts
- **Runtime context** is static, per-invocation data (not conversation state), defined with a dataclass.
- Pass it with `context_schema` on `create_agent`, then supply `context=...` on `invoke`.
- Tools read it through `ToolRuntime[ContextType]` as `runtime.context`.

### Code Examples

#### Define and use context
```python
from dataclasses import dataclass

@dataclass
class ColourContext:
    favourite_colour: str = "blue"
    least_favourite_colour: str = "yellow"

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    context_schema=ColourContext
)

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=ColourContext()
)
```

#### Access context from tools
```python
from langchain.tools import tool, ToolRuntime

@tool
def get_favourite_colour(runtime: ToolRuntime[ColourContext]) -> str:
    """Get the favourite colour of the user"""
    return runtime.context.favourite_colour

@tool
def get_least_favourite_colour(runtime: ToolRuntime[ColourContext]) -> str:
    """Get the least favourite colour of the user"""
    return runtime.context.least_favourite_colour

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=[get_favourite_colour, get_least_favourite_colour],
    context_schema=ColourContext
)

response = agent.invoke(
    {"messages": [HumanMessage(content="What is my favourite colour?")]},
    context=ColourContext(favourite_colour="green")
)
```

The notebook compares the runs: the first run (no context tools) answers "I don't know", because the agent has no way to read the context. Once tools read `runtime.context`, the agent answers using the value passed at call time (`green` in the last run).

### Key Takeaways
- Context = read-only, per-call configuration (user id, preferences, API settings).
- State = data that changes during the conversation.
- Use `context_schema` + `context=` for context, and tools read it via `runtime.context`.

---

## 2.3 Multi-Agent (sub-agents as tools)

### Key Concepts
- A **sub-agent** is a normal `create_agent` agent with its own tools.
- Each sub-agent is wrapped in a `@tool` function, so the **main agent** can call it like any other tool.
- The main agent chooses which sub-agent to call based on the tool docstrings.

### Code Examples

#### Create sub-agents
```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2

subagent_1 = create_agent(model="openrouter:openai/gpt-5-nano", tools=[square_root])
subagent_2 = create_agent(model="openrouter:openai/gpt-5-nano", tools=[square])
```

#### Wrap sub-agents as tools and build the main agent
```python
from langchain.messages import HumanMessage

@tool
def call_subagent_1(x: float) -> float:
    """Call subagent 1 in order to calculate the square root of a number"""
    response = subagent_1.invoke({"messages": [HumanMessage(content=f"Calculate the square root of {x}")]})
    return response["messages"][-1].content

@tool
def call_subagent_2(x: float) -> float:
    """Call subagent 2 in order to calculate the square of a number"""
    response = subagent_2.invoke({"messages": [HumanMessage(content=f"Calculate the square of {x}")]})
    return response["messages"][-1].content

main_agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=[call_subagent_1, call_subagent_2]
)

response = main_agent.invoke({"messages": [HumanMessage(content="What is the square root of 456?")]})
```

### Key Takeaways
- Sub-agents let you split work by specialism, each with a narrow tool set.
- Wrapping an agent in a tool is the basic multi-agent pattern in this module.
- Clear docstrings on the wrapper tools are what route the request to the right sub-agent.

---

## 2.4 Wedding Planner (full multi-agent system)

This is the capstone lab. It combines MCP tools, a SQL database, custom state, sub-agents, retries and a coordinator agent.

### Key Concepts
- **Setup Tools**: a `RetryMCPInterceptor` class retries transient MCP errors (code `-32603`) with exponential backoff, and returns non-retryable errors (e.g. `-32602`) to the agent right away.
- **Search tool with a budget**: `web_search` takes `search_number` and `max_search_number`, and stops with a message once the limit is reached. Queries must use plain ASCII text.
- **Custom state**: `WeddingState(AgentState)` stores `origin`, `destination`, `guest_count` and `genre`.
- **Specialist sub-agents**: travel agent (flights via MCP tools), venue agent (web search), and playlist agent (SQL over Chinook).
- **Coordinator**: a main agent with tools `search_flights`, `search_venues`, `suggest_playlist` and `update_state`, using `state_schema=WeddingState`.
- **Tracing**: `config={"tags": ["WP"], "recursion_limit": 40}` tags the run for LangSmith and allows more steps.

### Code Examples

#### Search tool with a search budget
```python
from typing import Dict, Any
from tavily import TavilyClient
from langchain.tools import tool

tavily_client = TavilyClient()

@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. You must track your search count by providing
    search_number (starting at 1) and max_search_number on every call.
    """
    if search_number > max_search_number:
        return {"message": "Search limit reached. Please summarize your findings and provide your final answer."}
    # ... search logic (see notebook)
```

#### SQL playlist tool
```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

@tool
def query_playlist_db(query: str) -> str:
    """Query the database for playlist information"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error querying database: {e}"
```

#### State schema
```python
from langchain.agents import AgentState

class WeddingState(AgentState):
    origin: str
    destination: str
    guest_count: str
    genre: str
```

#### Sub-agents
```python
travel_agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=tools,            # MCP flight tools from the setup cells
    system_prompt="..."     # find the best flights, no follow-up questions
)

playlist_agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=[query_playlist_db],
    system_prompt="..."     # discover the SQLite schema first, then build a playlist by genre
)
```

The venue agent is built the same way, with a limit of 12 web searches and rules for price, exact capacity and reviews.

#### Coordinator wrapper tools
```python
from langchain.tools import ToolRuntime
from langchain.messages import HumanMessage

@tool
async def search_flights(runtime: ToolRuntime) -> str:
    """Travel agent searches for flights to the desired destination wedding location."""
    origin = runtime.state["origin"]
    destination = runtime.state["destination"]
    response = await travel_agent.ainvoke({"messages": [HumanMessage(content=f"Find flights from {origin} to {destination}")]})
    return response["messages"][-1].content

# search_venues and suggest_playlist follow the same pattern (read state, call sub-agent, return text)
# update_state uses Command(update=...) to write the WeddingState fields
```

#### Coordinator and test run
```python
coordinator = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=[search_flights, search_venues, suggest_playlist, update_state],
    state_schema=WeddingState,
    system_prompt="You are a wedding coordinator. First update the state, then delegate to the specialists."
)

response = await coordinator.ainvoke(
    {"messages": [HumanMessage(content="I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre")]},
    config={"tags": ["WP"], "recursion_limit": 40},
)
print(response["messages"][-1].content)
```

The test run extracts `destination='Paris'`, `genre='jazz'`, `guest_count='100'` into state, then returns flight, venue and playlist options.

### Key Takeaways
- Return tool errors to the agent instead of raising them, so it can retry with a different input.
- Put shared data (origin, destination, guests, genre) in custom state so every sub-agent can read it.
- Give each sub-agent a narrow job and a clear system prompt, and let the coordinator orchestrate.
- Use limits (search budget, recursion limit) to keep runs cheap and bounded.

---

## Bonus: RAG over the Employee Handbook

### Key Concepts
- Load a PDF with `PyPDFLoader`, then split it with `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`).
- Embed chunks with `OpenAIEmbeddings(model="text-embedding-3-large")` and store them in `InMemoryVectorStore`.
- Expose similarity search as a tool, so an agent can answer questions from the handbook.

### Code Example
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage

data = PyPDFLoader("resources/acmecorp-employee-handbook.pdf").load()
splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True).split_documents(data)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(documents=splits)

@tool
def search_handbook(query: str) -> str:
    """Search the employee handbook for information"""
    results = vector_store.similarity_search(query)
    return results[0].page_content

agent = create_agent(
    model="gpt-5-nano",
    tools=[search_handbook],
    system_prompt="You are a helpful agent that can search the employee handbook for information."
)
response = agent.invoke({"messages": [HumanMessage(content="How many days of vacation does an employee get in their first year?")]})
```

### Key Takeaways
- RAG = load, split, embed, store, retrieve, then give retrieval to the agent as a tool.
- `add_start_index=True` records where each chunk starts in the source document.
- This notebook uses `OpenAIEmbeddings` directly, so it needs an OpenAI key, not only the OpenRouter key.

---

## Bonus: SQL Agent (Chinook database)

### Key Concepts
- `SQLDatabase.from_uri("sqlite:///resources/Chinook.db")` connects to the sample music store database.
- The `sql_query` tool returns `"Error: ..."` text on failure instead of raising, so the agent can see the error.
- The agent writes its own SQL, and the notebook inspects the query it generated.

### Code Example
```python
from langchain_community.utilities import SQLDatabase
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage

db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

@tool
def sql_query(query: str) -> str:
    """Obtain information from the database using SQL queries"""
    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"

agent = create_agent(model="gpt-5-nano", tools=[sql_query])
response = agent.invoke({"messages": [HumanMessage(content="Who is the most popular artist beginning with 'S' in this database?")]})
print(response["messages"][-3].tool_calls[0]["args"]["query"])
```

### Key Takeaways
- An agent can answer database questions by writing SQL through a tool.
- Returning errors as strings lets the agent read them and correct its query.

---

## Summary
1. **MCP**: brings in external tools from local or remote servers.
2. **State**: stores data that changes during a conversation (`Command` writes, `runtime.state` reads).
3. **Runtime context**: passes static per-call data (`context_schema` + `context=`).
4. **Sub-agents**: wrapped as tools for a main agent.
5. **Wedding planner**: combines all of the above into a multi-agent system with retries and limits.
6. **Bonus**: RAG and SQL shown as agent tools.
