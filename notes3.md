# Module 3 - Notes

## Overview
Module 3 is about **controlling agents at runtime**. Instead of only giving an agent tools and a prompt, you attach **middleware** that changes what happens before and during each model or tool call. The topics are:

| Notebook | Topic |
|---|---|
| `3.2_managing_messages.ipynb` | Summarising and trimming the message history |
| `3.3_hitl.ipynb` | Human-in-the-loop approval of tool calls (approve, reject, edit) |
| `3.4_dynamic_models.ipynb` | Choosing the model per call based on conversation length |
| `3.4_dynamic_prompts.ipynb` | Building the system prompt from runtime context (language) |
| `3.4_dynamic_tools.ipynb` | Choosing which tools are visible based on the user's role |
| `3.5_email_agent.ipynb` | Capstone: authentication, dynamic tools and prompt, and HITL for sending email |

The `agent-chat-ui` folder is a Next.js front end for talking to these agents through LangGraph. The `3.5_email_agent.py` file and `langgraph.json` configure the same agent for LangGraph Studio. These are not notebooks, so these notes do not cover them in detail.

---

## Common Setup (all notebooks)
As in modules 1 and 2, each notebook loads the key from the project `.env` file:

```python
import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path.cwd().parents[1] / ".env"
load_dotenv(dotenv_path=env_path, override=True)

api_key = os.getenv("OPENROUTER_API_KEY")
print("API key exists:", api_key is not None)
```

Key ideas used across this module:
- **Middleware** is passed as `middleware=[...]` to `create_agent`. It can change the model, prompt, tools or messages.
- **Hooks** come in two kinds. `@before_agent` runs once before the agent loop. `@wrap_model_call` wraps each model call, and `@dynamic_prompt` builds the system prompt for each call.
- **Context** (`context_schema` + `context=`) and **state** (`AgentState` subclasses) are both available to middleware through `request.runtime.context` and `request.state`.
- **Checkpointer** (`InMemorySaver`) and `thread_id` are needed for anything that pauses or remembers across calls, such as HITL.

---

## 3.2 Managing Messages

### 3.2a Summarising messages

#### Key Concepts
- `SummarizationMiddleware` replaces older messages with a summary once a threshold is reached, so the context does not grow without limit.
- `trigger=("tokens", 100)` starts summarising when the history reaches about 100 tokens (a very low value for demonstration).
- `keep=("messages", 1)` keeps only the most recent message verbatim. Everything earlier is folded into the summary.
- The summary model (`openrouter:openai/gpt-4o-mini`) is separate from the main model (`gpt-5-nano`).

#### Code Example
```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="openrouter:openai/gpt-4o-mini",
            trigger=("tokens", 100),
            keep=("messages", 1)
        )
    ],
)

response = agent.invoke(
    {"messages": [...several HumanMessage / AIMessage turns...]},
    {"configurable": {"thread_id": "1"}}
)
print(response["messages"][0].content)   # the generated summary
```

The notebook passes a long fictional conversation about "Lunapolis". The first message in the returned state is the summary, which starts with "Here is a summary of the conversation to date:" and has sections such as `## SESSION INTENT` and `## SUMMARY`.

### 3.2b Trimming (deleting) messages

#### Key Concepts
- A `@before_agent` middleware runs before each agent turn and can edit the state.
- Returning `RemoveMessage(id=...)` deletes the matching messages from the history.
- This example removes every `ToolMessage`, so only the human and AI turns remain.

#### Code Example
```python
from typing import Any
from langchain.agents import AgentState
from langchain.messages import RemoveMessage, ToolMessage
from langgraph.runtime import Runtime
from langchain.agents.middleware import before_agent

@before_agent
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Remove all the tool messages from the state"""
    messages = state["messages"]
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
    return {"messages": [RemoveMessage(id=m.id) for m in tool_messages]}

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    checkpointer=InMemorySaver(),
    middleware=[trim_messages],
)
```

The notebook then runs a device-troubleshooting conversation (thread `"2"`) that includes `ToolMessage` entries. The printed reply answers the last question, and the tool messages have been removed from the history.

### Key Takeaways
- Summarisation keeps long conversations within the context window while preserving the key facts.
- `keep=("messages", n)` controls how many recent messages are kept as-is.
- `RemoveMessage` lets middleware delete specific messages from the stored history.
- Use `thread_id` with a checkpointer so the trimmed or summarised history is saved for the next turn.

---

## 3.3 Human-in-the-Loop (HITL)

### Key Concepts
- `HumanInTheLoopMiddleware` pauses the agent before a tool runs, so a person can decide.
- `interrupt_on` maps tool names to `True` (pause for review) or `False` (run without review).
- `description_prefix` sets the text shown to the reviewer for each pending action.
- The paused run is returned in `response["__interrupt__"]`, and it contains `action_requests` with the tool name and arguments.
- The run is resumed with `Command(resume={"decisions": [...]})` on the **same `thread_id`**. Each decision is one of:
  - `{"type": "approve"}`: run the tool as proposed.
  - `{"type": "reject", "message": "..."}`: skip the tool and give the reason to the model.
  - `{"type": "edit", "edited_action": {"name": ..., "args": {...}}}`: change the arguments, then run the tool.

### Code Example: setup
```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware

@tool
def read_email(runtime: ToolRuntime) -> str:
    """Read an email from the given address."""
    return runtime.state["email"]        # email is read from state

@tool
def send_email(body: str) -> str:
    """Send an email to the given address with the given subject and body."""
    return f"Email sent"                  # fake send

class EmailState(AgentState):
    email: str

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=[read_email, send_email],
    state_schema=EmailState,
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"read_email": False, "send_email": True},
            description_prefix="Tool execution requires approval",
        ),
    ],
)
```

### Code Example: pause, then approve, reject or edit
```python
from langchain.messages import HumanMessage
from langgraph.types import Command

config = {"configurable": {"thread_id": "1"}}

# 1. Run until the agent wants to send an email; it pauses
response = agent.invoke(
    {
        "messages": [HumanMessage(content="Please read my email and send a response immediately. Send the reply now in the same thread.")],
        "email": "Hi Seán, I'm going to be late for our meeting tomorrow. Can we reschedule? Best, John.",
    },
    config=config,
)
print(response["__interrupt__"][0].value["action_requests"][0]["args"]["body"])

# 2a. Approve: send the draft as is
response = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)

# 2b. Reject: do not send, and tell the model why
response = agent.invoke(Command(resume={"decisions": [{"type": "reject", "message": "No please sign off - Your merciful leader, Seán."}]}), config=config)

# 2c. Edit: change the body before sending
response = agent.invoke(
    Command(resume={"decisions": [{"type": "edit", "edited_action": {"name": "send_email", "args": {"body": "This is the last straw, you're fired!"}}}]}),
    config=config,
)
```

Notes from the notebook:
- The draft reply the model produced was a polite reply agreeing to reschedule (suggesting 2:00 PM or 3:00 PM).
- Approving sends that draft. Rejecting returns the message to the model. Editing sends your own text instead.
- Every cell uses the same `thread_id` ("1"), so each resume continues the same paused run. The reject and edit examples each start a new run first, so they do not depend on the earlier approval.

### Key Takeaways
- HITL is set per tool through `interrupt_on`, so you can review only risky actions (here, sending email).
- The same `thread_id` must be used to resume.
- `approve`, `reject` and `edit` give the human three levels of control.

---

## 3.4 Dynamic Models

### Key Concepts
- `@wrap_model_call` wraps each model call, so you can choose a different model for each turn.
- `request.override(model=...)` swaps the model for that call only.
- `request.messages` is a shortcut for `request.state["messages"]`, so the choice can depend on conversation length.

### Code Example
```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

large_model = init_chat_model("openrouter:anthropic/claude-sonnet-4-6")
standard_model = init_chat_model("openrouter:openai/gpt-5-nano")

@wrap_model_call
def state_based_model(request: ModelRequest,
                      handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Select model based on State conversation length."""
    message_count = len(request.messages)
    if message_count > 10:
        model = large_model       # long conversation: larger model
    else:
        model = standard_model    # short conversation: cheaper model
    request = request.override(model=model)
    return handler(request)

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    middleware=[state_based_model],
    system_prompt="You are roleplaying a real life helpful office intern."
)
```

The notebook checks which model answered with `response["messages"][-1].response_metadata["model_name"]`:
- A short one-question conversation is answered by `openai/gpt-5-nano`.
- A longer 11-message conversation (over the threshold of 10) is answered by `anthropic/claude-sonnet-4.6`.

### Key Takeaways
- You can route cheap requests to a small model and long or hard ones to a larger model, without changing the agent.
- Check `response_metadata["model_name"]` to see which model was used.

---

## 3.4 Dynamic Prompts

### Key Concepts
- `@dynamic_prompt` builds the system prompt on each call from runtime data.
- The data comes from a context dataclass passed in `invoke(..., context=...)`.

### Code Example
```python
from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class LanguageContext:
    user_language: str = "English"

@dynamic_prompt
def user_language_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_language = request.runtime.context.user_language
    base_prompt = "You are a helpful assistant."
    if user_language != "English":
        return f"{base_prompt} only respond in {user_language}."
    elif user_language == "English":
        return base_prompt

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    context_schema=LanguageContext,
    middleware=[user_language_prompt]
)

response = agent.invoke(
    {"messages": [HumanMessage(content="Hello, how are you?")]},
    context=LanguageContext(user_language="Irish")
)
```

The same question with `Spanish` and `French` in the context returns a reply in that language. The Irish example replies in Irish ("Dia dhuit! Tá mé go maith...").

### Key Takeaways
- The system prompt can change per request without rebuilding the agent.
- Put per-user settings (language, role, plan) in context, and read them in the prompt function.

---

## 3.4 Dynamic Tools

### Key Concepts
- `@wrap_model_call` can call `request.override(tools=[...])` to limit which tools the model sees on a given call.
- The tool list is chosen from `request.runtime.context`, so each user gets a different set of tools.
- The tools still exist on the agent. The middleware only changes what the model is offered.

### Code Example
```python
from dataclasses import dataclass
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@dataclass
class UserRole:
    user_role: str = "external"

@wrap_model_call
def dynamic_tool_call(request: ModelRequest,
                      handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Dynamically call tools based on the runtime context"""
    user_role = request.runtime.context.user_role
    if user_role == "internal":
        pass                          # internal users see all tools
    else:
        tools = [web_search]          # external users: web search only
        request = request.override(tools=tools)
    return handler(request)

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=[web_search, sql_query],
    middleware=[dynamic_tool_call],
    context_schema=UserRole
)

response = agent.invoke(
    {"messages": [HumanMessage(content="How many artists are in the database?")]},
    context={"user_role": "external"}
)
```

The notebook's `web_search` tool is a Tavily search, and `sql_query` runs queries against `Chinook.db`. An external user is offered only `web_search`, so `sql_query` is not visible to the model. In the saved output, the model asks which database is meant and offers alternatives, instead of querying Chinook.

### Key Takeaways
- Use middleware to restrict tools by role, so the model cannot even see sensitive tools.
- `request.override(tools=...)` affects only that call.

---

## 3.5 Email Agent (capstone)

This notebook puts the module together. The agent must authenticate before it can read the inbox or send email, and sending requires human approval.

### Flow
1. **Authenticate**: the only tool offered at first is `authenticate(email, password)`.
2. **Unlock tools**: after a successful login, `check_inbox` and `send_email` become available, and the prompt changes.
3. **Read**: `check_inbox` returns a hard-coded message from Jane.
4. **Draft and approve**: the agent drafts a reply, and `HumanInTheLoopMiddleware` pauses before `send_email`.

### Key Concepts
- `EmailContext` holds the expected credentials (`email_address`, `password`), so they are not hard-coded in the tool.
- `AuthenticatedState(AgentState)` adds an `authenticated` flag that is stored in state.
- `authenticate` returns a `Command` that sets `authenticated` to `True` or `False` and adds a `ToolMessage`.
- `dynamic_tool_call` (a `@wrap_model_call`) offers `[authenticate]` until the user is authenticated, then `[check_inbox, send_email]`.
- A `@dynamic_prompt` function picks one system prompt for the unauthenticated state and another for the authenticated state.

### Code Example: context, state and tools
```python
from dataclasses import dataclass
from langchain.agents import AgentState
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage

@dataclass
class EmailContext:
    email_address: str = "julie@example.com"
    password: str = "password123"

class AuthenticatedState(AgentState):
    authenticated: bool

@tool
def check_inbox() -> str:
    """Check the inbox for recent emails"""
    return """
    Hi Julie,
    I'm going to be in town next week and was wondering if we could grab a coffee?
    - best, Jane (jane@example.com)
    """

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an response email"""
    return f"Email sent to {to} with subject {subject} and body {body}"

@tool
def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
    """Authenticate the user with the given email and password"""
    if email == runtime.context.email_address and password == runtime.context.password:
        return Command(update={
            "authenticated": True,
            "messages": [ToolMessage("Successfully authenticated", tool_call_id=runtime.tool_call_id)]})
    else:
        return Command(update={
            "authenticated": False,
            "messages": [ToolMessage("Authentication failed", tool_call_id=runtime.tool_call_id)]})
```

### Code Example: middleware (tools and prompt depend on state)
```python
from langchain.agents.middleware import wrap_model_call, dynamic_prompt, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def dynamic_tool_call(request: ModelRequest,
                      handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Allow read inbox and send email tools only if user provides correct email and password"""
    authenticated = request.state.get("authenticated")
    if authenticated:
        tools = [check_inbox, send_email]
    else:
        tools = [authenticate]
    request = request.override(tools=tools)
    return handler(request)

authenticated_prompt = """You are a helpful assistant that can check the inbox and send emails.
Your first step after authentication is to check the inbox."""
unauthenticated_prompt = "You are a helpful assistant that can authenticate users."

@dynamic_prompt
def auth_prompt(request: ModelRequest) -> str:   # the notebook names this function dynamic_prompt
    """Generate system prompt based on authentication status"""
    authenticated = request.state.get("authenticated")
    return authenticated_prompt if authenticated else unauthenticated_prompt
```

The notebook names the prompt function `dynamic_prompt`, which shadows the decorator of the same name. The version above renames it to `auth_prompt` for clarity. The behaviour is the same.

### Code Example: agent with HITL
```python
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    "gpt-5-nano",
    tools=[authenticate, check_inbox, send_email],
    checkpointer=InMemorySaver(),
    state_schema=AuthenticatedState,
    context_schema=EmailContext,
    middleware=[
        dynamic_tool_call,
        auth_prompt,
        HumanInTheLoopMiddleware(
            interrupt_on={
                "authenticate": False,
                "check_inbox": False,
                "send_email": True,
            })
    ],
)
```

### Running the flow
```python
from langchain.messages import HumanMessage
from langgraph.types import Command

config = {"configurable": {"thread_id": "1"}}

# 1. Log in with the credentials in EmailContext
response = agent.invoke(
    {"messages": [HumanMessage(content="julie@example.com, password123")]},
    context=EmailContext(), config=config
)
print(response["messages"][-1].content)     # summary of the inbox and a reply draft

# 2. Ask for any draft. The send_email call is paused for approval.
response = agent.invoke(
    {"messages": [HumanMessage(content="any draft is fine. don't check back.")]},
    context=EmailContext(), config=config
)
print(response["__interrupt__"][0].value["action_requests"][0]["args"]["body"])

# 3. Approve the send
response = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
print(response["messages"][-1].content)
```

In the saved output, step 1 reads the inbox and offers a draft. Step 2 pauses with a body that agrees to meet and asks for a Tuesday or Thursday afternoon. Step 3 sends it to `jane@example.com` with the subject "Re: Coffee next week".

### Key Takeaways
- Gate tools on state: until login succeeds, the model is only offered `authenticate`.
- Change the system prompt with the same state, so the model's instructions match what it can do.
- Combine dynamic tools, dynamic prompts and HITL in one agent for a secure, multi-step workflow.
- Credentials and other settings live in context, not in the tool code.

---

## Summary
1. **Managing messages**: `SummarizationMiddleware` compresses old turns, and `@before_agent` with `RemoveMessage` deletes messages.
2. **Human-in-the-loop**: `HumanInTheLoopMiddleware` pauses a tool, and `Command(resume=...)` approves, rejects or edits it.
3. **Dynamic models**: `@wrap_model_call` plus `request.override(model=...)` chooses the model per turn.
4. **Dynamic prompts**: `@dynamic_prompt` builds the system prompt from context or state.
5. **Dynamic tools**: `request.override(tools=...)` limits which tools the model sees, based on role or auth state.
6. **Email agent**: combines authentication, dynamic tools and prompts, and HITL in one workflow.

---
