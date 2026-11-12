# Module 1 - LangChain Foundations Notes

## Overview
This module covers the foundational concepts of LangChain including models, prompting, tools, memory, multimodal inputs, and building a practical application.

---

## 1.1 Foundational Models

### Key Concepts
- **Model Initialization**: Using `init_chat_model()` from LangChain to initialize chat models
- **OpenRouter Integration**: Using OpenRouter as a proxy to access various LLM models (like GPT-5-nano)
- **Agent Creation**: Using `create_agent()` to create conversational agents

### Code Examples

#### Basic Model Setup
```python
from langchain.chat_models import init_chat_model
import os

model = init_chat_model(
    model="openai/gpt-5-nano",
    model_provider="openai",
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    temperature=1.0
)
```

#### Simple Agent Creation
```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage

agent = create_agent(
    model="openrouter:openai/gpt-5-nano"
)

response = agent.invoke({"messages": [HumanMessage(content="Your question")]})
```

#### Streaming Responses
```python
for token, metadata in agent.stream(
    {"messages": [HumanMessage(content="Tell me about...")]},
    stream_mode="messages"
):
    if token.content:
        print(token.content, end="", flush=True)
```

### Key Takeaways
- Models can be invoked synchronously or streamed for real-time responses
- Temperature controls randomness (higher = more creative)
- Responses include metadata like token usage and reasoning details

---

## 1.1 Prompting

### Key Concepts
- **System Prompts**: Guide the model's behavior and role
- **Structured Output**: Using Pydantic models to enforce response format
- **Response Format**: Extracting structured data from AI responses

### Code Examples

#### Basic Prompting with System Prompt
```python
agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    system_prompt="You are a science fiction writer, create a capital city at the users request."
)

question = HumanMessage(content="What is the capital of The Moon?")
response = agent.invoke({"messages": [question]})
```

#### Structured Output with Pydantic
```python
from pydantic import BaseModel

class CapitalInfo(BaseModel):
    name: str
    location: str
    vibe: str
    economy: str

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    system_prompt="You are a science fiction writer...",
    response_format=CapitalInfo
)

response = agent.invoke({"messages": [question]})
capital_info = response["structured_response"]

# Access structured fields
print(capital_info.name)  # e.g., "Selene Prime"
print(capital_info.location)  # e.g., "Shackleton Crater rim"
```

### Key Takeaways
- System prompts define the agent's personality and behavior
- Structured output ensures consistent, parseable responses
- Pydantic models provide type safety and validation

---

## 1.2 Tools

### Key Concepts
- **Tool Definition**: Using `@tool` decorator to create custom tools
- **Tool Invocation**: How agents decide when and how to use tools
- **Tool Calls**: Examining tool call metadata in responses

### Code Examples

#### Defining a Tool
```python
from langchain.tools import tool

@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5
```

#### Using Tools with Agents
```python
agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=[square_root],
    system_prompt="You are an arithmetic wizard..."
)

response = agent.invoke(
    {"messages": [HumanMessage(content="What is the square root of 467?")]}
)
```

### Key Takeaways
- Tools extend agent capabilities beyond text generation
- Function docstrings guide the agent on when to use the tool
- Agents automatically decide when to use tools

---

## 1.2 Web Search

### Key Concepts
- **Web Search Integration**: Using Tavily API for real-time searches
- **Knowledge Cutoff**: Understanding model training limitations
- **Tool-Augmented Responses**: Combining model knowledge with real-time data

### Code Examples
```python
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=[web_search]
)
```

### Key Takeaways
- Web search enables access to current information
- Tavily provides structured results for LLM consumption
- Agents decide when web search is needed

---

## 1.3 Memory

### Key Concepts
- **Stateless vs Stateful**: Agents without memory vs with memory
- **Checkpointing**: Using `InMemorySaver()` to persist conversation state
- **Thread ID**: Identifying conversation sessions

### Code Examples

#### Without Memory (Stateless)
```python
agent = create_agent(model="openrouter:openai/gpt-5-nano")

response = agent.invoke({
    "messages": [HumanMessage(content="My name is Seán")]
})

# Agent doesn't remember in next call
response = agent.invoke({
    "messages": [HumanMessage(content="What's my name?")]
})
```

#### With Memory (Stateful)
```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [HumanMessage(content="My name is Seán")]},
    config
)

# Agent remembers
response = agent.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config
)
# Response: "Seán"
```

### Key Takeaways
- Memory allows context across multiple interactions
- `InMemorySaver()` stores conversation history (not persistent)
- Thread IDs enable multiple separate conversations
- For production, use persistent checkpointers

---

## 1.4 Multimodal Messages

### Key Concepts
- **Text Input**: Traditional text queries
- **Image Input**: Analyzing images (URL or base64)
- **Audio Input**: Processing audio files
- **Content Arrays**: Structuring multimodal messages

### Code Examples

#### Image Input (URL)
```python
question = HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image", "url": "https://example.com/image.jpg"}
])

response = agent.invoke({"messages": [question]})
```

#### Image Input (Base64)
```python
import base64

with open("image.jpg", "rb") as img_file:
    img_b64 = base64.b64encode(img_file.read()).decode("utf-8")

question = HumanMessage(content=[
    {"type": "text", "text": "Describe this image"},
    {"type": "image", "base64": img_b64, "mime_type": "image/jpeg"}
])
```

#### Audio Input
```python
import sounddevice as sd
from scipy.io.wavfile import write

# Record audio
duration = 5
sample_rate = 44100
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()

# Convert to base64
buf = io.BytesIO()
write(buf, sample_rate, audio)
aud_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

# Use audio-capable model
agent = create_agent(model='openrouter:openai/gpt-audio')

question = HumanMessage(content=[
    {"type": "text", "text": "Tell me about this audio"},
    {"type": "audio", "base64": aud_b64, "mime_type": "audio/wav"}
])
```

### Key Takeaways
- Multimodal models process text, images, and audio
- Images via URL (public) or base64 (local/private)

## 1.5 Personal Chef (Practical Application)

### Key Concepts
- **Domain-Specific Agent**: Creating specialized agents
- **Combining Features**: Memory + Tools + Prompting
- **Vietnamese Language Support**: Multilingual agents
- **Web Search for Recipes**: Real-time recipe suggestions

### Code Examples

#### Complete Personal Chef Agent
```python
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)

system_prompt = """
Bạn là một đầu bếp. Người dùng sẽ cung cấp cho bạn nguyên liệu nấu ăn, 
sử dụng web search tool để tìm công thức nấu ăn với những nguyên liệu họ có
trả lời bằng những gợi ý công thức nấu ăn chi tiết
"""

agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    tools=[web_search],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)

# Use the agent
config = {"configurable": {"thread_id": "chef_session_1"}}
response = agent.invoke(
    {"messages": [HumanMessage(content="Tôi có cá Basa và nước mắm")]},
    config
)
```

### Agent Behavior
The personal chef agent:
1. **Understands Vietnamese input** about ingredients
2. **Uses web search** to find relevant recipes
3. **Synthesizes information** from multiple sources
4. **Provides detailed suggestions** with:
   - Recipe names and descriptions
   - Cooking methods
   - Reference links
   - Step-by-step instructions
5. **Remembers context** for follow-up questions

### Example Output
When given "cá Basa và nước mắm Nam Ngư", the agent suggests:
- Cá Basa hấp nước mắm gừng tỏi (Steamed)
- Cá Basa chiên nước mắm (Fried)
- Cá Basa kho tộ (Braised)
- Canh chua cá Basa (Sour soup)

Each suggestion includes links to detailed recipes.

### Key Takeaways
- Agents can be specialized for specific domains
- Combining memory, tools, and prompting creates powerful apps
- LLMs work in multiple languages
- Web search provides current, relevant information

---

## General Best Practices

### Environment Setup
- Use `.env` files for API keys
- Load environment variables with `dotenv`
- Verify API keys before making calls

### Model Selection
- Choose models based on capability needs
- Consider cost vs performance tradeoffs
- Use OpenRouter for multiple provider access

### Error Handling
- Always check if API keys are loaded
- Validate tool inputs and outputs
- Handle edge cases in tool implementations

### Performance
- Use streaming for better UX with long responses
- Implement proper memory management
- Consider token limits and costs

### Agent Design
- Write clear, descriptive tool docstrings
- Create focused system prompts
- Test tools independently before adding to agents
- Use structured outputs for consistent data

---

## Common Patterns

### Basic Agent Pattern
```python
agent = create_agent(
    model="openrouter:openai/gpt-5-nano",
    system_prompt="Your role definition...",
    tools=[tool1, tool2],
    checkpointer=InMemorySaver()
)
```

### Multimodal Query Pattern
```python
message = HumanMessage(content=[
    {"type": "text", "text": "Your question"},
    {"type": "image", "url": "image_url"},
    {"type": "audio", "base64": "audio_data", "mime_type": "audio/wav"}
])
```

### Stateful Conversation Pattern
```python
config = {"configurable": {"thread_id": "unique_session_id"}}
response = agent.invoke({"messages": [message]}, config)
```

---

## Resources & References

- **LangChain Documentation**: https://python.langchain.com/
- **OpenRouter**: https://openrouter.ai/
- **Tavily API**: https://tavily.com/
- **Pydantic**: https://docs.pydantic.dev/

---

## Summary

Module 1 provides a comprehensive introduction to building LangChain applications:

1. **Foundational Models** - Basic model initialization and invocation
2. **Prompting** - System prompts and structured outputs
3. **Tools** - Custom tools and web search integration
4. **Memory** - Maintaining context with checkpointing
5. **Multimodal** - Processing text, ưimages, and audio
6. **Practical Application** - Building complete, specialized agents

The progression demonstrates building production-ready AI agents for real-world problems.

- Audio requires specific models
- Content structured as type-specific objects

---

