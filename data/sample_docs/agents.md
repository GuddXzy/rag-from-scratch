# LangChain Agents

## What are Agents?

Agents use a language model to choose a sequence of actions to take. In chains, the sequence of actions is hardcoded in code. In agents, a language model is used as a reasoning engine to determine which actions to take and in which order.

## Key Components

### Tools
Tools are functions that an agent can call. A tool consists of:
- **name**: A unique name for the tool
- **description**: What the tool does (the LLM reads this to decide when to use it)
- **function**: The actual function to execute

```python
from langchain.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information about a topic."""
    # implementation here
    return results
```

### Agent Types

LangChain supports several agent types:

1. **ReAct Agent**: Uses Reasoning + Acting pattern. The agent thinks step by step about what to do, takes an action, observes the result, and repeats.

2. **OpenAI Functions Agent**: Uses OpenAI's function calling API for structured tool use.

3. **Structured Chat Agent**: Can handle multi-input tools using a structured output format.

### LangGraph for Agents

For more complex agent workflows, LangGraph provides a graph-based framework:

```python
from langgraph.graph import StateGraph

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue)
```

LangGraph supports:
- Cycles and branching in agent logic
- Persistent state across interactions
- Human-in-the-loop patterns
- Streaming of intermediate steps

## Memory

Agents can maintain memory across interactions:

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
agent = create_react_agent(llm, tools, memory=memory)
```

Memory types include:
- `ConversationBufferMemory` - Stores all messages
- `ConversationSummaryMemory` - Summarizes older messages
- `ConversationBufferWindowMemory` - Keeps last K messages
