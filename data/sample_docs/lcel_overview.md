# LangChain Expression Language (LCEL)

## Overview

LangChain Expression Language, or LCEL, is a declarative way to compose LangChain components into chains. LCEL was designed from day one to support putting prototypes in production, with no code changes, from the simplest "prompt + LLM" chain to the most complex chains with hundreds of steps.

## Why use LCEL?

LCEL makes it easy to build complex chains from basic components, and supports out-of-the-box functionality such as:

- **Streaming**: Get the best possible time-to-first-token when using LCEL. Some chains can stream output directly from an LLM to a streaming output parser, and you get back parsed, incremental chunks of output at the same rate as the LLM provider outputs the raw tokens.
- **Async support**: Any chain built with LCEL can be called both with the synchronous API (e.g. in your Jupyter notebook) and with the asynchronous API (e.g. in a LangServe server). This enables using the same code for prototypes and production.
- **Optimized parallel execution**: Whenever your LCEL chains have steps that can be executed in parallel, we automatically do it for best possible performance.
- **Retries and fallbacks**: Configure retries and fallbacks for any part of your LCEL chain. This is a great way to make your chains more reliable at scale.

## Basic Example

The most basic and common use case is chaining a prompt template and a model together:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
model = ChatAnthropic(model="claude-sonnet-4-20250514")
chain = prompt | model

response = chain.invoke({"topic": "bears"})
print(response.content)
```

The `|` operator is used to chain components together. The output of the previous component is passed as input to the next.

## RunnableSequence

Under the hood, when you use the `|` operator, LCEL creates a `RunnableSequence`. A RunnableSequence is itself a Runnable, which means it can be invoked, streamed, batched, and transformed just like any other Runnable.

## RunnableParallel

To run multiple Runnables in parallel and combine their outputs, you can use `RunnableParallel`:

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    joke=prompt_joke | model,
    poem=prompt_poem | model,
)
result = chain.invoke({"topic": "bears"})
```

## Configurable Chains

LCEL supports runtime configuration through `.configurable_fields()` and `.configurable_alternatives()`. This lets you change parameters like temperature or even swap out entire components at runtime.

```python
model = ChatAnthropic(model="claude-sonnet-4-20250514").configurable_fields(
    temperature=ConfigurableField(id="llm_temperature")
)
chain = prompt | model
chain.invoke({"topic": "bears"}, config={"configurable": {"llm_temperature": 0.9}})
```
