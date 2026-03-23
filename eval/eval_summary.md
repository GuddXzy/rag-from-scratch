# RAG 评估报告

> 测试用例: 10 条 | 模型: qwen2.5:7b (Ollama) | Embedding: all-MiniLM-L6-v2

## 汇总指标

| 指标 | 得分 | 可视化 | 说明 |
| --- | ---: | --- | --- |
| Faithfulness      | 0.281  | `███░░░░░░░`  | 回答是否忠于检索上下文（防幻觉） |
| Answer Relevancy  | 0.652  | `███████░░░`  | 回答是否切题 |
| Context Precision | 0.235 | `██░░░░░░░░` | 相关文档是否排在检索结果前列 |
| Context Recall    | 0.504 | `█████░░░░░` | ground truth 信息是否被检索覆盖 |
| Avg Latency       | 5158 ms | — | 平均端到端响应时间 |

## 逐条明细

| # | 问题 | Faith. | Relev. | C.Prec | C.Rec | 来源文件 | ms |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | What is RAG and how does it work? | 0.37 | 0.75 | 0.20 | 0.35 | langchain\_philosophy.md, langchain\_retrieval.md | 19707 |
| 2 | What vector stores does LangChain support? | 0.29 | 0.75 | 0.17 | 0.40 | integrations\_vectorstores.md, integrations\_providers\_chro | 3170 |
| 3 | What is the difference between agents and chains in Lan… | 0.29 | 0.43 | 0.30 | 0.45 | concepts\_products.md, langchain\_philosophy.md | 2777 |
| 4 | What is LangGraph and when should I use it? | 0.31 | 0.83 | 0.23 | 0.61 | concepts\_products.md, langgraph\_application-structure.md,  | 2052 |
| 5 | How to add memory to a LangChain agent? | 0.34 | 0.60 | 0.30 | 0.59 | langchain\_studio.md, langchain\_short-term-memory.md, langc | 3737 |
| 6 | What retrieval strategies are available in LangChain? | 0.26 | 0.40 | 0.11 | 0.36 | langchain\_messages.md, integrations\_providers\_aws.md, lan | 2137 |
| 7 | How does LangChain support streaming? | 0.16 | 0.33 | 0.32 | 0.67 | langchain\_streaming.md, langchain\_models.md, langgraph\_st | 3676 |
| 8 | What tools and integrations does LangChain support? | 0.07 | 0.80 | 0.24 | 0.64 | langchain\_models.md, integrations\_providers\_overview.md,  | 2768 |
| 9 | How to build a knowledge base for RAG in LangChain? | 0.37 | 0.88 | 0.19 | 0.40 | langchain\_philosophy.md, langchain\_retrieval.md, learn.md | 6634 |
| 10 | What is structured output in LangChain and how to use i… | 0.35 | 0.75 | 0.31 | 0.57 | langchain\_agents.md, langchain\_philosophy.md, langchain\_s | 4920 |

## 逐条问答详情

### Case 1: What is RAG and how does it work?

**Ground Truth:** RAG (Retrieval-Augmented Generation) enhances LLMs by retrieving relevant external knowledge at query time. It has two phases: indexing (documents loaded, split, embedded into a vector store) and retrieval (query is embedded, similar chunks fetched, passed to LLM with the question).

**Generated Answer:**

RAG (Retrieval-Augmented Generation) is a method used in the field of artificial intelligence to enhance the abilities of language models by integrating them with retrieval and reasoning capabilities. This integration allows LLMs to produce answers that are not only more coherent but also grounded in data, making the responses contextually relevant and accurate.

### How RAG Works

There are typically two types of RAG architectures:

1. **2-step RAG**:
- **Retrieval Step**: The document retrieval part, which involves searching for related documents from a large dataset or index.
- **Generation (Reasoning) Step**: Based on the retrieved context, the LLM provides an answer.

2. **Agentic RAG**:
In this architecture, the LLM decides whether and how to retrieve documents during reasoning. It enables a higher degree of flexibility in generating answers but could be more unpredictable compared to 2-step RAG because it's harder for humans to understand when and why specific actions are taken by the model.

Here are detailed explanations:

- **2-step RAG**: In this type, after the retrieval step (searching relevant documents), an appropriate generation step occurs (generation can involve creating a response based on what was found). This makes it more predictable but less flexible because LLMs usually make only one call to each tool or query.

### Examples and Use-cases

LangChain provides two versions of RAG:
- The **2-step RAG** is suitable for use in domains like FAQs, documentation bots; it's straightforward and efficient.
- A **Hybrid RAG**: It combines the advantages of 2-step RAG with Agentic RAG by incorporating checks on retrieval results (retrieval validation) and answering quality after a generation step.

Both these architectures enable flexible reasoning processes tailored to specific needs. 

### References

[来源: langchain_retrieval.md]

**Retrieved Sources:** langchain_retrieval.md, langchain_retrieval.md, langchain_retrieval.md, langchain_retrieval.md, langchain_philosophy.md

**Scores:** Faithfulness=0.37 | Relevancy=0.75 | Precision=0.20 | Recall=0.35 | Latency=19707ms

---

### Case 2: What vector stores does LangChain support?

**Ground Truth:** LangChain supports many vector stores including Chroma, Pinecone, PGVector, and others. They store embeddings and support similarity search to retrieve relevant document chunks.

**Generated Answer:**

根据现有文档，LangChain支持使用不同的嵌入模型初始化的自定义 `InMemoryVectorStore`。此外，它还提供了一个名为 Chroma 的包装器来访问 Chroma 矢量数据库。

1. **内置的.vectorstores**：用户可以通过向其提供一个嵌入模型来初始化一个自定义的 `InMemoryVectorStore`。
   - 示例代码：
     ```python
     from langchain_core.vectorstores import InMemoryVectorStore
     vector_store = InMemoryVectorStore(embedding=SomeEmbeddingModel())
     ```
   
2. **外部的.vectorstores包装器**：LangChain 还提供了一个围绕 Chroma 矢量数据库的包装器，允许用户用作 `.vectorstore` 容器。这是一个用于语义搜索或样本选择的功能。
   - 示例代码：
     ```python
     from langchain_chroma import Chroma
     ```

因此，它支持的矢量存储包括但不限于自定义的 `InMemoryVectorStore` 和通过其包裹的 Chromium 矢量数据库。

至于其他内置的矢量存储库，文档没有提及其他信息。[来源: integrations_vectorstores.md] [来源: integrations_providers_chroma.md]

参考来源:
- [来源: integrations_vectorstores.md]
- [来源: integrations_providers_chroma.md]

**Retrieved Sources:** integrations_providers_overview.md, integrations_vectorstores.md, integrations_vectorstores.md, integrations_providers_chroma.md, integrations_document_loaders.md

**Scores:** Faithfulness=0.29 | Relevancy=0.75 | Precision=0.17 | Recall=0.40 | Latency=3170ms

---

### Case 3: What is the difference between agents and chains in LangChain?

**Ground Truth:** In chains the sequence of actions is hardcoded. In agents a language model acts as a reasoning engine to dynamically decide which tools to call and in what order, enabling flexible multi-step workflows.

**Generated Answer:**

在LangChain中，"agents"和“chains”是两种不同层次的概念。我将分点解释这两者之间的差异。

1. **Agents**：Agents代表一个完整的智能体应用程序逻辑，它是基于一系列操作步骤来完成特定任务的集合。例如，一个简单的Agent可以包括读取数据（如从数据库或互联网抓取）、处理该数据、生成输出等内容。LangChain中的大多数应用都是通过Agent来构建和运行的。对于需要低级控制或者应对复杂非决定型任务的应用来说，代理是一个非常重要的层。

2. **Chains**：Chains则提供了一种高层次的抽象方式，用于创建和管理一组操作步骤。一个Chains可以在多个Model之间传递数据，并将它们组合成一系列逻辑一致的任务流程。从技术角度看，Chains可以看作是一系列操作指令（Chains或ReAct Agent）。用户在使用LangChain构建应用程序时需要了解这一点。

根据现有文档，我无法找到关于这个问题的确切答案。

**Retrieved Sources:** concepts_products.md, langchain_philosophy.md, langchain_philosophy.md, concepts_products.md, concepts_products.md

**Scores:** Faithfulness=0.29 | Relevancy=0.43 | Precision=0.30 | Recall=0.45 | Latency=2777ms

---

### Case 4: What is LangGraph and when should I use it?

**Ground Truth:** LangGraph is a low-level agent orchestration framework for building stateful, multi-step workflows. Use it when you need cycles, branching, durable execution, human-in-the-loop patterns, or fine-grained control over agent state.

**Generated Answer:**

LangGraph is a framework designed to handle the orchestration of stateful, long-running agents with low-level control. Agents in LangGraph are managed by an underlying system that allows you to define how different parts interact to achieve your desired result.

You should use LangGraph when:

1. You require fine-grained, low-level control over agent orchestration.
2. When durable execution is necessary for agents designed to run for a long period and retain state.
3. In scenarios where complex workflows involving deterministic steps combined with agentic (i.e., decision-based) behaviors are needed.
4. For production systems that necessitate an infrastructure ready for the deployment of these agent systems.

These circumstances highlight how LangGraph can address challenges in managing multi-agent environments, particularly those which need to maintain state and exhibit a hybrid behavior integrating automatic actions with manual strategic decisions.

**Retrieved Sources:** langgraph_application-structure.md, langgraph_application-structure.md, langgraph_application-structure.md, learn.md, concepts_products.md

**Scores:** Faithfulness=0.31 | Relevancy=0.83 | Precision=0.23 | Recall=0.61 | Latency=2052ms

---

### Case 5: How to add memory to a LangChain agent?

**Ground Truth:** Pass a checkpointer to create_agent for short-term (thread-level) memory, or a store for long-term (cross-session) memory. InMemoryStore works for development; use PostgreSQL or Redis for production.

**Generated Answer:**

为了向LangChain代理添加记忆，您可以根据代理的使用情况和所需的持久性选择合适的长期存储方式。主要有两种选项：InMemoryStore 和 PostgreSQL。

1. **使用 InMemoryStore 添加内存（仅限于本地、单个线程）**
   - **优点**：简单直接，适合简单的实验或本地开发环境。
   ```python
   from langchain.agents import create_agent
   from langgraph.store.memory import InMemoryStore
   
   store = InMemoryStore()
   
   agent = create_agent(
       "gpt-5",
       tools=[get_user_info],
       memory=store,  # 添加内存（使用 InMemoryStore）
   )
   ```

2. **将记忆存储在 PostgreSQL 数据库中**
   - **优点**：适合需要较长时间持久性和跨应用程序共享信息的情况。
   ```python
   from langchain.agents import create_agent
   from sqlmodel import create_engine, Session
   
   engine = create_engine("postgresql://user:password@host/db_name")
   
   store = MemoryStore(engine)  # 使用 PostgreSQL 的记忆存储
   
   agent = create_agent(
       "gpt-5",
       tools=[get_user_info],
       memory=store,  # 添加长时记忆
   )
   ```

在这些示例中，`create_agent` 函数用于创建代理实例，并将一个 `memory` 参数设置为您的存储对象来添加记忆。

请根据需要和环境选择最适合的持久性解决方案。
[来源: langchain_long-term-memory.md]

**Retrieved Sources:** langchain_long-term-memory.md, langchain_short-term-memory.md, langchain_long-term-memory.md, concepts_memory.md, langchain_studio.md

**Scores:** Faithfulness=0.34 | Relevancy=0.60 | Precision=0.30 | Recall=0.59 | Latency=3737ms

---

### Case 6: What retrieval strategies are available in LangChain?

**Ground Truth:** LangChain supports 2-Step RAG (retrieve then generate), Agentic RAG (agent decides when to retrieve), and Hybrid RAG (combining vector search with keyword search). Each component like loaders, splitters, embeddings and vector stores is modular and swappable.

**Generated Answer:**

在 LangChain 中，有几种检索策略可供选择。以下是对这些策略的解释：

1. **Agentic Retrieval-Augmented Generation (RAG)**: 通过使用具有工具（如文档加载器、Web API 或数据库查询）的代理进行基于知识推理来改进 LLM 的答案。在这种方法中，当需要查询信息时，代理会决定何时以及如何检索这些信息。

2. **Amazon Kendra**: 是 `AWS` 提供的一种智能搜索服务，使用先进的自然语言处理（NLP）和机器学习算法来增强组织内各种数据源之间的强大搜索能力。

这些策略都允许从外部获取知识库中的相关信息，并为 LLM 增加上下文以回答用户查询的问题。根据现有文档，我无法找到关于这个问题的确切答案。

**Retrieved Sources:** langchain_retrieval.md, langchain_messages.md, learn.md, langchain_retrieval.md, integrations_providers_aws.md

**Scores:** Faithfulness=0.26 | Relevancy=0.40 | Precision=0.11 | Recall=0.36 | Latency=2137ms

---

### Case 7: How does LangChain support streaming?

**Ground Truth:** LangChain agents support streaming of tokens and intermediate steps. You can stream both the final answer tokens and tool call events using the agent's stream() method, enabling real-time UI updates.

**Generated Answer:**

LangChain 主要支持通过以下几种方法进行流式处理：

1. **Use with any LLM**: 使用 `stream_mode="custom"`，你可以为任何LLM API实现自定义流式传输。用户可以使用自己的流式客户端或外部服务来实现流式传输接口。这使得LangGraph能够针对定制搭建方案非常灵活。

2. **Auto-streaming Chat Models**: LangChain 提供自动启用流式模式的功能，即使在不显式的调用流式方法的情况下，也允许将整体应用程序的内容流式处理到聊天模型。例如，在LangGraph代理节点中，你可以通过调用`model.invoke()`来执行实际操作，而LangChain会将它们自动委托给流式模式。

3. **Streaming in LangGraph Agents**: 使用非流式的方法 (如 `invoke()`），但即使在进行流式处理（streaming）的整体应用程序时，Chat Model本身仍然继续执行。在这种情况下，当聊天模型正在运行时,LangChain 会确保在其回调系统中调用手动流式标记 (`on_llm_new_token`) 处理事件。

4. **Real-time Updates in LangGraph**: 当使用语言生成模型（`invoke()`）进行操作调用的时候，如果LLM的响应仍在继续生成，则用户界面在LLM生成完整响应前就能预览聊天模型的输出。这样显著提升了用户体验，特别适用于处理LLM的延迟。

这些方法共同确保了LangChain支持流式传输，并有效提高了基于LLM的应用程序的实时性及其用户体验。

**Retrieved Sources:** langgraph_streaming.md, langchain_models.md, langchain_streaming.md, langchain_models.md, langgraph_streaming.md

**Scores:** Faithfulness=0.16 | Relevancy=0.33 | Precision=0.32 | Recall=0.67 | Latency=3676ms

---

### Case 8: What tools and integrations does LangChain support?

**Ground Truth:** LangChain provides built-in tools and supports custom tools via the @tool decorator. It integrates with providers like OpenAI, Anthropic, Google, Ollama, HuggingFace, AWS Bedrock, and many others.

**Generated Answer:**

根据现有文档，LangChain 支持多种工具和集成，具体涵盖以下方面：

1. **聊天模型**：
   - LangChain 支持与不同的聊天模型提供商进行交互，包括 OpenAI、Anthropic 等。

2. **嵌入式模型**（Embedding models）：
   - 未在文档中详细说明相关支持及集成的模型。
   
3. **工具和脚手架工具集**（Tools and toolkits）：
   - 文档片段没有提供更多细节。
   
4. **文件和文档加载器**（Document loaders）：
   - 文档中提及 LangChain 提供不同的文件存储库。
   
5. **向量存储器**（Vector stores）：
   - 未详细说明具体支持的模型和服务。

总体来说，LangChain 提供了一个广泛的集成生态体系，允许开发者与多种第三方服务对接，不过某些特定领域如嵌入模型的具体支持细节并未在现有文档中有进一步介绍。关于工具集的整合或使用的文档信息也有所欠缺。建议直接查阅有关章节获取更详细的资料。

[来源: integrations_providers_overview.md]

**Retrieved Sources:** integrations_providers_overview.md, learn.md, langgraph_overview.md, langchain_models.md, concepts_products.md

**Scores:** Faithfulness=0.07 | Relevancy=0.80 | Precision=0.24 | Recall=0.64 | Latency=2768ms

---

### Case 9: How to build a knowledge base for RAG in LangChain?

**Ground Truth:** Use document loaders to ingest data, text splitters to chunk it into manageable pieces, an embedding model to convert chunks to vectors, and a vector store to persist and search them. The pipeline is modular so each component can be swapped independently.

**Generated Answer:**

To build a knowledge base for Retrieval-Augmented Generation (RAG) in LangChain, you can create your own custom knowledge base from existing data or connect an existing knowledge base as a tool. Here are two methods based on the provided documentation:

### Method 1: Build a Custom Knowledge Base

You can use LangChain's document loaders and vector stores to build a new knowledge base from your own data.

- **Document Loaders**: These are tools that read documents, be it websites or internal company documentation.
  
- **Vector Stores**: LangChain supports various vector store backends like Pinecone and Weaviate which help in storing and retrieving vectorized document representations efficiently. The tool `langchain.vectorstores` provides a unified interface for interacting with these vector stores.

Here is an example of how to use the `docs_loader.py` provided by LangChain:

```python
from langchain.document_loaders import UnstructuredHTMLLoader

# Load your HTML file or URL
loader = UnstructuredHTMLLoader('path_to_your_file')
pages = loader.load()

# Initialize a vector store using OpenSearchVectorStore as the backend.
from langchain.vectorstores.opensearch_vector_store import OpenSearchVectorStore 
from opensearchpy import OpenSearch

es_client = OpenSearch(
    hosts=["localhost:9200"],
)
store = OpenSearchVectorStore(es_client, pages)

# Connect your document loader & vector store as a tool.
@tool
def custom_knowledge_base(url_or_path: str) -> List[str]:
    """Retrieve documents from a URL or local file."""
    if url_or_path.startswith('http://') or url_or_path.startswith('https://'):
        # Fetch pages via HTTP requests and load them using UnstructuredHTMLLoader
        with open(url_or_path, "r") as f:
            text = loader.load()[0]['text']
    else:
        text = loader.load()[0].page_content

    return [text]

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[custom_knowledge_base], # A tool for retrieval using custom document loader
)
```

### Method 2: Connect an Existing Knowledge Base

If you already have a knowledge base (SQL database, CRM, or internal documentation system), you can connect it as a tool:

1. **Connect the Knowledge Base**: Access your existing knowledge source.
2. **Use LangChain's Tools** to interact with it.

Refer to the provided examples and tutorials in the documentation for detailed steps:
- [Building a Minimal RAG Workflow](#building-a-minimal-rrag-workflow).
- [Using Agentic Retrieval-Augmented Generation(RAG)](#agentic-rag).

### References
[来源: langchain_retrieval.md]

**Retrieved Sources:** langchain_retrieval.md, langchain_retrieval.md, langchain_retrieval.md, langchain_philosophy.md, learn.md

**Scores:** Faithfulness=0.37 | Relevancy=0.88 | Precision=0.19 | Recall=0.40 | Latency=6634ms

---

### Case 10: What is structured output in LangChain and how to use it?

**Ground Truth:** Structured output lets you constrain LLM responses to a defined schema (e.g. a Pydantic model). Use model.with_structured_output(Schema) to get typed, validated JSON responses instead of raw text.

**Generated Answer:**

Structured output in LangChain allows agents to return data in a predefined or predictable format, such as JSON objects, Pydantic models, or dataclasses, rather than parsing the raw natural language responses [来源: langchain_structured-output.md]。

为了使用结构化输出，可以通过 `response_format` 参数设置自定义的响应格式。支持两种方法之一：

1. **ProviderStrategy**: 在模型和模型提供商都直接支持自然形式输出时使用。
2. **ToolStrategy**: 适用于不需要与工具同时处理结构化输出的情况。

以下是如何进行具体配置的例子：

- 当您希望让Agent以特定格式返回结果时，可以通过设置 `response_format` 参数为对应的自定义响应类型来实现该功能。例如：给定一个 `UserQueryResponseSchema` 类型的 `StructuredResponseT` (这是一个数据类)，您可以像下面这样定义参数：

```python
user_query_response = UserQueryResponseSchema()
```

- 当您希望自定义响应时，可以使用 `ToolStrategy[StructuredResponseT]` 参数来实现。在这种情况下，需要告知模型如何解析非结构化输入，并以期望的格式生成响应代码。
  
例如:

```python
response_format: ToolStrategy[UserQueryResponseSchema]
```

此外，“langchain>=1.1” 版本将支持动态读取模型预设信息中的 native structured output 功能，通过在构建模型时提供自定义模型配置 (model profile)，您可以在未明确指定 `response_format` 的情况下实现结构化响应功能。例如：

```python
custom_profile = {
    "structured_output": True,
    # ...
}
model_with_structured_output = init_chat_model("...", profile=custom_profile)
```

总结一下，在使用 LangChain 进行相关操作时，可通过 `create_agent`, 修改模型配置 (custom_profile) 或在 `agent_params` 中指定相应的 `response_format` 参数等方式实现结构化输出的配置及应用。

参考信息来自 [来源: langchain_structured-output.md]。

**Retrieved Sources:** langchain_structured-output.md, langchain_structured-output.md, langchain_agents.md, langchain_agents.md, langchain_philosophy.md

**Scores:** Faithfulness=0.35 | Relevancy=0.75 | Precision=0.31 | Recall=0.57 | Latency=4920ms

---
