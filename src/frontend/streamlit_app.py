"""
LangChain Docs AI Assistant — Streamlit Frontend

Run from project root:
    .venv/Scripts/python -m streamlit run src/frontend/streamlit_app.py
"""

import json
import sys
import os
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensure we can import from src.* when launched from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Change cwd so relative paths in config (chroma_db, etc.) resolve correctly
os.chdir(PROJECT_ROOT)

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="LangChain Docs AI Assistant",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy-load RAG modules (cached so they are only initialised once) ──────────
@st.cache_resource(show_spinner="正在初始化 RAG 系统...")
def load_rag_components():
    from src.embeddings.store import VectorStore
    from src.generation.generator import RAGGenerator
    from src.document_pipeline.chunker import Chunk

    store = VectorStore()
    generator = RAGGenerator()

    # Load all chunks for BM25 index
    chunks = store.get_all_chunks() if store.count() > 0 else []
    return store, generator, chunks


@st.cache_data(show_spinner=False)
def load_eval_report():
    report_path = PROJECT_ROOT / "eval" / "chunk_experiment_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Session state bootstrap ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role": "user"|"assistant", "content": str, "sources": [...]}

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""  # auto-fill from sample questions


# ── Sample questions ──────────────────────────────────────────────────────────
SAMPLE_QUESTIONS = [
    "What is RAG and how does it work?",
    "What is the difference between agents and chains in LangChain?",
    "What is LangGraph and when should I use it?",
    "How to add memory to a LangChain agent?",
    "How does LangChain support streaming?",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔗 LangChain Docs RAG")

    # Project info
    st.subheader("项目信息")
    st.markdown(
        """
        **名称**: LangChain Docs AI Assistant
        **技术栈**:
        - Embeddings: `all-MiniLM-L6-v2`
        - 向量库: ChromaDB
        - 检索: Hybrid (Semantic + BM25 + RRF)
        - LLM: Ollama `qwen2.5:7b`
        - Frontend: Streamlit
        """
    )

    st.divider()

    # System status
    st.subheader("系统状态")
    try:
        store, generator, chunks = load_rag_components()
        doc_count = store.count()

        from src import config as cfg
        col1, col2 = st.columns(2)
        col1.metric("ChromaDB Chunks", doc_count)
        col2.metric("Chunk Size", cfg.CHUNK_SIZE)

        if doc_count == 0:
            st.warning("ChromaDB 暂无数据，请先运行 ingest。")
        else:
            st.success(f"系统就绪 ✓  (模型: {cfg.LLM_MODEL})")
    except Exception as e:
        st.error(f"初始化失败: {e}")

    st.divider()

    # Evaluation metrics (best config = chunk_size 512)
    st.subheader("评估指标 (chunk_size=512)")
    report = load_eval_report()
    if report:
        best = next(
            (r for r in report.get("results", []) if r["chunk_size"] == 512),
            None,
        )
        if best:
            m1, m2 = st.columns(2)
            m1.metric("Faithfulness", f"{best['faithfulness']:.3f}")
            m2.metric("Answer Relevancy", f"{best['answer_relevancy']:.3f}")
            m3, m4 = st.columns(2)
            m3.metric("Context Precision", f"{best['context_precision']:.3f}")
            m4.metric("Context Recall", f"{best['context_recall']:.3f}")
            composite = report.get("best_config", {}).get("composite_score", 0)
            st.caption(f"Composite score: {composite:.4f} — 最优配置")
    else:
        st.info("eval/chunk_experiment_report.json 未找到")

    st.divider()

    # Sample questions
    st.subheader("示例问题")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sq_{q[:20]}", use_container_width=True):
            st.session_state.pending_question = q


# ── Helper: run the RAG pipeline ─────────────────────────────────────────────
def run_rag(question: str):
    """
    Directly invoke Python modules (no HTTP).
    Returns (answer: str, sources: list[dict])
    """
    store, generator, chunks = load_rag_components()

    if store.count() == 0:
        return (
            "ChromaDB 中没有文档数据，请先运行:\n\n"
            "```bash\npython -m scripts.ingest --docs-dir ./data/langchain_docs\n```",
            [],
        )

    # Retrieval — use HybridRetriever when chunks are available
    if chunks:
        from src.retrieval.hybrid import HybridRetriever
        retriever = HybridRetriever(store, chunks)
        results = retriever.search(question, k=5)
    else:
        results = store.search(question, k=5)

    # Generation
    gen = generator.generate(question=question, context_chunks=results)

    # Build source info list (top-3)
    source_info = []
    for r in results[:3]:
        source_info.append({
            "source": r.get("metadata", {}).get("source", "unknown"),
            "score": r.get("score", 0),
            "content": r.get("content", ""),
        })

    return gen["answer"], source_info


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("LangChain Docs AI Assistant")
st.caption("基于 LangChain 官方文档的智能问答系统 | Hybrid RAG + Ollama qwen2.5:7b")

# Consume any pending question from the sidebar sample buttons
pending = st.session_state.pop("pending_question", "")  # pop resets to ""

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("引用来源 (Sources)", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}] {src['source']}** — 相关度: `{src['score']:.4f}`"
                    )
                    preview = src["content"][:300].replace("\n", " ")
                    st.caption(f"> {preview}...")

# Chat input — pre-fill if a sample question was clicked
user_input = st.chat_input(
    "请输入您的问题...",
    key="chat_input",
)

# Merge: explicit typing takes priority; pending fills in otherwise
question = user_input or pending

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("正在检索并生成回答..."):
            try:
                answer, sources = run_rag(question)
            except Exception as e:
                answer = f"发生错误: {e}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("引用来源 (Sources)", expanded=False):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f"**[{i}] {src['source']}** — 相关度: `{src['score']:.4f}`"
                    )
                    preview = src["content"][:300].replace("\n", " ")
                    st.caption(f"> {preview}...")

    # Persist to session state
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
