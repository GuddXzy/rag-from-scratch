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
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir(PROJECT_ROOT)

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangChain Docs AI Assistant",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Translations ──────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "zh": {
        "page_title":          "LangChain 文档 AI 助手",
        "page_caption":        "基于 LangChain 官方文档的智能问答 | 混合检索 + Ollama qwen2.5:7b",
        "sidebar_title":       "🔗 LangChain Docs RAG",
        "lang_label":          "🌐 语言 / Language",
        "project_info":        "项目信息",
        "tech_stack":          (
            "**名称**: LangChain 文档 AI 助手\n\n"
            "**技术栈**:\n"
            "- Embeddings: `all-MiniLM-L6-v2`\n"
            "- 向量库: ChromaDB\n"
            "- 检索: Hybrid (Semantic + BM25 + RRF)\n"
            "- LLM: Ollama `qwen2.5:7b`\n"
            "- Frontend: Streamlit"
        ),
        "system_status":       "系统状态",
        "chunks_label":        "Chunks 总数",
        "chunk_size_label":    "Chunk 大小",
        "no_data_warning":     "ChromaDB 暂无数据，请先运行 ingest。",
        "system_ready":        "系统就绪 ✓  (模型: {})",
        "init_error":          "初始化失败: {}",
        "eval_header":         "评估指标 (chunk_size=512)",
        "composite_caption":   "综合得分: {:.4f} — 最优配置",
        "no_eval":             "eval/chunk_experiment_report.json 未找到",
        "sample_q_header":     "示例问题",
        "sample_questions": [
            "RAG 是什么？它是如何工作的？",
            "LangChain 中 Agent 和 Chain 有什么区别？",
            "LangGraph 是什么？什么时候应该使用它？",
            "如何给 LangChain Agent 添加记忆功能？",
            "LangChain 如何支持流式输出？",
        ],
        "metric_faithfulness":       "忠实度",
        "metric_answer_relevancy":   "答案相关性",
        "metric_context_precision":  "上下文精度",
        "metric_context_recall":     "上下文召回",
        "chat_placeholder":    "请输入您的问题...",
        "thinking_spinner":    "正在检索并生成回答...",
        "sources_expander":    "引用来源 (Sources)",
        "score_label":         "相关度",
        "no_data_answer": (
            "ChromaDB 中没有文档数据，请先运行:\n\n"
            "```bash\npython -m scripts.ingest --docs-dir ./data/langchain_docs\n```"
        ),
        "error_answer":        "发生错误: {}",
        "clear_btn":           "🗑️ 清空对话",
        "init_spinner":        "正在初始化 RAG 系统...",
    },
    "en": {
        "page_title":          "LangChain Docs AI Assistant",
        "page_caption":        "Q&A over LangChain official docs | Hybrid Retrieval + Ollama qwen2.5:7b",
        "sidebar_title":       "🔗 LangChain Docs RAG",
        "lang_label":          "🌐 Language / 语言",
        "project_info":        "Project Info",
        "tech_stack":          (
            "**Name**: LangChain Docs AI Assistant\n\n"
            "**Stack**:\n"
            "- Embeddings: `all-MiniLM-L6-v2`\n"
            "- Vector DB: ChromaDB\n"
            "- Retrieval: Hybrid (Semantic + BM25 + RRF)\n"
            "- LLM: Ollama `qwen2.5:7b`\n"
            "- Frontend: Streamlit"
        ),
        "system_status":       "System Status",
        "chunks_label":        "Total Chunks",
        "chunk_size_label":    "Chunk Size",
        "no_data_warning":     "No data in ChromaDB. Please run ingest first.",
        "system_ready":        "System ready ✓  (model: {})",
        "init_error":          "Init failed: {}",
        "eval_header":         "Eval Metrics (chunk_size=512)",
        "composite_caption":   "Composite: {:.4f} — best config",
        "no_eval":             "eval/chunk_experiment_report.json not found",
        "metric_faithfulness":       "Faithfulness",
        "metric_answer_relevancy":   "Answer Relevancy",
        "metric_context_precision":  "Context Precision",
        "metric_context_recall":     "Context Recall",
        "sample_q_header":     "Sample Questions",
        "sample_questions": [
            "What is RAG and how does it work?",
            "What is the difference between agents and chains?",
            "What is LangGraph and when should I use it?",
            "How to add memory to a LangChain agent?",
            "How does LangChain support streaming?",
        ],
        "chat_placeholder":    "Ask a question about LangChain...",
        "thinking_spinner":    "Retrieving and generating answer...",
        "sources_expander":    "Sources",
        "score_label":         "score",
        "no_data_answer": (
            "No documents in ChromaDB. Please run:\n\n"
            "```bash\npython -m scripts.ingest --docs-dir ./data/langchain_docs\n```"
        ),
        "error_answer":        "Error: {}",
        "clear_btn":           "🗑️ Clear chat",
        "init_spinner":        "Initialising RAG system...",
    },
}


def t(key: str, *args) -> str:
    """Get a translated string for the current language, optionally formatting args."""
    lang = st.session_state.get("lang", "zh")
    text = TRANSLATIONS[lang].get(key, TRANSLATIONS["zh"].get(key, key))
    return text.format(*args) if args else text


# ── Session state bootstrap ───────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "zh"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""


# ── Lazy-load RAG modules (cached so they are only initialised once) ──────────
@st.cache_resource(show_spinner=False)
def load_rag_components():
    from src.embeddings.store import VectorStore
    from src.generation.generator import RAGGenerator

    store = VectorStore()
    generator = RAGGenerator()
    chunks = store.get_all_chunks() if store.count() > 0 else []
    return store, generator, chunks


@st.cache_data(show_spinner=False)
def load_eval_report():
    report_path = PROJECT_ROOT / "eval" / "chunk_experiment_report.json"
    if report_path.exists():
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(t("sidebar_title"))

    # Language toggle ── placed at the very top for visibility
    lang_choice = st.radio(
        t("lang_label"),
        options=["中文", "English"],
        index=0 if st.session_state.lang == "zh" else 1,
        horizontal=True,
        key="lang_radio",
    )
    new_lang = "zh" if lang_choice == "中文" else "en"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.divider()

    # Project info
    st.subheader(t("project_info"))
    st.markdown(t("tech_stack"))

    st.divider()

    # System status
    st.subheader(t("system_status"))
    try:
        with st.spinner(t("init_spinner")):
            store, generator, chunks = load_rag_components()
        doc_count = store.count()

        from src import config as cfg
        col1, col2 = st.columns(2)
        col1.metric(t("chunks_label"), doc_count)
        col2.metric(t("chunk_size_label"), cfg.CHUNK_SIZE)

        if doc_count == 0:
            st.warning(t("no_data_warning"))
        else:
            st.success(t("system_ready", cfg.LLM_MODEL))
    except Exception as e:
        st.error(t("init_error", e))

    st.divider()

    # Evaluation metrics
    st.subheader(t("eval_header"))
    report = load_eval_report()
    if report:
        best = next(
            (r for r in report.get("results", []) if r["chunk_size"] == 512),
            None,
        )
        if best:
            m1, m2 = st.columns(2)
            m1.metric(t("metric_faithfulness"),      f"{best['faithfulness']:.3f}")
            m2.metric(t("metric_answer_relevancy"),  f"{best['answer_relevancy']:.3f}")
            m3, m4 = st.columns(2)
            m3.metric(t("metric_context_precision"), f"{best['context_precision']:.3f}")
            m4.metric(t("metric_context_recall"),    f"{best['context_recall']:.3f}")
            composite = report.get("best_config", {}).get("composite_score", 0)
            st.caption(t("composite_caption", composite))
    else:
        st.info(t("no_eval"))

    st.divider()

    # Sample questions
    st.subheader(t("sample_q_header"))
    for q in t("sample_questions"):
        if st.button(q, key=f"sq_{q[:20]}", use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    # Clear chat
    if st.button(t("clear_btn"), use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Helper: run the RAG pipeline ─────────────────────────────────────────────
def run_rag(question: str):
    store, generator, chunks = load_rag_components()

    if store.count() == 0:
        return t("no_data_answer"), []

    if chunks:
        from src.retrieval.hybrid import HybridRetriever
        retriever = HybridRetriever(store, chunks)
        results = retriever.search(question, k=5)
    else:
        results = store.search(question, k=5)

    gen = generator.generate(question=question, context_chunks=results)

    source_info = [
        {
            "source":  r.get("metadata", {}).get("source", "unknown"),
            "score":   r.get("score", 0),
            "content": r.get("content", ""),
        }
        for r in results[:3]
    ]

    return gen["answer"], source_info


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title(t("page_title"))
st.caption(t("page_caption"))

# Consume any pending question from sidebar sample buttons
pending = st.session_state.pop("pending_question", "")

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(t("sources_expander"), expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**[{i}] {src['source']}** — {t('score_label')}: `{src['score']:.4f}`"
                    )
                    st.caption(f"> {src['content'][:300].replace(chr(10), ' ')}...")

# Chat input
user_input = st.chat_input(t("chat_placeholder"), key="chat_input")
question = user_input or pending

if question:
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(t("thinking_spinner")):
            try:
                answer, sources = run_rag(question)
            except Exception as e:
                answer = t("error_answer", e)
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander(t("sources_expander"), expanded=False):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f"**[{i}] {src['source']}** — {t('score_label')}: `{src['score']:.4f}`"
                    )
                    st.caption(f"> {src['content'][:300].replace(chr(10), ' ')}...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
