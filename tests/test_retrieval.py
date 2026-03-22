"""
测试向量存储和检索模块
运行: pytest tests/test_retrieval.py -v

注意: 这个测试会真正加载 embedding 模型 (~80MB)，首次运行会下载模型
"""

import gc
import shutil
import tempfile
import pytest

from src.document_pipeline.chunker import Chunk
from src.embeddings.store import VectorStore
from src.retrieval.hybrid import HybridRetriever


# ============================================
# 测试 chunks
# ============================================
TEST_CHUNKS = [
    Chunk(
        content="LCEL is LangChain Expression Language. It uses the pipe operator to chain components together.",
        metadata={"source": "lcel.md", "filename": "lcel.md", "format": "md"},
        chunk_id="lcel_0_0",
    ),
    Chunk(
        content="RAG stands for Retrieval Augmented Generation. It retrieves documents and passes them to an LLM.",
        metadata={"source": "rag.md", "filename": "rag.md", "format": "md"},
        chunk_id="rag_0_0",
    ),
    Chunk(
        content="ChromaDB is a lightweight vector database. It is great for prototyping and local development.",
        metadata={"source": "vectordb.md", "filename": "vectordb.md", "format": "md"},
        chunk_id="vectordb_0_0",
    ),
    Chunk(
        content="Agents use LLMs to dynamically decide which tools to call. ReAct is a common agent pattern.",
        metadata={"source": "agents.md", "filename": "agents.md", "format": "md"},
        chunk_id="agents_0_0",
    ),
    Chunk(
        content="Python is a programming language widely used in machine learning and data science applications.",
        metadata={"source": "python.md", "filename": "python.md", "format": "md"},
        chunk_id="python_0_0",
    ),
]


@pytest.fixture
def vector_store():
    """创建临时 VectorStore"""
    tmpdir = tempfile.mkdtemp()
    store = VectorStore(
        persist_dir=tmpdir,
        collection_name="test_collection",
    )
    store.add_chunks(TEST_CHUNKS)
    yield store
    # Windows: ChromaDB 持有 SQLite 文件句柄，需先释放引用再删除临时目录
    store._client = None
    store._collection = None
    gc.collect()
    shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================
# VectorStore 测试
# ============================================
class TestVectorStore:

    def test_add_and_count(self, vector_store):
        assert vector_store.count() == len(TEST_CHUNKS)

    def test_search_returns_results(self, vector_store):
        results = vector_store.search("What is LCEL?", k=3)
        assert len(results) == 3
        # 每个结果都应有正确的字段
        for r in results:
            assert "content" in r
            assert "metadata" in r
            assert "score" in r
            assert 0 <= r["score"] <= 1

    def test_search_relevance(self, vector_store):
        """检查语义检索是否返回正确的结果"""
        results = vector_store.search("What is a vector database?", k=2)
        # ChromaDB 相关的 chunk 应该排在前面
        top_content = results[0]["content"]
        assert "ChromaDB" in top_content or "vector" in top_content

    def test_search_lcel(self, vector_store):
        results = vector_store.search("How to use pipe operator in LangChain?", k=2)
        top_content = results[0]["content"]
        assert "LCEL" in top_content or "pipe" in top_content

    def test_reset(self, vector_store):
        vector_store.reset()
        # reset 后 collection 被删除，重新访问会创建空的
        assert vector_store.count() == 0


# ============================================
# HybridRetriever 测试
# ============================================
class TestHybridRetriever:

    @pytest.fixture
    def retriever(self, vector_store):
        return HybridRetriever(vector_store, TEST_CHUNKS)

    def test_hybrid_search(self, retriever):
        results = retriever.search("What is LCEL?", k=3)
        assert len(results) == 3
        for r in results:
            assert "content" in r
            assert "score" in r
            assert r.get("source") == "hybrid"

    def test_hybrid_beats_keyword_on_semantic(self, retriever):
        """语义查询: 混合检索应该能理解同义词"""
        results = retriever.search("chain components together", k=2)
        top = results[0]["content"]
        assert "LCEL" in top or "chain" in top or "pipe" in top

    def test_bm25_catches_exact_terms(self, retriever):
        """关键词查询: BM25应该精确匹配术语"""
        results = retriever.search_bm25_only("ChromaDB", k=2)
        assert len(results) > 0
        assert "ChromaDB" in results[0]["content"]

    def test_semantic_only(self, retriever):
        results = retriever.search_semantic_only("vector database for local use", k=2)
        assert len(results) == 2

    def test_bm25_only(self, retriever):
        results = retriever.search_bm25_only("ReAct agent pattern", k=2)
        assert len(results) > 0
