"""
测试文档处理管线
运行: pytest tests/test_pipeline.py -v
"""

import tempfile
from pathlib import Path
import pytest

from src.document_pipeline.loader import DocumentLoader, Document
from src.document_pipeline.chunker import DocumentChunker, Chunk
from src.document_pipeline.processor import DocumentProcessor


# ============================================
# 测试数据
# ============================================
SAMPLE_MD = """# Test Document

## Section 1

This is the first section with some content about LangChain.
LangChain is a framework for building applications with large language models (LLMs).
It provides tools for chaining together prompts, retrievers, and output parsers.
Developers use LangChain to build chatbots, document Q&A systems, and agents.

## Section 2

This is the second section about RAG (Retrieval Augmented Generation).
RAG combines retrieval with generation for better answers by grounding the model
in external knowledge. Instead of relying solely on parametric memory, the model
is given relevant context chunks retrieved from a vector database at query time.

## Section 3

Agents use LLMs to decide which actions to take dynamically at each step.
Unlike chains with a fixed sequence of steps, agents can choose tools, call APIs,
or perform calculations based on the user query and the results of previous steps.
This makes agents flexible for open-ended tasks that require multi-step reasoning.
"""

SAMPLE_HTML = """
<html>
<head><title>Test</title></head>
<body>
<nav>Navigation bar</nav>
<main>
<h1>Main Content</h1>
<p>This is the important text about vector databases and their role in RAG systems.</p>
<p>ChromaDB is a lightweight, open-source vector database that is great for
prototyping and small-to-medium scale deployments. It supports cosine similarity
and runs fully in-process without any external infrastructure dependencies.</p>
<p>Other popular choices include Pinecone for managed cloud deployments and
Qdrant for self-hosted production use cases with advanced filtering capabilities.</p>
</main>
<footer>Footer content</footer>
</body>
</html>
"""


@pytest.fixture
def temp_docs_dir():
    """创建临时文档目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 写入测试文件
        Path(tmpdir, "test.md").write_text(SAMPLE_MD, encoding="utf-8")
        Path(tmpdir, "test.html").write_text(SAMPLE_HTML, encoding="utf-8")
        Path(tmpdir, "empty.md").write_text("", encoding="utf-8")  # 空文件，应被跳过
        Path(tmpdir, "ignored.pdf").write_bytes(b"fake pdf")  # 不支持的格式
        yield tmpdir


# ============================================
# Loader 测试
# ============================================
class TestDocumentLoader:

    def test_load_markdown(self, temp_docs_dir):
        loader = DocumentLoader(temp_docs_dir)
        doc = loader.load_single(Path(temp_docs_dir, "test.md"))
        assert doc is not None
        assert "LangChain" in doc.content
        assert doc.metadata["format"] == "md"

    def test_load_html_strips_tags(self, temp_docs_dir):
        loader = DocumentLoader(temp_docs_dir)
        doc = loader.load_single(Path(temp_docs_dir, "test.html"))
        assert doc is not None
        assert "vector databases" in doc.content
        # nav 和 footer 应该被移除
        assert "Navigation bar" not in doc.content
        assert "Footer content" not in doc.content

    def test_skip_empty_files(self, temp_docs_dir):
        loader = DocumentLoader(temp_docs_dir)
        doc = loader.load_single(Path(temp_docs_dir, "empty.md"))
        assert doc is None  # 空文件应返回 None

    def test_load_all(self, temp_docs_dir):
        loader = DocumentLoader(temp_docs_dir)
        docs = loader.load_all()
        # 应该只加载 md 和 html，跳过空文件和 pdf
        assert len(docs) == 2
        formats = {d.metadata["format"] for d in docs}
        assert formats == {"md", "html"}

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            DocumentLoader("/nonexistent/path/xyz")


# ============================================
# Chunker 测试
# ============================================
class TestDocumentChunker:

    def test_basic_chunking(self):
        doc = Document(
            content=SAMPLE_MD,
            metadata={"source": "test.md", "filename": "test.md", "format": "md"},
        )
        chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 0
        # 每个 chunk 都应有 metadata
        for c in chunks:
            assert c.metadata["source"] == "test.md"
            assert c.chunk_id != ""

    def test_small_doc_single_chunk(self):
        """小文档应该只产生1个chunk"""
        doc = Document(
            content="This is a short document.",
            metadata={"source": "short.md", "filename": "short.md", "format": "md"},
        )
        chunker = DocumentChunker(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1

    def test_chunk_overlap_works(self):
        """长文档的chunk之间应该有重叠"""
        long_text = "word " * 2000  # 约10000字符
        doc = Document(content=long_text, metadata={"source": "long.md", "filename": "long.md", "format": "md"})
        chunker = DocumentChunker(chunk_size=128, chunk_overlap=20)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        # 检查相邻chunk有重叠文本
        for i in range(len(chunks) - 1):
            tail = chunks[i].content[-50:]
            head = chunks[i + 1].content[:200]
            # 至少有一些公共词
            tail_words = set(tail.split())
            head_words = set(head.split())
            assert len(tail_words & head_words) > 0, "相邻chunks应有重叠"

    def test_batch_chunking(self, temp_docs_dir):
        """批量处理多个文档"""
        loader = DocumentLoader(temp_docs_dir)
        docs = loader.load_all()
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) >= len(docs)  # 至少每个文档产生1个chunk


# ============================================
# Processor 端到端测试
# ============================================
class TestDocumentProcessor:

    def test_full_pipeline(self, temp_docs_dir):
        processor = DocumentProcessor(temp_docs_dir)
        chunks = processor.run()
        assert len(chunks) > 0
        # 每个chunk都应有完整的metadata
        for c in chunks:
            assert "source" in c.metadata
            assert "chunk_id" in c.metadata
            assert len(c.content) > 0
