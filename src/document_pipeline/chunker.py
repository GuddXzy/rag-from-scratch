"""
文本分块器 (Chunker)

面试必问: 为什么 chunk_size=512, overlap=50?
- 512 tokens 大约是一个完整段落，足够包含一个概念
- 太小(128): 上下文不足，检索到的内容断章取义
- 太大(2048): 包含太多无关信息，稀释了相关性
- overlap=50: 防止关键信息恰好被切断在边界上

面试加分: 你可以展示不同 chunk_size 的评估对比实验
"""

from dataclasses import dataclass, field
from .loader import Document
from rich.console import Console

console = Console()


@dataclass
class Chunk:
    """一个文本块"""
    content: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __repr__(self) -> str:
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"Chunk(id={self.chunk_id}, len={len(self.content)}, preview='{preview}')"


class DocumentChunker:
    """
    将长文档切分为小块 (chunks)
    
    策略: 按 Markdown 标题分段 → 再按 token 数切分
    这比简单按字符数切分更好，因为尊重了文档的语义结构
    
    用法:
        chunker = DocumentChunker(chunk_size=512, overlap=50)
        chunks = chunker.chunk_document(doc)
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 粗略估算: 1 token ≈ 4 个英文字符 (中文约 1.5 字符)
        self.char_size = chunk_size * 4
        self.char_overlap = chunk_overlap * 4

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """将一篇文档切分为多个 Chunk"""
        # 第一步: 按 Markdown 标题 (#, ##, ###) 做初步分段
        sections = self._split_by_headers(doc.content)

        # 第二步: 对每个段落，如果太长就再按字符数切分
        chunks: list[Chunk] = []
        for section_idx, section in enumerate(sections):
            if len(section.strip()) < 20:
                continue

            sub_chunks = self._split_by_size(section)
            for sub_idx, text in enumerate(sub_chunks):
                chunk_id = f"{doc.metadata.get('filename', 'unknown')}_{section_idx}_{sub_idx}"
                chunks.append(Chunk(
                    content=text.strip(),
                    metadata={
                        **doc.metadata,
                        "chunk_id": chunk_id,
                        "section_index": section_idx,
                        "sub_index": sub_idx,
                    },
                    chunk_id=chunk_id,
                ))

        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """批量切分多篇文档"""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))

        console.print(
            f"[green]✓ {len(docs)} 篇文档 → {len(all_chunks)} 个 chunks "
            f"(avg {sum(len(c.content) for c in all_chunks) // max(len(all_chunks), 1)} chars/chunk)[/green]"
        )
        return all_chunks

    @staticmethod
    def _split_by_headers(text: str) -> list[str]:
        """按 Markdown 标题拆分，保留标题行作为每段的开头"""
        import re
        # 匹配 # 开头的标题行
        pattern = r"(?=^#{1,4}\s+)", 
        parts = re.split(r"(?=\n#{1,4}\s+)", text)
        return [p for p in parts if p.strip()]

    def _split_by_size(self, text: str) -> list[str]:
        """按字符数切分，带重叠"""
        if len(text) <= self.char_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.char_size

            # 尝试在句号、换行处切断，而不是生硬切断
            if end < len(text):
                # 往回找最近的段落/句子结束位置
                for sep in ["\n\n", "\n", ". ", "。", "；"]:
                    last_sep = text.rfind(sep, start + self.char_size // 2, end)
                    if last_sep != -1:
                        end = last_sep + len(sep)
                        break

            chunks.append(text[start:end])
            start = end - self.char_overlap  # 带重叠地向前移动

        return chunks
