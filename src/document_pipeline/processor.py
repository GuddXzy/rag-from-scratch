"""
文档处理管线 - 把 Loader 和 Chunker 串起来
一条命令完成: 加载文档 → 清洗 → 分块 → 输出
"""

from pathlib import Path
from rich.console import Console
from rich.progress import track

from .loader import DocumentLoader, Document
from .chunker import DocumentChunker, Chunk
from .. import config

console = Console()


class DocumentProcessor:
    """
    端到端文档处理管线
    
    用法:
        processor = DocumentProcessor("./data/langchain_docs")
        chunks = processor.run()
    """

    def __init__(self, docs_dir: str, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self.loader = DocumentLoader(docs_dir)
        self.chunker = DocumentChunker(
            chunk_size=chunk_size or config.CHUNK_SIZE,
            chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
        )

    def run(self) -> list[Chunk]:
        """执行完整的文档处理流程"""
        console.print("\n[bold blue]📄 Step 1/2: 加载文档...[/bold blue]")
        docs = self.loader.load_all()

        if not docs:
            console.print("[red]✗ 没有找到任何文档！请检查文档目录。[/red]")
            return []

        console.print(f"\n[bold blue]✂️  Step 2/2: 分块处理...[/bold blue]")
        chunks = self.chunker.chunk_documents(docs)

        # 过滤内容过短的 chunk（通常是列表页的单行条目，对检索无意义）
        min_len = 150
        before = len(chunks)
        chunks = [c for c in chunks if len(c.content) >= min_len]
        if before != len(chunks):
            console.print(f"  [dim]过滤掉 {before - len(chunks)} 个过短 chunk（< {min_len} 字符）[/dim]")

        # 打印统计信息
        self._print_stats(docs, chunks)
        return chunks

    @staticmethod
    def _print_stats(docs: list[Document], chunks: list[Chunk]) -> None:
        """打印处理统计"""
        total_chars = sum(len(c.content) for c in chunks)
        avg_chars = total_chars // max(len(chunks), 1)
        sources = set(c.metadata.get("source", "") for c in chunks)

        console.print("\n[bold green]═══ 处理完成 ═══[/bold green]")
        console.print(f"  📁 文档数量: {len(docs)}")
        console.print(f"  🧩 Chunk 数量: {len(chunks)}")
        console.print(f"  📏 平均 Chunk 长度: {avg_chars} 字符")
        console.print(f"  📂 来源文件数: {len(sources)}")
