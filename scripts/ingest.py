"""
文档导入脚本
将文档目录中的文件 → 分块 → 向量化 → 存入 ChromaDB

用法:
    python -m scripts.ingest --docs-dir ./data/sample_docs
    python -m scripts.ingest --docs-dir ./data/sample_docs --reset  # 清空重导
"""

import argparse
import sys
from pathlib import Path

# 让 Python 能找到 src 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_pipeline.processor import DocumentProcessor
from src.embeddings.store import VectorStore
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="导入文档到 RAG 系统")
    parser.add_argument("--docs-dir", required=True, help="文档目录路径")
    parser.add_argument("--reset", action="store_true", help="清空现有数据后重新导入")
    parser.add_argument("--chunk-size", type=int, default=None, help="覆盖默认 chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="覆盖默认 chunk overlap")
    args = parser.parse_args()

    console.print("\n[bold]🚀 LangChain Docs RAG - 文档导入[/bold]\n")

    # 1. 初始化向量存储
    store = VectorStore()

    if args.reset:
        console.print("[yellow]⚠ 正在清空现有数据...[/yellow]")
        store.reset()

    # 2. 处理文档
    processor = DocumentProcessor(
        args.docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = processor.run()

    if not chunks:
        console.print("[red]✗ 没有产生任何 chunks，请检查文档目录[/red]")
        sys.exit(1)

    # 3. 写入向量数据库
    store.add_chunks(chunks)

    console.print(f"\n[bold green]✅ 导入完成！共 {store.count()} 个 chunks 已就绪[/bold green]")
    console.print(f"   数据存储在: {store.persist_dir}")
    console.print(f"\n   下一步: python -m scripts.query '你的问题'\n")


if __name__ == "__main__":
    main()
