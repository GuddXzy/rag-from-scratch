"""
命令行查询脚本 - 快速测试 RAG 系统
不需要启动 API 服务就能测试

用法:
    python -m scripts.query "What is LCEL?"
    python -m scripts.query "How to use agents?" --top-k 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.store import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import RAGGenerator
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def main():
    parser = argparse.ArgumentParser(description="查询 RAG 系统")
    parser.add_argument("question", help="你的问题")
    parser.add_argument("--top-k", type=int, default=5, help="检索结果数量")
    parser.add_argument("--no-generate", action="store_true", help="只检索不生成 (不消耗API)")
    args = parser.parse_args()

    store = VectorStore()

    if store.count() == 0:
        console.print("[red]✗ 没有数据！请先运行: python -m scripts.ingest --docs-dir ./data/sample_docs[/red]")
        sys.exit(1)

    console.print(f"\n[bold blue]🔍 检索: {args.question}[/bold blue]\n")

    # 1. 混合检索 (语义 + BM25 + RRF)
    all_chunks = store.get_all_chunks()
    retriever = HybridRetriever(store, all_chunks)
    results = retriever.search(args.question, k=args.top_k)

    console.print(f"[dim]找到 {len(results)} 个相关片段:[/dim]\n")
    for i, r in enumerate(results, 1):
        source = r["metadata"].get("source", "?")
        score = r["score"]
        preview = r["content"][:150].replace("\n", " ")
        console.print(f"  {i}. [cyan]{source}[/cyan] (相关度: {score:.4f})")
        console.print(f"     {preview}...\n")

    if args.no_generate:
        return

    # 2. 生成回答
    console.print("[bold blue]🤖 生成回答...[/bold blue]\n")

    generator = RAGGenerator()
    result = generator.generate(question=args.question, context_chunks=results)

    console.print(Panel(
        Markdown(result["answer"]),
        title="RAG 回答",
        border_style="green",
    ))

    console.print(f"\n[dim]模型: {result['model']} | "
                  f"Token: {result['tokens_used']['input']}in + {result['tokens_used']['output']}out | "
                  f"来源: {', '.join(result['sources'])}[/dim]\n")


if __name__ == "__main__":
    main()
