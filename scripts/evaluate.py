"""
运行 RAG 评估

用法:
    python -m scripts.evaluate
    python -m scripts.evaluate --test-set ./eval/test_set.json --top-k 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.store import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import RAGGenerator
from src.evaluation.evaluator import RAGEvaluator
from rich.console import Console

console = Console()


def main():
    parser = argparse.ArgumentParser(description="运行 RAG 系统评估")
    parser.add_argument("--test-set", default="./eval/test_set.json", help="测试集路径")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="./eval/eval_report.json", help="JSON 报告路径")
    parser.add_argument("--summary", default="./eval/eval_summary.md", help="Markdown 摘要路径")
    args = parser.parse_args()

    console.print("\n[bold]📊 LangChain Docs RAG - 系统评估[/bold]\n")

    # 1. 加载向量存储
    store = VectorStore()
    if store.count() == 0:
        console.print("[red]✗ 没有数据！请先运行 ingest[/red]")
        sys.exit(1)
    console.print(f"[green]✓ 已加载 {store.count()} 个 chunks[/green]")

    # 2. 从 ChromaDB 取回全部 chunks 给 BM25 建索引
    console.print("[blue]⏳ 构建 BM25 索引...[/blue]")
    all_chunks = store.get_all_chunks()
    console.print(f"[green]✓ BM25 索引: {len(all_chunks)} docs[/green]")

    # 3. 初始化组件
    retriever = HybridRetriever(store, all_chunks)
    generator = RAGGenerator()

    # 4. 运行评估
    evaluator = RAGEvaluator(retriever, generator)
    results = evaluator.run_eval(args.test_set, top_k=args.top_k)

    # 5. 打印报告
    evaluator.print_report(results)

    # 6. 保存 JSON 报告
    evaluator.save_report(results, args.output)

    # 7. 生成 Markdown 摘要
    evaluator.save_summary(results, args.summary)


if __name__ == "__main__":
    main()
