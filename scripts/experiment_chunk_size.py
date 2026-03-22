"""
Chunk Size 对比实验
对比 chunk_size = 256 / 512 / 1024 三种配置的 RAG 检索效果

用法:
    python -m scripts.experiment_chunk_size
"""

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.document_pipeline.processor import DocumentProcessor
from src.embeddings.store import VectorStore
from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import RAGGenerator
from src.evaluation.evaluator import RAGEvaluator
from src import config
from rich.console import Console
from rich.table import Table

console = Console()

DOCS_DIR    = "./data/langchain_docs"
TEST_SET    = "./eval/test_set.json"
CHUNK_SIZES = [256, 512, 1024]
TOP_K       = 5


# ──────────────────────────────────────────────
# 单轮实验
# ──────────────────────────────────────────────

def run_one(chunk_size: int) -> dict:
    chunk_overlap   = round(chunk_size * 0.2)
    collection_name = f"langchain_docs_{chunk_size}"

    console.rule(f"[bold cyan]chunk_size={chunk_size}  overlap={chunk_overlap}[/bold cyan]")

    # 1. 创建独立 collection，先清理旧数据
    store = VectorStore(
        collection_name=collection_name,
        persist_dir=config.CHROMA_PERSIST_DIR,
    )
    store.reset()
    # Windows: 确保 SQLite 锁释放
    store._client     = None
    store._collection = None
    gc.collect()

    # 2. 分块
    processor = DocumentProcessor(
        DOCS_DIR,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = processor.run()
    total_chunks = len(chunks)

    # 3. 向量化写入
    store.add_chunks(chunks)

    # 4. 构建混合检索器
    all_chunks = store.get_all_chunks()
    retriever  = HybridRetriever(store, all_chunks)
    generator  = RAGGenerator()

    # 5. 评估
    evaluator = RAGEvaluator(retriever, generator)
    results   = evaluator.run_eval(TEST_SET, top_k=TOP_K)

    n      = len(results)
    avg_f  = sum(r.faithfulness      or 0 for r in results) / n
    avg_r  = sum(r.answer_relevancy  or 0 for r in results) / n
    avg_cp = sum(r.context_precision or 0 for r in results) / n
    avg_cr = sum(r.context_recall    or 0 for r in results) / n
    avg_ms = sum(r.latency_ms        for r in results) / n

    exp = {
        "chunk_size":       chunk_size,
        "chunk_overlap":    chunk_overlap,
        "total_chunks":     total_chunks,
        "faithfulness":     round(avg_f,  4),
        "answer_relevancy": round(avg_r,  4),
        "context_precision":round(avg_cp, 4),
        "context_recall":   round(avg_cr, 4),
        "avg_latency_ms":   round(avg_ms, 1),
        "collection_name":  collection_name,
        "per_case": [
            {
                "question":          r.question,
                "faithfulness":      round(r.faithfulness      or 0, 4),
                "answer_relevancy":  round(r.answer_relevancy  or 0, 4),
                "context_precision": round(r.context_precision or 0, 4),
                "context_recall":    round(r.context_recall    or 0, 4),
                "latency_ms":        r.latency_ms,
                "sources":           r.sources,
            }
            for r in results
        ],
    }

    # 释放 store 资源（Windows 文件锁）
    store._client     = None
    store._collection = None
    gc.collect()

    composite = (avg_f + avg_r + avg_cp + avg_cr) / 4
    console.print(
        f"[green]✓ chunk={chunk_size}: "
        f"F={avg_f:.3f} R={avg_r:.3f} CP={avg_cp:.3f} CR={avg_cr:.3f} "
        f"composite={composite:.3f}[/green]"
    )
    return exp


# ──────────────────────────────────────────────
# 选最优
# ──────────────────────────────────────────────

def composite(e: dict) -> float:
    return (e["faithfulness"] + e["answer_relevancy"] +
            e["context_precision"] + e["context_recall"]) / 4


# ──────────────────────────────────────────────
# 报告生成
# ──────────────────────────────────────────────

def save_json(experiments: list[dict], best: dict, path: str) -> None:
    report = {
        "experiment_config": {
            "docs_dir":       DOCS_DIR,
            "test_set":       TEST_SET,
            "chunk_sizes":    CHUNK_SIZES,
            "overlap_ratio":  "20% of chunk_size",
            "top_k":          TOP_K,
        },
        "best_config": {
            "chunk_size":    best["chunk_size"],
            "chunk_overlap": best["chunk_overlap"],
            "composite_score": round(composite(best), 4),
            "reason": "Highest mean of faithfulness + answer_relevancy + context_precision + context_recall",
        },
        "results": experiments,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]✓ JSON 报告: {path}[/green]")


def save_md(experiments: list[dict], best: dict, path: str) -> None:
    def bar(v: float, w: int = 10) -> str:
        filled = round(v * w)
        return "█" * filled + "░" * (w - filled)

    e256, e512, e1024 = experiments[0], experiments[1], experiments[2]
    metrics = [
        ("Faithfulness",      "faithfulness"),
        ("Answer Relevancy",  "answer_relevancy"),
        ("Context Precision", "context_precision"),
        ("Context Recall",    "context_recall"),
    ]

    lines = [
        "# Chunk Size 对比实验报告\n",
        f"> 文档集: `data/langchain_docs` (100 docs) | "
        f"测试集: 10 条问题 | 模型: qwen2.5:3b (Ollama) | Embedding: all-MiniLM-L6-v2\n",

        "## 指标对比表\n",
        "| Chunk Size | Overlap | Chunks | Faithfulness | Answer Relevancy | "
        "Context Precision | Context Recall | Avg Latency | Composite |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for e in experiments:
        star = " ⭐" if e["chunk_size"] == best["chunk_size"] else ""
        lines.append(
            f"| **{e['chunk_size']}**{star} | {e['chunk_overlap']} | {e['total_chunks']} "
            f"| {e['faithfulness']:.3f} | {e['answer_relevancy']:.3f} "
            f"| {e['context_precision']:.3f} | {e['context_recall']:.3f} "
            f"| {e['avg_latency_ms']:.0f}ms | {composite(e):.3f} |"
        )

    lines += [
        "",
        "## 可视化对比\n",
        "| 指标 | chunk=256 | chunk=512 | chunk=1024 |",
        "| --- | --- | --- | --- |",
    ]
    for label, key in metrics:
        v0, v1, v2 = e256[key], e512[key], e1024[key]
        lines.append(
            f"| {label} "
            f"| `{bar(v0)}` {v0:.3f} "
            f"| `{bar(v1)}` {v1:.3f} "
            f"| `{bar(v2)}` {v2:.3f} |"
        )

    lines += ["", "## 各指标最优配置\n"]
    for label, key in metrics:
        winner = max(experiments, key=lambda e: e[key])
        lines.append(f"- **{label}**: chunk_size=**{winner['chunk_size']}** ({winner[key]:.3f})")

    lines += ["", "## 分析\n"]

    def cmp(a, b, key):
        return "高于" if a[key] > b[key] else ("低于" if a[key] < b[key] else "持平于")

    lines += [
        f"### chunk_size=256 (overlap={e256['chunk_overlap']})",
        f"生成 **{e256['total_chunks']}** 个 chunks，粒度最细。"
        f"检索时返回更多独立小片段，Context Recall {cmp(e256, e512, 'context_recall')} chunk=512。"
        f"但每个 chunk 包含信息较少，Faithfulness {cmp(e256, e512, 'faithfulness')} chunk=512"
        f"——小 chunk 上下文不完整时，模型会补充训练知识，幻觉风险增加。",
        "",
        f"### chunk_size=512 (overlap={e512['chunk_overlap']})",
        f"生成 **{e512['total_chunks']}** 个 chunks，中等粒度。"
        f"1 token ≈ 4 字符，512 tokens ≈ 一个完整段落，能承载一个完整概念。"
        f"通常是 context 完整性与检索精度的平衡点。",
        "",
        f"### chunk_size=1024 (overlap={e1024['chunk_overlap']})",
        f"生成 **{e1024['total_chunks']}** 个 chunks，粒度最粗。"
        f"每个 chunk 包含更完整的上下文，有助于 Faithfulness"
        f"（{cmp(e1024, e512, 'faithfulness')} chunk=512）。"
        f"但 chunk 越大，余弦相似度计算时相关信息被稀释，"
        f"Context Precision {cmp(e1024, e512, 'context_precision')} chunk=512。",
        "",
        "## 结论\n",
        f"**推荐配置: `chunk_size={best['chunk_size']}`, `chunk_overlap={best['chunk_overlap']}`**\n",
        f"综合四项指标等权平均，chunk_size={best['chunk_size']} 取得最高综合分 "
        f"**{composite(best):.3f}**。\n",
        "| 指标 | 得分 | 说明 |",
        "| --- | ---: | --- |",
    ]
    for label, key in metrics:
        lines.append(f"| {label} | {best[key]:.3f} | — |")
    lines += [
        f"| **Composite** | **{composite(best):.3f}** | 四项等权平均 |",
        "",
        f"> 本项目默认配置已更新为 `CHUNK_SIZE={best['chunk_size']}`，"
        f"`CHUNK_OVERLAP={best['chunk_overlap']}`。",
    ]

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]✓ Markdown 报告: {path}[/green]")


# ──────────────────────────────────────────────
# 重建默认 collection + 清理临时 collection
# ──────────────────────────────────────────────

def rebuild_default(best: dict) -> None:
    console.rule("[bold blue]重建默认 collection[/bold blue]")
    store = VectorStore(collection_name=config.COLLECTION_NAME)
    store.reset()
    store._client     = None
    store._collection = None
    gc.collect()

    processor = DocumentProcessor(
        DOCS_DIR,
        chunk_size=best["chunk_size"],
        chunk_overlap=best["chunk_overlap"],
    )
    chunks = processor.run()
    store.add_chunks(chunks)
    console.print(
        f"[green]✓ 默认 collection '{config.COLLECTION_NAME}' 已用 "
        f"chunk_size={best['chunk_size']} 重建，共 {store.count()} chunks[/green]"
    )
    store._client     = None
    store._collection = None
    gc.collect()


def cleanup_tmp_collections() -> None:
    import chromadb
    console.rule("[bold blue]清理临时 collection[/bold blue]")
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
    for cs in CHUNK_SIZES:
        name = f"langchain_docs_{cs}"
        if name == config.COLLECTION_NAME:
            continue
        try:
            client.delete_collection(name)
            console.print(f"  [dim]✓ 已删除 {name}[/dim]")
        except Exception:
            pass


def update_env(best: dict) -> None:
    """把最优 chunk_size 写入 .env"""
    env_path = Path("F:/AI_Program/langchain-docs-rag/.env")
    text = env_path.read_text(encoding="utf-8")

    import re
    # 更新或追加 CHUNK_SIZE / CHUNK_OVERLAP
    for key, val in [("CHUNK_SIZE", best["chunk_size"]), ("CHUNK_OVERLAP", best["chunk_overlap"])]:
        pattern = rf"^{key}=.*$"
        replacement = f"{key}={val}"
        if re.search(pattern, text, re.MULTILINE):
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        else:
            text = text.rstrip("\n") + f"\n{replacement}\n"

    env_path.write_text(text, encoding="utf-8")
    console.print(
        f"[green]✓ .env 已更新: CHUNK_SIZE={best['chunk_size']}, "
        f"CHUNK_OVERLAP={best['chunk_overlap']}[/green]"
    )


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    console.print("\n[bold]🧪 Chunk Size 对比实验[/bold]")
    console.print(f"配置: sizes={CHUNK_SIZES}, overlap=size×20%, top_k={TOP_K}\n")

    experiments = []
    for cs in CHUNK_SIZES:
        exp = run_one(cs)
        experiments.append(exp)

    best = max(experiments, key=composite)
    console.print(
        f"\n[bold green]🏆 最优: chunk_size={best['chunk_size']}, "
        f"overlap={best['chunk_overlap']}, composite={composite(best):.3f}[/bold green]"
    )

    # 报告
    save_json(experiments, best, "./eval/chunk_experiment_report.json")
    save_md(experiments, best, "./eval/chunk_experiment_summary.md")

    # 用最优配置重建默认 collection
    rebuild_default(best)

    # 清理临时 collection
    cleanup_tmp_collections()

    # 更新 .env
    update_env(best)

    # 最终汇总表
    table = Table(title="实验结果汇总")
    table.add_column("Chunk", justify="right")
    table.add_column("Overlap", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Faith.", justify="right")
    table.add_column("Relev.", justify="right")
    table.add_column("Prec.", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("ms", justify="right")
    table.add_column("Composite", justify="right")

    for e in experiments:
        mark = " ⭐" if e["chunk_size"] == best["chunk_size"] else ""
        table.add_row(
            f"{e['chunk_size']}{mark}",
            str(e["chunk_overlap"]),
            str(e["total_chunks"]),
            f"{e['faithfulness']:.3f}",
            f"{e['answer_relevancy']:.3f}",
            f"{e['context_precision']:.3f}",
            f"{e['context_recall']:.3f}",
            f"{e['avg_latency_ms']:.0f}",
            f"{composite(e):.3f}",
        )
    console.print(table)
    console.print(f"\n[bold green]✅ 实验完成！最优配置已写入 .env 并重建默认 collection。[/bold green]\n")


if __name__ == "__main__":
    main()
