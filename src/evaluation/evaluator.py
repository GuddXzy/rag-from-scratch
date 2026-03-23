"""
RAG 评估模块

面试最加分的部分！大多数人做RAG项目都没有评估，你有就赢了。

评估指标 (RAGAS):
- Faithfulness: 回答是否忠于检索到的上下文？(防幻觉)
- Answer Relevancy: 回答是否切题？
- Context Precision: 检索到的内容中，排在前面的是否更相关？
- Context Recall: 是否检索到了回答问题所需的所有信息？

面试要点: 
"我做了chunk_size从256到1024的对比实验，发现512在Faithfulness和Context Precision
之间取得了最佳平衡。256的Context Recall更高但Faithfulness下降，因为chunk太小
导致上下文不完整，模型倾向于自己补充信息。"
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class EvalCase:
    """一条评估用例"""
    question: str
    ground_truth: str           # 标准答案
    relevant_docs: list[str]    # 应该检索到的文档来源


@dataclass
class EvalResult:
    """一条评估结果"""
    question: str
    generated_answer: str
    ground_truth: str
    retrieved_contexts: list[str]
    sources: list[str]
    # 指标
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    latency_ms: float = 0.0


class RAGEvaluator:
    """
    RAG 系统评估器
    
    两种模式:
    1. 手动评估: 用人工标注的测试集，跑检索+生成，计算指标
    2. 简单评估: 不用RAGAS，用简单规则打分（不需要额外API调用）
    
    用法:
        evaluator = RAGEvaluator(retriever, generator)
        results = evaluator.run_eval("./eval/test_set.json")
        evaluator.print_report(results)
    """

    def __init__(self, retriever, generator):
        """
        参数:
            retriever: HybridRetriever 实例
            generator: RAGGenerator 实例
        """
        self.retriever = retriever
        self.generator = generator

    def load_test_set(self, path: str) -> list[EvalCase]:
        """加载测试集 (JSON格式)"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cases = []
        for item in data:
            cases.append(EvalCase(
                question=item["question"],
                ground_truth=item["ground_truth"],
                relevant_docs=item.get("relevant_docs", []),
            ))
        console.print(f"[green]✓ 加载了 {len(cases)} 条评估用例[/green]")
        return cases

    def run_eval(self, test_set_path: str, top_k: int = 5) -> list[EvalResult]:
        """对测试集执行完整的 检索→生成→评估 流程"""
        cases = self.load_test_set(test_set_path)
        results: list[EvalResult] = []

        for i, case in enumerate(cases, 1):
            console.print(f"\n[blue]📝 评估 {i}/{len(cases)}: {case.question[:50]}...[/blue]")

            start = time.time()

            # 1. 检索
            retrieved = self.retriever.search(case.question, k=top_k)
            contexts = [r["content"] for r in retrieved]
            sources = [r.get("metadata", {}).get("source", "") for r in retrieved]

            # 2. 生成
            gen_result = self.generator.generate(
                question=case.question,
                context_chunks=retrieved,
            )

            latency = (time.time() - start) * 1000

            # 3. 简单评估打分
            result = EvalResult(
                question=case.question,
                generated_answer=gen_result["answer"],
                ground_truth=case.ground_truth,
                retrieved_contexts=contexts,
                sources=sources,
                latency_ms=round(latency, 2),
            )

            # 简单评估指标 (不依赖RAGAS，用规则计算)
            result.faithfulness = self._score_faithfulness(result)
            result.answer_relevancy = self._score_relevancy(result)
            result.context_precision = self._score_context_precision(result, case)
            result.context_recall = self._score_context_recall(result, case)

            results.append(result)
            console.print(f"  ✓ F={result.faithfulness:.2f} R={result.answer_relevancy:.2f} "
                         f"CP={result.context_precision:.2f} CR={result.context_recall:.2f} "
                         f"({result.latency_ms:.0f}ms)")

        return results

    def print_report(self, results: list[EvalResult]) -> None:
        """打印评估报告"""
        if not results:
            console.print("[yellow]没有评估结果[/yellow]")
            return

        # 汇总统计
        n = len(results)
        avg_f = sum(r.faithfulness or 0 for r in results) / n
        avg_r = sum(r.answer_relevancy or 0 for r in results) / n
        avg_cp = sum(r.context_precision or 0 for r in results) / n
        avg_cr = sum(r.context_recall or 0 for r in results) / n
        avg_latency = sum(r.latency_ms for r in results) / n

        console.print("\n[bold green]═══════════════════════════════════════[/bold green]")
        console.print("[bold green]         RAG 评估报告                  [/bold green]")
        console.print("[bold green]═══════════════════════════════════════[/bold green]")

        # 汇总表
        summary = Table(title="汇总指标")
        summary.add_column("指标", style="cyan")
        summary.add_column("平均分", style="green", justify="right")
        summary.add_column("说明")

        summary.add_row("Faithfulness", f"{avg_f:.3f}", "回答是否忠于上下文")
        summary.add_row("Answer Relevancy", f"{avg_r:.3f}", "回答是否切题")
        summary.add_row("Context Precision", f"{avg_cp:.3f}", "检索排序是否准确")
        summary.add_row("Context Recall", f"{avg_cr:.3f}", "是否检索到了所需信息")
        summary.add_row("Avg Latency", f"{avg_latency:.0f}ms", "平均响应时间")

        console.print(summary)

        # 逐条明细
        detail = Table(title="逐条结果")
        detail.add_column("#", style="dim", width=3)
        detail.add_column("问题", max_width=40)
        detail.add_column("F", justify="right", width=5)
        detail.add_column("R", justify="right", width=5)
        detail.add_column("CP", justify="right", width=5)
        detail.add_column("CR", justify="right", width=5)
        detail.add_column("ms", justify="right", width=6)

        for i, r in enumerate(results, 1):
            detail.add_row(
                str(i),
                r.question[:38] + ("…" if len(r.question) > 38 else ""),
                f"{r.faithfulness:.2f}" if r.faithfulness else "-",
                f"{r.answer_relevancy:.2f}" if r.answer_relevancy else "-",
                f"{r.context_precision:.2f}" if r.context_precision else "-",
                f"{r.context_recall:.2f}" if r.context_recall else "-",
                f"{r.latency_ms:.0f}",
            )

        console.print(detail)

    def save_report(self, results: list[EvalResult], path: str) -> None:
        """保存完整评估结果到 JSON 文件（含检索上下文和汇总）"""
        n = len(results)
        avg = lambda key: round(sum(getattr(r, key) or 0 for r in results) / n, 4)

        data = {
            "summary": {
                "total_cases": n,
                "faithfulness":       avg("faithfulness"),
                "answer_relevancy":   avg("answer_relevancy"),
                "context_precision":  avg("context_precision"),
                "context_recall":     avg("context_recall"),
                "avg_latency_ms":     round(sum(r.latency_ms for r in results) / n, 1),
            },
            "cases": [],
        }

        for i, r in enumerate(results, 1):
            data["cases"].append({
                "id": i,
                "question": r.question,
                "ground_truth": r.ground_truth,
                "generated_answer": r.generated_answer,
                "retrieved_sources": r.sources,
                "retrieved_contexts": r.retrieved_contexts,
                "metrics": {
                    "faithfulness":      round(r.faithfulness or 0, 4),
                    "answer_relevancy":  round(r.answer_relevancy or 0, 4),
                    "context_precision": round(r.context_precision or 0, 4),
                    "context_recall":    round(r.context_recall or 0, 4),
                },
                "latency_ms": r.latency_ms,
            })

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[green]✓ JSON 报告已保存到 {path}[/green]")

    def save_summary(self, results: list[EvalResult], path: str) -> None:
        """生成可读性 Markdown 摘要（含汇总表 + 逐条明细）"""
        n = len(results)
        avg_f  = sum(r.faithfulness      or 0 for r in results) / n
        avg_r  = sum(r.answer_relevancy  or 0 for r in results) / n
        avg_cp = sum(r.context_precision or 0 for r in results) / n
        avg_cr = sum(r.context_recall    or 0 for r in results) / n
        avg_ms = sum(r.latency_ms for r in results) / n

        def bar(score: float, width: int = 10) -> str:
            filled = round(score * width)
            return "█" * filled + "░" * (width - filled)

        lines = [
            "# RAG 评估报告\n",
            f"> 测试用例: {n} 条 | 模型: qwen2.5:7b (Ollama) | Embedding: all-MiniLM-L6-v2\n",
            "## 汇总指标\n",
            "| 指标 | 得分 | 可视化 | 说明 |",
            "| --- | ---: | --- | --- |",
            f"| Faithfulness      | {avg_f:.3f}  | `{bar(avg_f)}`  | 回答是否忠于检索上下文（防幻觉） |",
            f"| Answer Relevancy  | {avg_r:.3f}  | `{bar(avg_r)}`  | 回答是否切题 |",
            f"| Context Precision | {avg_cp:.3f} | `{bar(avg_cp)}` | 相关文档是否排在检索结果前列 |",
            f"| Context Recall    | {avg_cr:.3f} | `{bar(avg_cr)}` | ground truth 信息是否被检索覆盖 |",
            f"| Avg Latency       | {avg_ms:.0f} ms | — | 平均端到端响应时间 |",
            "",
            "## 逐条明细\n",
            "| # | 问题 | Faith. | Relev. | C.Prec | C.Rec | 来源文件 | ms |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
        ]

        for i, r in enumerate(results, 1):
            q = r.question[:55] + ("…" if len(r.question) > 55 else "")
            srcs = ", ".join(set(s.replace("_", "\\_") for s in r.sources if s))[:60]
            lines.append(
                f"| {i} | {q} "
                f"| {r.faithfulness:.2f} "
                f"| {r.answer_relevancy:.2f} "
                f"| {r.context_precision:.2f} "
                f"| {r.context_recall:.2f} "
                f"| {srcs} "
                f"| {r.latency_ms:.0f} |"
            )

        lines += [
            "",
            "## 逐条问答详情\n",
        ]
        for i, r in enumerate(results, 1):
            lines += [
                f"### Case {i}: {r.question}\n",
                f"**Ground Truth:** {r.ground_truth}\n",
                f"**Generated Answer:**\n\n{r.generated_answer}\n",
                f"**Retrieved Sources:** {', '.join(r.sources)}\n",
                f"**Scores:** Faithfulness={r.faithfulness:.2f} | "
                f"Relevancy={r.answer_relevancy:.2f} | "
                f"Precision={r.context_precision:.2f} | "
                f"Recall={r.context_recall:.2f} | "
                f"Latency={r.latency_ms:.0f}ms\n",
                "---\n",
            ]

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]✓ Markdown 摘要已保存到 {path}[/green]")

    # ============================================
    # 简单评估函数 (不依赖 RAGAS API)
    # ============================================

    @staticmethod
    def _score_faithfulness(result: EvalResult) -> float:
        """
        Faithfulness: 回答中的关键信息是否出现在检索上下文中？
        简单方法: 计算回答中的关键词在上下文中的命中率
        """
        answer_tokens = set(result.generated_answer.lower().split())
        context_text = " ".join(result.retrieved_contexts).lower()
        context_tokens = set(context_text.split())

        # 过滤掉停用词（太常见的词）
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                     "to", "for", "of", "and", "or", "but", "not", "you", "can", "this",
                     "that", "it", "with", "from", "by", "as", "be", "have", "has",
                     "的", "是", "在", "了", "不", "和", "也", "就", "都", "而"}
        answer_keywords = answer_tokens - stopwords

        if not answer_keywords:
            return 1.0

        hits = sum(1 for w in answer_keywords if w in context_tokens)
        return min(hits / len(answer_keywords), 1.0)

    @staticmethod
    def _score_relevancy(result: EvalResult) -> float:
        """
        Answer Relevancy: 回答是否与问题相关？
        简单方法: 问题关键词在回答中的出现率
        """
        stopwords = {"what", "how", "why", "when", "where", "which", "who",
                     "is", "are", "the", "a", "an", "do", "does", "can",
                     "什么", "如何", "怎么", "为什么", "吗", "呢", "的"}
        q_tokens = set(result.question.lower().split()) - stopwords
        answer_lower = result.generated_answer.lower()

        if not q_tokens:
            return 1.0

        hits = sum(1 for w in q_tokens if w in answer_lower)
        return min(hits / len(q_tokens), 1.0)

    @staticmethod
    def _score_context_precision(result: EvalResult, case: EvalCase) -> float:
        """
        Context Precision: 检索结果中，相关文档是否排在前面？
        简单方法: 检查 ground_truth 中的关键词在 top-k 上下文中的分布
        """
        if not result.retrieved_contexts:
            return 0.0

        gt_tokens = set(case.ground_truth.lower().split())
        scores = []
        for ctx in result.retrieved_contexts:
            ctx_tokens = set(ctx.lower().split())
            overlap = len(gt_tokens & ctx_tokens)
            scores.append(overlap / max(len(gt_tokens), 1))

        # 加权: 排在前面的权重更高
        weighted_sum = 0.0
        total_weight = 0.0
        for i, s in enumerate(scores):
            weight = 1.0 / (i + 1)  # 1, 0.5, 0.33, ...
            weighted_sum += s * weight
            total_weight += weight

        return min(weighted_sum / max(total_weight, 1e-6), 1.0)

    @staticmethod
    def _score_context_recall(result: EvalResult, case: EvalCase) -> float:
        """
        Context Recall: ground_truth 中的信息是否被检索到？
        简单方法: ground_truth 关键词在全部检索上下文中的覆盖率
        """
        stopwords = {"the", "a", "an", "is", "are", "in", "on", "to", "for", "of",
                     "and", "or", "it", "this", "that", "的", "是", "了", "在"}
        gt_tokens = set(case.ground_truth.lower().split()) - stopwords
        all_context = " ".join(result.retrieved_contexts).lower()

        if not gt_tokens:
            return 1.0

        hits = sum(1 for w in gt_tokens if w in all_context)
        return min(hits / len(gt_tokens), 1.0)
