"""
混合检索 (Hybrid Retrieval)

这是整个 RAG 系统最核心的模块，也是区分"玩具"和"生产级"的关键。

面试必问: 为什么要混合检索？
- 纯语义检索: 擅长理解"意思相近"的查询，但可能漏掉精确关键词匹配
  例: 搜"如何配置API密钥" → 语义检索可能返回"认证设置"相关内容 ✓
- 纯BM25(关键词): 擅长精确匹配术语，但不懂同义词
  例: 搜"ChatModel" → BM25 直接匹配到含 ChatModel 的文档 ✓
- 混合检索: 两者互补，用 RRF (Reciprocal Rank Fusion) 合并排名

面试加分: 你可以展示混合 vs 纯语义 的召回率对比实验
"""

from rank_bm25 import BM25Okapi
from rich.console import Console

from ..embeddings.store import VectorStore
from ..document_pipeline.chunker import Chunk
from .. import config

console = Console()


class HybridRetriever:
    """
    混合检索器: 语义检索 + BM25 关键词检索 + RRF 融合
    
    用法:
        retriever = HybridRetriever(vector_store, chunks)
        results = retriever.search("如何使用 LCEL?")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        chunks: list[Chunk],
        bm25_weight: float | None = None,
        semantic_weight: float | None = None,
    ):
        self.vector_store = vector_store
        self.bm25_weight = bm25_weight or config.BM25_WEIGHT
        self.semantic_weight = semantic_weight or config.SEMANTIC_WEIGHT

        # 构建 BM25 索引
        self.chunks = chunks
        self.chunk_texts = [c.content for c in chunks]
        self.chunk_map = {c.chunk_id: c for c in chunks}

        # BM25 需要分词后的文本
        tokenized = [self._tokenize(text) for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

        console.print(f"[green]✓ 混合检索器已就绪 (BM25 索引: {len(chunks)} docs)[/green]")

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """
        执行混合检索
        
        返回: [{"content": ..., "metadata": ..., "score": ..., "source": "hybrid"}, ...]
        """
        k = k or config.TOP_K
        fetch_k = k * 3  # 多取一些候选，融合后再截断

        # 1. 语义检索
        semantic_results = self.vector_store.search(query, k=fetch_k)

        # 2. BM25 检索
        bm25_results = self._bm25_search(query, k=fetch_k)

        # 3. RRF 融合
        fused = self._rrf_fusion(semantic_results, bm25_results, k=k)

        return fused

    def search_semantic_only(self, query: str, k: int | None = None) -> list[dict]:
        """纯语义检索 (用于对比实验)"""
        return self.vector_store.search(query, k=k or config.TOP_K)

    def search_bm25_only(self, query: str, k: int | None = None) -> list[dict]:
        """纯BM25检索 (用于对比实验)"""
        return self._bm25_search(query, k=k or config.TOP_K)

    def _bm25_search(self, query: str, k: int = 10) -> list[dict]:
        """BM25 关键词检索"""
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # 取 top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回有分数的
                results.append({
                    "content": self.chunk_texts[idx],
                    "metadata": self.chunks[idx].metadata,
                    "score": round(float(scores[idx]), 4),
                })
        return results

    def _rrf_fusion(
        self,
        semantic_results: list[dict],
        bm25_results: list[dict],
        k: int = 5,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion (RRF) 
        
        公式: score = Σ 1 / (rrf_k + rank_i)
        rrf_k=60 是论文推荐的默认值
        
        面试要点: RRF 的优点是不需要对不同检索器的分数做归一化，
        只看排名，所以可以无缝合并不同来源的结果
        """
        # 用 content 作为去重 key
        score_map: dict[str, dict] = {}

        # 语义检索的贡献
        for rank, item in enumerate(semantic_results):
            key = item["content"][:200]  # 用前200字符做key避免完全重复
            if key not in score_map:
                score_map[key] = {**item, "rrf_score": 0.0}
            score_map[key]["rrf_score"] += self.semantic_weight / (rrf_k + rank + 1)

        # BM25 的贡献
        for rank, item in enumerate(bm25_results):
            key = item["content"][:200]
            if key not in score_map:
                score_map[key] = {**item, "rrf_score": 0.0}
            score_map[key]["rrf_score"] += self.bm25_weight / (rrf_k + rank + 1)

        # 按 RRF 分数排序，取 top-k
        sorted_results = sorted(score_map.values(), key=lambda x: x["rrf_score"], reverse=True)[:k]

        # 整理输出
        for item in sorted_results:
            item["score"] = round(item.pop("rrf_score"), 6)
            item["source"] = "hybrid"

        return sorted_results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """简单分词: 按空格和标点切分，转小写"""
        import re
        # 保留字母数字和中文，其余当分隔符
        tokens = re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text.lower())
        return tokens
