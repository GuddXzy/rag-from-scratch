# Chunk Size 对比实验报告

> 文档集: `data/langchain_docs` (100 docs) | 测试集: 10 条问题 | 模型: qwen2.5:3b (Ollama) | Embedding: all-MiniLM-L6-v2

## 指标对比表

| Chunk Size | Overlap | Chunks | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Avg Latency | Composite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **256** | 51 | 2189 | 0.294 | 0.610 | 0.227 | 0.497 | 3846ms | 0.407 |
| **512** ⭐ | 102 | 1596 | 0.305 | 0.645 | 0.237 | 0.507 | 3998ms | 0.424 |
| **1024** | 205 | 1440 | 0.306 | 0.628 | 0.234 | 0.500 | 3931ms | 0.417 |

## 可视化对比

| 指标 | chunk=256 | chunk=512 | chunk=1024 |
| --- | --- | --- | --- |
| Faithfulness | `███░░░░░░░` 0.294 | `███░░░░░░░` 0.305 | `███░░░░░░░` 0.306 |
| Answer Relevancy | `██████░░░░` 0.610 | `██████░░░░` 0.645 | `██████░░░░` 0.628 |
| Context Precision | `██░░░░░░░░` 0.227 | `██░░░░░░░░` 0.237 | `██░░░░░░░░` 0.234 |
| Context Recall | `█████░░░░░` 0.497 | `█████░░░░░` 0.507 | `█████░░░░░` 0.500 |

## 各指标最优配置

- **Faithfulness**: chunk_size=**1024** (0.306)
- **Answer Relevancy**: chunk_size=**512** (0.645)
- **Context Precision**: chunk_size=**512** (0.237)
- **Context Recall**: chunk_size=**512** (0.507)

## 分析

### chunk_size=256 (overlap=51)
生成 **2189** 个 chunks，粒度最细。检索时返回更多独立小片段，Context Recall 低于 chunk=512。但每个 chunk 包含信息较少，Faithfulness 低于 chunk=512——小 chunk 上下文不完整时，模型会补充训练知识，幻觉风险增加。

### chunk_size=512 (overlap=102)
生成 **1596** 个 chunks，中等粒度。1 token ≈ 4 字符，512 tokens ≈ 一个完整段落，能承载一个完整概念。通常是 context 完整性与检索精度的平衡点。

### chunk_size=1024 (overlap=205)
生成 **1440** 个 chunks，粒度最粗。每个 chunk 包含更完整的上下文，有助于 Faithfulness（高于 chunk=512）。但 chunk 越大，余弦相似度计算时相关信息被稀释，Context Precision 低于 chunk=512。

## 结论

**推荐配置: `chunk_size=512`, `chunk_overlap=102`**

综合四项指标等权平均，chunk_size=512 取得最高综合分 **0.424**。

| 指标 | 得分 | 说明 |
| --- | ---: | --- |
| Faithfulness | 0.305 | — |
| Answer Relevancy | 0.645 | — |
| Context Precision | 0.237 | — |
| Context Recall | 0.507 | — |
| **Composite** | **0.424** | 四项等权平均 |

> 本项目默认配置已更新为 `CHUNK_SIZE=512`，`CHUNK_OVERLAP=102`。