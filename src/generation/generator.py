"""
RAG 生成模块 - 将检索结果 + 用户问题 → 高质量回答

面试要点: Prompt Engineering 的设计
- 为什么要在 prompt 中强调"如果上下文中没有相关信息，请说不知道"？
  → 防止幻觉 (hallucination)，这是 RAG 系统的核心价值
- 为什么要让模型标注引用来源？
  → 可追溯性，用户可以验证答案的准确性
"""

from rich.console import Console
from .. import config

console = Console()

# ============================================
# 系统提示词 - 这是 RAG 的灵魂
# ============================================
SYSTEM_PROMPT = """你是一个 LangChain 技术文档专家助手。你的任务是基于提供的文档内容，准确回答用户关于 LangChain 的技术问题。

## 规则
1. **只基于提供的上下文回答**。不要编造信息。
2. 如果上下文中没有足够信息来回答问题，明确说："根据现有文档，我无法找到关于这个问题的确切答案。"
3. 回答中**必须标注引用来源**，格式为 [来源: 文件名]。
4. 使用清晰的技术语言，适当提供代码示例。
5. 如果问题涉及多个概念，分点回答。

## 回答格式
- 先给出直接回答
- 再展开解释细节
- 最后列出参考来源"""

CONTEXT_TEMPLATE = """
## 检索到的相关文档

{context}

---

## 用户问题
{question}
"""


class RAGGenerator:
    """
    RAG 生成器: 将检索上下文 + 问题 → 回答

    使用 OpenAI 兼容接口调用 Ollama 本地模型。

    用法:
        gen = RAGGenerator()
        answer = gen.generate(question="什么是LCEL?", context_chunks=[...])
    """

    def __init__(self, model: str | None = None):
        self.model = model or config.LLM_MODEL
        self.base_url = config.OLLAMA_BASE_URL
        self._client = None

    @property
    def client(self):
        """懒加载 OpenAI 兼容客户端 (指向 Ollama)"""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key="ollama",
            )
        return self._client

    def generate(self, question: str, context_chunks: list[dict]) -> dict:
        """
        生成回答

        参数:
            question: 用户问题
            context_chunks: 检索到的文档块 [{"content": ..., "metadata": ..., "score": ...}]

        返回:
            {"answer": "...", "sources": [...], "model": "...", "tokens_used": ...}
        """
        # 1. 拼接上下文
        context = self._format_context(context_chunks)

        # 2. 构建完整 prompt
        user_message = CONTEXT_TEMPLATE.format(context=context, question=question)

        # 3. 调用 LLM
        console.print(f"[blue]🤖 正在调用 {self.model}...[/blue]")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )

        answer_text = response.choices[0].message.content

        # 4. 提取引用来源
        sources = list(set(
            c["metadata"].get("source", "unknown")
            for c in context_chunks
        ))

        # 5. token 用量
        usage = response.usage
        return {
            "answer": answer_text,
            "sources": sources,
            "model": self.model,
            "tokens_used": {
                "input": usage.prompt_tokens if usage else 0,
                "output": usage.completion_tokens if usage else 0,
            },
        }

    def generate_stream(self, question: str, context_chunks: list[dict]):
        """
        流式生成回答 (用于 API streaming 响应)

        用法:
            for chunk in gen.generate_stream(question, contexts):
                print(chunk, end="")
        """
        context = self._format_context(context_chunks)
        user_message = CONTEXT_TEMPLATE.format(context=context, question=question)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        """将检索到的 chunks 格式化为 prompt 中的上下文"""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", "unknown")
            score = chunk.get("score", 0)
            parts.append(
                f"### 文档片段 {i} [来源: {source}] (相关度: {score})\n"
                f"{chunk['content']}\n"
            )
        return "\n".join(parts)
