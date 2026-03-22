"""
FastAPI REST API 服务

端点:
  POST /query          - 提问并获取回答
  POST /query/stream   - 提问并获取流式回答
  POST /ingest         - 导入新文档
  GET  /health         - 健康检查
  GET  /stats          - 系统统计信息
"""

import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..embeddings.store import VectorStore
from ..retrieval.hybrid import HybridRetriever
from ..generation.generator import RAGGenerator
from ..document_pipeline.chunker import Chunk
from .. import config


# ============================================
# 请求/响应模型
# ============================================
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    top_k: int = Field(default=5, ge=1, le=20, description="检索结果数量")

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieval_results: list[dict]
    model: str
    tokens_used: dict
    latency_ms: float

class IngestRequest(BaseModel):
    docs_dir: str = Field(..., description="文档目录路径")

class StatsResponse(BaseModel):
    total_chunks: int
    collection_name: str
    embedding_model: str
    llm_model: str


# ============================================
# 全局状态 (应用启动时初始化)
# ============================================
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期: 启动时加载模型和数据"""
    print("🚀 正在初始化 RAG 系统...")

    store = VectorStore()
    generator = RAGGenerator()

    # 检查是否已有数据
    doc_count = store.count()
    if doc_count == 0:
        print("⚠️  ChromaDB 中没有数据，请先运行 ingest 导入文档")
        print("   命令: python -m scripts.ingest --docs-dir ./data/langchain_docs")
    else:
        print(f"✓ ChromaDB 已加载 {doc_count} 个 chunks")

    _state["store"] = store
    _state["generator"] = generator
    _state["chunks"] = []  # 会在 ingest 时填充

    print("✓ RAG 系统就绪！")
    yield
    print("👋 RAG 系统关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="LangChain Docs RAG API",
        description="Production-grade RAG system for LangChain documentation Q&A",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "message": "RAG system is running"}

    @app.get("/stats", response_model=StatsResponse)
    async def get_stats():
        store: VectorStore = _state["store"]
        return StatsResponse(
            total_chunks=store.count(),
            collection_name=config.COLLECTION_NAME,
            embedding_model=config.EMBEDDING_MODEL,
            llm_model=config.LLM_MODEL,
        )

    @app.post("/query", response_model=QueryResponse)
    async def query(req: QueryRequest):
        """主查询端点: 问题 → 检索 → 生成 → 回答"""
        store: VectorStore = _state["store"]
        generator: RAGGenerator = _state["generator"]

        if store.count() == 0:
            raise HTTPException(status_code=503, detail="没有文档数据，请先运行 ingest")

        start = time.time()

        # 1. 检索
        chunks = _state.get("chunks", [])
        if chunks:
            retriever = HybridRetriever(store, chunks)
            results = retriever.search(req.question, k=req.top_k)
        else:
            # fallback: 纯语义检索
            results = store.search(req.question, k=req.top_k)

        # 2. 生成
        gen_result = generator.generate(
            question=req.question,
            context_chunks=results,
        )

        latency = (time.time() - start) * 1000

        return QueryResponse(
            answer=gen_result["answer"],
            sources=gen_result["sources"],
            retrieval_results=results,
            model=gen_result["model"],
            tokens_used=gen_result["tokens_used"],
            latency_ms=round(latency, 2),
        )

    @app.post("/query/stream")
    async def query_stream(req: QueryRequest):
        """流式查询端点"""
        store: VectorStore = _state["store"]
        generator: RAGGenerator = _state["generator"]

        if store.count() == 0:
            raise HTTPException(status_code=503, detail="没有文档数据，请先运行 ingest")

        results = store.search(req.question, k=req.top_k)

        def event_stream():
            # 先发送检索来源
            sources = list(set(r.get("metadata", {}).get("source", "") for r in results))
            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
            # 再流式发送回答
            for chunk in generator.generate_stream(req.question, results):
                yield f"data: {json.dumps({'type': 'token', 'data': chunk})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/ingest")
    async def ingest(req: IngestRequest):
        """导入文档到向量数据库"""
        from ..document_pipeline.processor import DocumentProcessor

        store: VectorStore = _state["store"]

        try:
            processor = DocumentProcessor(req.docs_dir)
            chunks = processor.run()

            if not chunks:
                raise HTTPException(status_code=400, detail="没有找到可处理的文档")

            store.add_chunks(chunks)
            _state["chunks"] = chunks

            return {
                "status": "success",
                "chunks_added": len(chunks),
                "total_chunks": store.count(),
            }
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))

    return app


# 直接运行: python -m src.api.app
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
