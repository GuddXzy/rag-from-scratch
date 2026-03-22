"""
向量存储模块 - 把文本变成向量，存进 ChromaDB

面试要点:
- 为什么选 all-MiniLM-L6-v2? 
  → 轻量(80MB)、速度快、效果不差，适合原型和中小规模
  → 生产环境可以换 bge-large 或 OpenAI text-embedding-3-small
- 为什么选 ChromaDB?
  → 本地运行、零配置、Python原生、对个人项目足够
  → 企业级可以换 Pinecone(托管) 或 Qdrant(自部署)
"""

from pathlib import Path
from rich.console import Console
from rich.progress import track

from ..document_pipeline.chunker import Chunk
from .. import config

console = Console()


class VectorStore:
    """
    向量化 + 持久化存储
    
    用法:
        store = VectorStore()
        store.add_chunks(chunks)           # 写入
        results = store.search("问题", k=5) # 查询
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
    ):
        self.persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.embedding_model_name = embedding_model or config.EMBEDDING_MODEL

        self._embedding_model = None
        self._client = None
        self._collection = None

    @property
    def embedding_model(self):
        """懒加载 embedding 模型 (首次使用时才加载，节省启动时间)"""
        if self._embedding_model is None:
            console.print(f"[blue]⏳ 加载 Embedding 模型: {self.embedding_model_name}...[/blue]")
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            console.print("[green]✓ Embedding 模型加载完成[/green]")
        return self._embedding_model

    @property
    def collection(self):
        """懒加载 ChromaDB collection"""
        if self._collection is None:
            import chromadb
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
            )
        return self._collection

    def add_chunks(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        """
        将 chunks 向量化并存入 ChromaDB
        分批处理避免内存溢出
        """
        if not chunks:
            console.print("[yellow]⚠ 没有 chunks 需要存储[/yellow]")
            return

        console.print(f"\n[bold blue]🔢 向量化 {len(chunks)} 个 chunks...[/bold blue]")

        # 分批处理
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [c.metadata for c in batch]

            # 生成 embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False).tolist()

            # 存入 ChromaDB
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            console.print(f"  ✓ 已存储 {min(i + batch_size, len(chunks))}/{len(chunks)}")

        console.print(f"[green]✓ 全部 {len(chunks)} 个 chunks 已存入 ChromaDB[/green]")

    def search(self, query: str, k: int | None = None) -> list[dict]:
        """
        语义检索: 找到与 query 最相关的 k 个 chunks
        
        返回格式:
        [
            {"content": "...", "metadata": {...}, "score": 0.85},
            ...
        ]
        """
        k = k or config.TOP_K
        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # 整理返回格式
        items = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            items.append({
                "content": doc,
                "metadata": meta,
                "score": round(1 - dist, 4),  # cosine distance → similarity
            })

        return items

    def count(self) -> int:
        """返回 collection 中的文档数量"""
        return self.collection.count()

    def get_all_chunks(self) -> list:
        """返回 collection 中所有 chunks，用于 BM25 建索引"""
        from ..document_pipeline.chunker import Chunk
        result = self.collection.get(include=["documents", "metadatas"])
        chunks = []
        for doc, meta, chunk_id in zip(
            result["documents"], result["metadatas"], result["ids"]
        ):
            chunks.append(Chunk(content=doc, metadata=meta, chunk_id=chunk_id))
        return chunks

    def reset(self) -> None:
        """清空 collection (用于重新导入)"""
        import chromadb
        client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
            console.print("[yellow]⚠ Collection 已清空[/yellow]")
        except Exception:
            pass
