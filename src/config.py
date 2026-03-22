"""
全局配置 - 所有可调参数集中管理
面试加分点：集中配置管理，方便做对比实验
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === 路径 ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DATA_DIR / "chroma_db"))

# === Embedding 配置 ===
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# === LLM 配置 ===
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# === 分块配置 (这些参数面试必问，要能解释为什么这么选) ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))       # 每个chunk的token数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))   # chunk之间的重叠token数

# === 检索配置 ===
TOP_K = int(os.getenv("TOP_K", "5"))                    # 检索返回的top-k个结果
BM25_WEIGHT = 0.3   # BM25在混合检索中的权重
SEMANTIC_WEIGHT = 0.7  # 语义检索在混合检索中的权重

# === API 配置 ===
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# === ChromaDB Collection 名称 ===
COLLECTION_NAME = "langchain_docs"
