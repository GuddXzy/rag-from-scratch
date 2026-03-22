"""
启动 API 服务

用法:
    python -m scripts.serve
    python -m scripts.serve --port 8080
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config


def main():
    parser = argparse.ArgumentParser(description="启动 RAG API 服务")
    parser.add_argument("--host", default=config.API_HOST)
    parser.add_argument("--port", type=int, default=config.API_PORT)
    parser.add_argument("--reload", action="store_true", help="开发模式 (自动重载)")
    args = parser.parse_args()

    import uvicorn
    print(f"\n🚀 RAG API 启动中: http://{args.host}:{args.port}")
    print(f"   文档: http://localhost:{args.port}/docs\n")

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
