FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖 (利用 Docker 缓存层)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

# 复制源码
COPY . .

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

# 启动 API
CMD ["python", "-m", "scripts.serve"]
