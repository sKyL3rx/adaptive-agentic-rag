FROM python:3.11.14-slim-bookworm


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8002 \
    PROMETHEUS_PORT=9102 \
    EMBEDDING_MODEL_PATH=/app/models/gte-modernbert-base \
    EMBEDDING_ONNX_FILE=onnx/model_int8.onnx \
    EMBEDDING_BATCH_SIZE=4 \
    EMBEDDING_MAX_LENGTH=512 \
    RERANK_MODEL_PATH=/app/models/gte-reranker-modernbert-base \
    RERANK_ONNX_FILE=onnx/model_quantized.onnx \
    RERANK_BATCH_SIZE=2 \
    RERANK_MAX_LENGTH=512 \
    MAX_RERANK_PAIRS=12 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=3 \
    MKL_NUM_THREADS=3


WORKDIR /app


RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN python -m pip install \
        --no-cache-dir \
        --upgrade \
        pip==25.2 \
    && python -m pip install \
        --no-cache-dir \
        -r requirements.txt


COPY models/gte-modernbert-base \
    /app/models/gte-modernbert-base

COPY models/gte-reranker-modernbert-base \
    /app/models/gte-reranker-modernbert-base


COPY servers.py .
COPY onnx_reranker.py .
COPY retrieval_contracts.py .
COPY retrieval_runtime.py .
COPY retrieval_pipeline.py .
COPY retrieval_metrics.py .


RUN useradd \
        --create-home \
        --uid 10001 \
        appuser \
    && chown -R appuser:appuser /app


USER appuser


EXPOSE 8002 9102


CMD ["python", "-u", "servers.py"]