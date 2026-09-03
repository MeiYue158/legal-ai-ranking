# Minimal, single-container inference service for the legal statute ranker.
# No k8s/autoscaling/Redis -- this is a portfolio-scale deployment, not a
# distributed system, by explicit design choice.
FROM python:3.10-slim

WORKDIR /app

# system deps: none beyond build tools needed by lightgbm's wheel; keep image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch wheel index FIRST: the code always runs on CPU (see
# dense_index.py's comment on the MPS SIGSEGV), so the default GPU/CUDA
# wheel resolution pip would otherwise pick pulls in >2GB of unused
# nvidia-cuda-*/cudnn packages for a container with no GPU. Installing torch
# from the CPU index before the rest of requirements.txt keeps the image lean.
RUN pip install --no-cache-dir torch==2.14.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/statutes/articles.json ./data/statutes/articles.json
COPY data/statutes/article_embeddings_e5.npy ./data/statutes/article_embeddings_e5.npy
COPY data/statutes/article_ids_e5.json ./data/statutes/article_ids_e5.json
COPY data/processed/train_clean.parquet ./data/processed/train_clean.parquet
COPY data/processed/emb_cache/train_full_e5_embeddings.npy ./data/processed/emb_cache/train_full_e5_embeddings.npy
COPY reports/lgb_ranker_final_e5_noprior.pkl ./reports/lgb_ranker_final_e5_noprior.pkl

ENV OMP_NUM_THREADS=1
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV HF_HUB_OFFLINE=0

WORKDIR /app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
