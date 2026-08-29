# VectorMind Backend — Docker Image
#
# Runs the FastAPI app (model + FAISS index) under uvicorn.
#
# Model checkpoint, FAISS indices, and the Flickr30k images are mounted
# as volumes rather than baked in: the checkpoint is ~278MB and the
# image set is 1.3GB.
#
# Build:  docker build -f deployment/backend.Dockerfile -t vectormind-backend .
# Run:    see deployment/docker-compose.yml, or the CMD comment below.

FROM python:3.12-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep the tokenizer inside the image so the first query does not
    # depend on network access (see the pre-warm step below).
    HF_HOME=/opt/hf

# System deps for FAISS (CPU) and Pillow.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first so this layer caches across source edits.
#
# CPU torch is the default: the CUDA wheels weigh ~2.5GB, which a CPU-only
# serving container cannot use. To build a GPU-serving image instead, pass
# the CUDA wheel index, e.g.:
#
#   docker build -f deployment/backend.Dockerfile \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
#     -t vectormind-backend:gpu .
#
# (deployment/docker-compose.gpu.yml wires this up for the RTX 4050.)
#
# requirements-serving.txt rather than requirements.txt — the full set
# pulls in tensorboard, matplotlib, pytest, mypy and ruff, none of
# which are reachable from a running server.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY requirements-serving.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url $TORCH_INDEX_URL && \
    pip install -r requirements-serving.txt

# Install the package itself. Without this, `vectormind` is not on
# sys.path and backend/ fails at import: modules under src/ import
# `vectormind.*`, which only resolves once the package is installed.
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-deps -e .

COPY configs/ configs/
COPY backend/ backend/

# Pre-download the tokenizer named in configs/serving.yaml. The text
# search endpoint loads it on first request; without this the container
# needs outbound network access at query time, and fails without it.
RUN python -c "\
import yaml; \
from transformers import AutoTokenizer; \
name = yaml.safe_load(open('configs/serving.yaml'))['tokenizer']['name']; \
AutoTokenizer.from_pretrained(name); \
print(f'Cached tokenizer: {name}')"

# Everything the server loads is now in the image, so stop transformers
# from checking the Hub on first use. Set AFTER the download above, which
# needs the network.
#
# Without this the cache is still used, but only after an outbound
# request that succeeds slowly, fails slowly, or logs an unauthenticated-
# request warning on every cold start. Offline makes first-query latency
# a property of the image rather than of the host's egress.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Fail the build rather than shipping an image that cannot import.
RUN python -c "import backend.app; print('backend.app imports OK')"

# And fail it if the tokenizer cannot load from the baked-in cache, which
# is the assertion the two variables above turn into a real one: with the
# Hub unreachable, this line is the whole text-search path's first step.
RUN python -c "\
import yaml; \
from transformers import AutoTokenizer; \
name = yaml.safe_load(open('configs/serving.yaml'))['tokenizer']['name']; \
AutoTokenizer.from_pretrained(name); \
print(f'Tokenizer loads offline: {name}')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# Volumes expected at runtime (paths from configs/serving.yaml):
#   /app/checkpoints                 — model checkpoint
#   /app/backend/indices             — FAISS indices and index maps
#   /app/data/raw/flickr30k/images   — Flickr30k images
#   /app/frontend/dist               — built React frontend (optional)
#
# Missing volumes are logged and skipped: the app still starts and
# /health reports what loaded, so a misconfigured mount is diagnosable
# instead of a crash loop.
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
