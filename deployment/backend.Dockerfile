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
# CPU torch explicitly: the default wheels carry CUDA and weigh ~2.5GB,
# which a CPU-only serving container cannot use.
#
# requirements-serving.txt rather than requirements.txt — the full set
# pulls in tensorboard, wandb, matplotlib, pytest, mypy and ruff, none of
# which are reachable from a running server.
COPY requirements-serving.txt .
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
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

# Fail the build rather than shipping an image that cannot import.
RUN python -c "import backend.app; print('backend.app imports OK')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

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
