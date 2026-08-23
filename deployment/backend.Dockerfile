# VectorMind Backend — Docker Image
#
# Multi-stage build: installs dependencies, copies source, runs uvicorn.
# Model checkpoint, FAISS indices, and images are mounted as volumes
# (not baked in — checkpoint is ~292MB, images are 4GB+).
#
# Build:  docker build -f deployment/backend.Dockerfile -t vectormind-backend .
# Run:    docker run -p 8000:8000 \
#           -v $(pwd)/checkpoints:/app/checkpoints \
#           -v $(pwd)/backend/indices:/app/backend/indices \
#           -v $(pwd)/data/raw/flickr30k/images:/app/data/raw/flickr30k/images \
#           -v $(pwd)/frontend/dist:/app/frontend/dist \
#           vectormind-backend

FROM python:3.11-slim AS base

WORKDIR /app

# System deps for FAISS (CPU) and Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir faiss-cpu uvicorn[standard]

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY backend/ backend/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run — volumes must be mounted at runtime:
#   /app/checkpoints          — model checkpoint
#   /app/backend/indices      — FAISS index files
#   /app/data/raw/flickr30k/images — Flickr30k images
#   /app/frontend/dist        — built React frontend
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
