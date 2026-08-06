"""
VectorMind FastAPI Application — Serving Layer

Main FastAPI application with startup model loading, health checks,
and router registration. Loads model and FAISS index once at startup,
not per-request.

Usage:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
    python -m backend.app
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.index_builder import load_model
from backend.schemas import HealthResponse, ServerConfig

logger = logging.getLogger(__name__)


@dataclass
class AppState:
    """Application state for holding loaded model and index."""
    model: Any = None
    device: torch.device | None = None
    image_index: faiss.Index | None = None
    text_index: faiss.Index | None = None
    index_metadata: dict[str, Any] | None = None
    loaded: bool = False


# Global application state
app_state = AppState()


def create_app(
    model_config: dict[str, Any] | None = None,
    server_config: ServerConfig | None = None,
    test_mode: bool = False,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        model_config: Model configuration dictionary. If None, loads from config.
        server_config: Server configuration. If None, uses defaults.
        test_mode: If True, skip model loading (for testing).

    Returns:
        Configured FastAPI application.
    """
    if server_config is None:
        server_config = ServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan: load model and index at startup."""
        if not test_mode:
            _load_model_and_index(model_config)
        yield
        # Cleanup on shutdown
        app_state.model = None
        app_state.image_index = None
        app_state.text_index = None
        app_state.loaded = False
        logger.info("Application shutdown, resources released")

    app = FastAPI(
        title="VectorMind",
        description="Multimodal semantic search API for images and text",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        """Add processing time header to responses."""
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        return response

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            model_loaded=app_state.model is not None,
            index_loaded=app_state.image_index is not None,
            device=str(app_state.device) if app_state.device else "unknown",
            num_indexed_images=(
                app_state.image_index.ntotal
                if app_state.image_index
                else 0
            ),
        )

    # Root endpoint
    @app.get("/", tags=["root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "VectorMind",
            "version": "0.1.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
        }

    # Include routers (will be added in commits 4 and 5)
    # from backend.routers import text_search, image_search
    # app.include_router(text_search.router)
    # app.include_router(image_search.router)

    return app


def _load_model_and_index(model_config: dict[str, Any] | None = None) -> None:
    """
    Load the trained model and FAISS index into application state.

    Args:
        model_config: Model configuration dictionary.
    """
    import json

    # Load configuration
    if model_config is None:
        from src.vectormind.utils.config import load_config
        model_config = load_config("configs/model.yaml")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    # Load model checkpoint
    checkpoint_path = Path("checkpoints/train/best_model.pt")
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        logger.warning("Running in degraded mode — model not loaded")
        return

    try:
        model = load_model(checkpoint_path, model_config, device)
        app_state.model = model
        app_state.device = device
        logger.info(f"Model loaded from {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Load FAISS indices
    index_dir = Path("backend/indices")
    if not index_dir.exists():
        logger.warning(f"Index directory not found: {index_dir}")
        logger.warning("Running without index — search endpoints unavailable")
        return

    try:
        image_index_path = index_dir / "image_index.faiss"
        text_index_path = index_dir / "text_index.faiss"

        if image_index_path.exists():
            app_state.image_index = faiss.read_index(str(image_index_path))
            logger.info(f"Loaded image index: {app_state.image_index.ntotal} vectors")

        if text_index_path.exists():
            app_state.text_index = faiss.read_index(str(text_index_path))
            logger.info(f"Loaded text index: {app_state.text_index.ntotal} vectors")

        # Load metadata
        metadata_path = index_dir / "index_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                app_state.index_metadata = json.load(f)

        app_state.loaded = True
        logger.info("FAISS indices loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load FAISS indices: {e}")


# Create the application instance
app = create_app()


def main():
    """CLI entry point for running the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Run VectorMind server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    uvicorn.run(
        "backend.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
