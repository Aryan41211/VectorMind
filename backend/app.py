"""VectorMind FastAPI application: startup loading, health, routing.

Main FastAPI application with startup model loading, health checks,
and router registration. Loads model and FAISS index once at startup,
not per-request.

Usage:
    uvicorn backend.app:app --host 0.0.0.0 --port 8000
    python -m backend.app
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.index_builder import load_model
from backend.middleware import (
    MaxBodySizeMiddleware,
    RateLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from backend.schemas import HealthResponse, ServerConfig
from vectormind.utils.config import load_config, require_keys

logger = logging.getLogger(__name__)

# Serving configuration lives in config, not in this module (CLAUDE.md §6).
SERVING_CONFIG_PATH = Path("configs/serving.yaml")
MODEL_CONFIG_PATH = Path("configs/model.yaml")


def load_serving_config(
    path: Path | str = SERVING_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the serving-layer configuration.

    Args:
        path: Path to the serving YAML. Defaults to
            ``configs/serving.yaml``.

    Returns:
        Parsed configuration with ``server``, ``cors``, ``paths``,
        ``search`` and ``tokenizer`` sections.

    Raises:
        FileNotFoundError: If the config file does not exist.
        KeyError: If a required top-level section is missing.
    """
    config: dict[str, Any] = load_config(str(path))
    require_keys(
        config, ["server", "cors", "paths", "search", "tokenizer", "limits"]
    )
    return config


@dataclass
class AppState:
    """Application state for holding loaded model and index."""
    model: Any = None
    device: torch.device | None = None
    image_index: faiss.Index | None = None
    text_index: faiss.Index | None = None
    index_metadata: dict[str, Any] | None = None
    # One record per image-index position: filename, path, all captions.
    image_samples: list[dict[str, Any]] | None = None
    # One record per text-index position: caption, filename, path.
    caption_samples: list[dict[str, Any]] | None = None
    loaded: bool = False


# Global application state
app_state = AppState()


def create_app(
    model_config: dict[str, Any] | None = None,
    server_config: ServerConfig | None = None,
    test_mode: bool = False,
    serving_config: dict[str, Any] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        model_config: Model configuration dictionary. If None, loads from config.
        server_config: Server configuration. If None, uses defaults.
        test_mode: If True, skip model loading (for testing).
        serving_config: Parsed configs/serving.yaml. If None, loaded from
            disk. Injectable so tests can point at temporary directories
            instead of the real checkpoint and index.

    Returns:
        Configured FastAPI application.

    Raises:
        FileNotFoundError: If serving_config is None and
            configs/serving.yaml is missing.
    """
    if server_config is None:
        server_config = ServerConfig()
    if serving_config is None:
        serving_config = load_serving_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load model and index at startup, release them at shutdown."""
        if not test_mode:
            _load_model_and_index(model_config, serving_config)
        yield
        # Cleanup on shutdown
        app_state.model = None
        app_state.image_index = None
        app_state.text_index = None
        app_state.image_samples = None
        app_state.caption_samples = None
        app_state.loaded = False
        logger.info("Application shutdown, resources released")

    app = FastAPI(
        title="VectorMind",
        description="Multimodal semantic search API for images and text",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware. Origins, methods and headers come from
    # configs/serving.yaml. The previous wildcard-plus-credentials pair
    # was rejected by browsers outright — the wildcard origin is not
    # permitted when allow_credentials is true.
    cors_cfg = serving_config["cors"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_cfg["allow_origins"],
        allow_credentials=cors_cfg["allow_credentials"],
        allow_methods=cors_cfg["allow_methods"],
        allow_headers=cors_cfg["allow_headers"],
    )

    # Middleware runs outermost-last, so these are registered in reverse
    # order of execution: request context wraps everything (it must see
    # rejections too), then security headers, the size guard, and finally
    # the rate limiter closest to the handler.
    limits = serving_config["limits"]
    app.add_middleware(
        RateLimitMiddleware,
        max_requests=int(limits["rate_limit_requests"]),
        window_seconds=float(limits["rate_limit_window_seconds"]),
    )
    app.add_middleware(
        MaxBodySizeMiddleware, max_bytes=int(limits["max_upload_bytes"])
    )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestContextMiddleware)

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

    # Readiness, distinct from liveness.
    #
    # /health answers "is this process alive" and returns 200 as soon as
    # the app is up. /ready answers "can this process serve traffic", and
    # returns 503 until the model and index are both loaded. An
    # orchestrator that routes on /health sends searches to a container
    # that will 503 every one of them for the ~30s the checkpoint takes
    # to load; that is what this separation prevents.
    @app.get("/ready", tags=["health"])
    async def readiness_check() -> JSONResponse:
        """Readiness probe: 200 only when searches can actually be served."""
        ready = (
            app_state.loaded
            and app_state.model is not None
            and app_state.image_index is not None
            and app_state.text_index is not None
        )
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "ready": ready,
                "model_loaded": app_state.model is not None,
                "image_index_loaded": app_state.image_index is not None,
                "text_index_loaded": app_state.text_index is not None,
                "index_maps_loaded": app_state.image_samples is not None
                and app_state.caption_samples is not None,
            },
        )

    # API info endpoint (moved from / to avoid conflict with SPA)
    @app.get("/api/info", tags=["root"])
    async def api_info() -> dict[str, str]:
        """API information endpoint."""
        return {
            "name": "VectorMind",
            "version": "0.1.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
        }

    # Include routers
    from backend.routers import image_search, text_search
    app.include_router(text_search.router)
    app.include_router(image_search.router)

    # Serve Flickr30k images as static files
    images_dir = Path(serving_config["paths"]["images_dir"])
    if images_dir.exists():
        app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")
        logger.info(f"Mounted static images from {images_dir}")
    else:
        logger.warning(f"Images directory not found: {images_dir}")

    # Serve frontend SPA (production static build)
    frontend_dist = Path(serving_config["paths"]["frontend_dist"])
    if frontend_dist.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dist)), name="frontend_static")
        logger.info(f"Mounted frontend static assets from {frontend_dist}")

        @app.get("/", include_in_schema=False)
        async def serve_spa() -> FileResponse:
            """Serve the SPA index.html at root."""
            return FileResponse(str(frontend_dist / "index.html"))

        @app.get("/{full_path:path}", include_in_schema=False)
        async def serve_spa_fallback(full_path: str) -> FileResponse:
            """Catch-all: serve index.html for SPA client-side routing."""
            file_path = frontend_dist / full_path
            if file_path.is_file():
                return FileResponse(str(file_path))
            return FileResponse(str(frontend_dist / "index.html"))
    else:
        logger.info("frontend/dist not found — SPA not served (use Vite dev server)")

    return app


def _load_model_and_index(
    model_config: dict[str, Any] | None = None,
    serving_config: dict[str, Any] | None = None,
) -> None:
    """Load the trained model and FAISS index into application state.

    Missing artifacts are logged and skipped rather than raised: the app
    still starts, /health reports what is loaded, and the search
    endpoints return 503. That keeps a container without its volumes
    mounted diagnosable instead of crash-looping.

    Args:
        model_config: Model configuration dictionary. Loaded from
            configs/model.yaml when None.
        serving_config: Parsed configs/serving.yaml. Loaded from disk
            when None.
    """
    import json

    if model_config is None:
        model_config = load_config(str(MODEL_CONFIG_PATH))
    if serving_config is None:
        serving_config = load_serving_config()

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")

    # Load model checkpoint
    paths = serving_config["paths"]
    checkpoint_path = Path(paths["checkpoint"])
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
    index_dir = Path(paths["index_dir"])
    if not index_dir.exists():
        logger.warning(f"Index directory not found: {index_dir}")
        logger.warning("Running without index — search endpoints unavailable")
        return

    try:
        image_index_path = index_dir / paths["image_index"]
        text_index_path = index_dir / paths["text_index"]

        if image_index_path.exists():
            app_state.image_index = faiss.read_index(str(image_index_path))
            logger.info(f"Loaded image index: {app_state.image_index.ntotal} vectors")

        if text_index_path.exists():
            app_state.text_index = faiss.read_index(str(text_index_path))
            logger.info(f"Loaded text index: {app_state.text_index.ntotal} vectors")

        # Load metadata
        metadata_path = index_dir / paths["index_metadata"]
        if metadata_path.exists():
            with open(metadata_path) as f:
                app_state.index_metadata = json.load(f)

        # Load the two index maps. Each must line up with its own index;
        # a mismatch means the indices and maps were built from
        # different runs, and every result would carry wrong metadata.
        for key, attr, index in (
            ("image_samples", "image_samples", app_state.image_index),
            ("caption_samples", "caption_samples", app_state.text_index),
        ):
            samples_path = index_dir / paths[key]
            if not samples_path.exists():
                logger.warning("Index map not found: %s", samples_path)
                continue
            with open(samples_path, encoding="utf-8") as f:
                records = json.load(f)
            if index is not None and len(records) != index.ntotal:
                logger.error(
                    "%s has %d records but its index holds %d vectors — "
                    "rebuild with 'python -m backend.index_builder'.",
                    samples_path,
                    len(records),
                    index.ntotal,
                )
                continue
            setattr(app_state, attr, records)
            logger.info("Loaded %s: %d records", key, len(records))

        app_state.loaded = True
        logger.info("FAISS indices loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load FAISS indices: {e}")


# Create the application instance
app = create_app()


def main() -> None:
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
