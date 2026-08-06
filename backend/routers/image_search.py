"""
VectorMind Image Search Router — /search/image endpoint

Handles image queries and returns captions ranked by similarity.
Uses the image encoder to generate query embeddings and FAISS
text index for efficient similarity search.

NOTE: This endpoint uses synchronous def (not async def) because
it performs blocking operations (image transforms, model inference,
FAISS search). FastAPI runs synchronous endpoints in a threadpool
to avoid blocking the event loop.
"""

from __future__ import annotations

import io
import logging
import time

import faiss
import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from backend.app import app_state
from backend.schemas import ErrorResponse, SearchResponse, SearchResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])

# Image preprocessing transforms
_transforms = None


def _get_transforms():
    """Get image preprocessing transforms."""
    global _transforms
    if _transforms is None:
        from torchvision import transforms
        _transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return _transforms


@router.post(
    "/image",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Search captions by image query",
    description="Find captions matching an uploaded image using semantic similarity.",
)
def search_by_image(
    file: UploadFile = File(..., description="Image file to search with"),
    top_k: int = 10,
) -> SearchResponse:
    """
    Search captions by image query.

    Encodes the uploaded image using the image encoder, then searches the
    FAISS text index for the most similar captions.

    Args:
        file: Uploaded image file.
        top_k: Number of results to return.

    Returns:
        SearchResponse with ranked caption results.

    Raises:
        HTTPException: If image is invalid or service is unavailable.
    """
    # Check if service is available
    if not app_state.loaded or app_state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: model or index not loaded",
        )

    if app_state.text_index is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: text index not loaded",
        )

    start_time = time.time()

    try:
        # Validate and load image (synchronous file read)
        image = _validate_image_sync(file)

        # Preprocess image
        transforms = _get_transforms()
        image_tensor = transforms(image).unsqueeze(0)  # [1, 3, 224, 224]

        # Move to device
        device = app_state.device
        image_tensor = image_tensor.to(device)

        # Generate image embedding
        with torch.no_grad():
            image_embedding = app_state.model.encode_image(image_tensor)
            image_embedding = image_embedding.cpu().numpy().astype(np.float32)

        # Normalize for inner product search
        faiss.normalize_L2(image_embedding)

        # Search FAISS index
        k = min(top_k, app_state.text_index.ntotal)
        distances, indices = app_state.text_index.search(image_embedding, k)

        # Build results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:  # FAISS returns -1 for failed searches
                continue

            result = SearchResult(
                index=int(idx),
                score=float(score),
                caption=f"caption_{idx}",  # Placeholder — real caption from metadata
            )
            results.append(result)

        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Image search: {file.filename} -> "
            f"{len(results)} results in {latency_ms:.1f}ms"
        )

        return SearchResponse(
            results=results,
            query=f"image:{file.filename}",
            search_type="image_to_text",
            total_results=len(results),
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )


def _validate_image_sync(file: UploadFile) -> Image.Image:
    """
    Validate and load an uploaded image file (synchronous).

    Args:
        file: Uploaded file.

    Returns:
        PIL Image.

    Raises:
        HTTPException: If file is not a valid image.
    """
    # Check content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image.",
        )

    # Read file content synchronously via the underlying file object
    content = file.file.read()

    # Check file size (max 10MB)
    max_size = 10 * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {len(content)} bytes. Max: {max_size} bytes.",
        )

    # Try to open as image
    try:
        image = Image.open(io.BytesIO(content))
        image.load()  # Force load to catch corrupted images
        return image.convert("RGB")
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}",
        )
