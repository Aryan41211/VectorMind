"""
VectorMind Text Search Router — /search/text endpoint

Handles text queries and returns images ranked by similarity.
Uses the text encoder to generate query embeddings and FAISS
index for efficient similarity search.

NOTE: This endpoint uses synchronous def (not async def) because
it performs blocking operations (tokenization, model inference, FAISS
search). FastAPI runs synchronous endpoints in a threadpool to avoid
blocking the event loop.
"""

from __future__ import annotations

import logging
import time

import faiss
import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from backend.app import app_state
from backend.schemas import ErrorResponse, SearchResponse, SearchResult, TextSearchRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


@router.post(
    "/text",
    response_model=SearchResponse,
    responses={
        503: {"model": ErrorResponse, "description": "Service unavailable"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Search images by text query",
    description="Find images matching a text description using semantic similarity.",
)
def search_by_text(request: TextSearchRequest) -> SearchResponse:
    """
    Search images by text query.

    Encodes the text query using the text encoder, then searches the
    FAISS image index for the most similar images.

    Args:
        request: TextSearchRequest with query and top_k.

    Returns:
        SearchResponse with ranked image results.

    Raises:
        HTTPException: If service is unavailable or search fails.
    """
    # Check if service is available
    if not app_state.loaded or app_state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: model or index not loaded",
        )

    if app_state.image_index is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: image index not loaded",
        )

    start_time = time.time()

    try:
        # Tokenize the query
        tokenizer = _get_tokenizer()
        encoded = tokenizer(
            request.query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        # Move to device — extract input_ids and attention_mask separately
        device = app_state.device
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Generate text embedding — pass as separate tensor arguments
        with torch.no_grad():
            text_embedding = app_state.model.encode_text(input_ids, attention_mask)
            text_embedding = text_embedding.cpu().numpy().astype(np.float32)

        # Normalize for inner product search
        faiss.normalize_L2(text_embedding)

        # Search FAISS index
        k = min(request.top_k, app_state.image_index.ntotal)
        distances, indices = app_state.image_index.search(text_embedding, k)

        # Build results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
            if idx == -1:  # FAISS returns -1 for failed searches
                continue

            result = SearchResult(
                index=int(idx),
                score=float(score),
                image_path=f"image_{idx}",  # Placeholder — real path from metadata
            )
            results.append(result)

        latency_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Text search: '{request.query[:50]}...' -> "
            f"{len(results)} results in {latency_ms:.1f}ms"
        )

        return SearchResponse(
            results=results,
            query=request.query,
            search_type="text_to_image",
            total_results=len(results),
            latency_ms=latency_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )


def _get_tokenizer():
    """Get the tokenizer for text encoding."""
    from transformers import BertTokenizer

    # Cache tokenizer to avoid reloading
    if not hasattr(_get_tokenizer, "_tokenizer"):
        _get_tokenizer._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return _get_tokenizer._tokenizer
