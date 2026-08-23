"""VectorMind text search router, serving POST /search/text.

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
from functools import lru_cache
from typing import Any

import faiss
import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from backend.app import app_state
from backend.schemas import (
    ErrorResponse,
    SearchResponse,
    SearchResult,
    TextSearchRequest,
)

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
    """Search images by text query.

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
        # Tokenize the query exactly as training did.
        #
        # This previously used padding=True (pad to the longest item in
        # the call, i.e. the query's own length) while training used
        # padding="max_length" with a fixed 77 tokens. That fed the text
        # encoder a sequence-length distribution it never saw during
        # training. Both settings now come from configs/serving.yaml,
        # which is pinned to configs/data.yaml.
        tok_cfg = get_tokenizer_config()
        tokenizer = _get_tokenizer()
        encoded = tokenizer(
            request.query,
            return_tensors="pt",
            padding=tok_cfg["padding_strategy"],
            truncation=True,
            max_length=tok_cfg["max_length"],
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
        for rank, (idx, score) in enumerate(zip(indices[0], distances[0], strict=True), start=1):
            if idx == -1:  # FAISS returns -1 for failed searches
                continue

            # This searched the IMAGE index, so idx is an image-index
            # position and must be resolved against image_samples.
            # Resolving it against a caption-indexed list is what
            # returned mismatched filenames before the indices were
            # split (docs/KNOWN_ISSUES.md §2).
            filename = None
            image_url = None
            caption = None
            samples = app_state.image_samples
            if samples and 0 <= idx < len(samples):
                entry = samples[idx]
                filename = entry.get("filename")
                # Show one caption as a label for the retrieved image.
                entry_captions = entry.get("captions") or []
                caption = entry_captions[0] if entry_captions else None
                if filename:
                    image_url = f"/images/{filename}"

            result = SearchResult(
                rank=rank,
                index=int(idx),
                score=float(score),
                filename=filename,
                image_url=image_url,
                caption=caption,
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
            detail=f"Search failed: {e!s}",
        ) from e


@lru_cache(maxsize=1)
def get_tokenizer_config() -> dict[str, Any]:
    """Return the tokenizer section of configs/serving.yaml.

    Returns:
        Mapping with ``name``, ``max_length`` and ``padding_strategy``.

    Raises:
        FileNotFoundError: If configs/serving.yaml is missing.
    """
    from backend.app import load_serving_config

    return dict(load_serving_config()["tokenizer"])


@lru_cache(maxsize=1)
def _get_tokenizer() -> Any:
    """Load the query tokenizer once and cache it.

    Uses AutoTokenizer with the name from config rather than a
    hardcoded BertTokenizer, so the serving tokenizer cannot silently
    diverge from the one configs/data.yaml trained with.

    Returns:
        A HuggingFace tokenizer instance.

    Raises:
        OSError: If the tokenizer is neither cached locally nor
            downloadable. Pre-warm the cache in the Docker image rather
            than relying on network access at first request.
    """
    from transformers import AutoTokenizer

    name = get_tokenizer_config()["name"]
    logger.info("Loading query tokenizer: %s", name)
    return AutoTokenizer.from_pretrained(name)
