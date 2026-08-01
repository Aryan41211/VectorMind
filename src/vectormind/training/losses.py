"""Symmetric InfoNCE contrastive loss for CLIP-style training.

Purpose: compute the symmetric contrastive loss between image and text
embeddings in the shared embedding space, as specified in
ARCHITECTURE.md §5.

Design decisions (locked in ARCHITECTURE.md §5):
- Symmetric InfoNCE: cross-entropy in both directions (image→text and
  text→image), averaged.
- Learnable temperature parameter (initialized as log(1/0.07) per
  CLIP convention) — the model calibrates similarity sharpness
  during training.
- Accepts optional extra negatives from a MoCo-style memory queue
  (ARCHITECTURE.md §6) concatenated onto the text-side for the
  image→text direction only.

Input:
  - image_embeds: [B, D], L2-normalized
  - text_embeds: [B, D], L2-normalized
  - temperature: scalar tensor (exp(log_temperature))
  - queue_embeddings: optional [K, D], L2-normalized extra negatives

Output:
  - Scalar loss (average of image→text and text→image cross-entropy)
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def symmetric_infonce(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: torch.Tensor,
    queue_embeddings: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute symmetric InfoNCE loss.

    Given L2-normalized image and text embeddings, computes the
    contrastive loss as the average of:
      - cross-entropy for image→text retrieval
      - cross-entropy for text→image retrieval

    When ``queue_embeddings`` is provided, extra negatives from the
    memory queue are concatenated onto the text-side for the
    image→text direction, increasing negative diversity without
    increasing batch size (ARCHITECTURE.md §6).

    Args:
        image_embeds: L2-normalized image embeddings, shape ``[B, D]``.
        text_embeds: L2-normalized text embeddings, shape ``[B, D]``.
        temperature: Learnable temperature scalar tensor. Higher values
            produce softer similarity distributions; lower values
            produce sharper ones.
        queue_embeddings: Optional L2-normalized extra negatives from
            the memory queue, shape ``[K, D]``. When provided, these
            are added as extra columns in the image→text similarity
            matrix (not extra rows).

    Returns:
        Scalar loss tensor (平均 of both directions).

    Raises:
        ValueError: If batch dimensions don't match.
        ValueError: If embedding dimensions don't match.
        ValueError: If temperature is not a scalar.

    Assumptions:
        Inputs are already L2-normalized (the model's projection
        heads handle this). Temperature is positive.

    Limitations:
        Queue negatives only apply to the image→text direction.
        The text→image direction uses only in-batch negatives.
    """
    _validate_inputs(image_embeds, text_embeds, temperature, queue_embeddings)

    B = image_embeds.shape[0]

    # Compute similarity matrix: [B, B]
    # Scale by temperature (higher temp = softer distribution)
    logits = image_embeds @ text_embeds.T * temperature

    # Labels: diagonal elements are positive pairs
    labels = torch.arange(B, device=logits.device)

    # --- Image → Text direction ---
    if queue_embeddings is not None:
        # Concatenate queue negatives onto text-side: [B, B+K]
        queue_logits = image_embeds @ queue_embeddings.T * temperature
        logits_i2t = torch.cat([logits, queue_logits], dim=1)
    else:
        logits_i2t = logits

    loss_i2t = F.cross_entropy(logits_i2t, labels)

    # --- Text → Image direction (in-batch only) ---
    logits_t2i = text_embeds @ image_embeds.T * temperature
    loss_t2i = F.cross_entropy(logits_t2i, labels)

    # Symmetric average
    loss = (loss_i2t + loss_t2i) / 2.0

    return loss


def _validate_inputs(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: torch.Tensor,
    queue_embeddings: torch.Tensor | None,
) -> None:
    """Validate input tensors for symmetric InfoNCE.

    Args:
        image_embeds: Image embeddings.
        text_embeds: Text embeddings.
        temperature: Temperature scalar.
        queue_embeddings: Optional queue negatives.

    Raises:
        ValueError: If any validation check fails.
    """
    if image_embeds.ndim != 2 or text_embeds.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D, got image_embeds.ndim={image_embeds.ndim}, "
            f"text_embeds.ndim={text_embeds.ndim}."
        )

    if image_embeds.shape != text_embeds.shape:
        raise ValueError(
            f"Image and text embeddings must have the same shape, got "
            f"image_embeds={image_embeds.shape}, text_embeds={text_embeds.shape}."
        )

    if temperature.ndim != 0:
        raise ValueError(
            f"Temperature must be a scalar, got temperature.ndim={temperature.ndim}."
        )

    if queue_embeddings is not None:
        if queue_embeddings.ndim != 2:
            raise ValueError(
                f"Queue embeddings must be 2D, got queue_embeddings.ndim={queue_embeddings.ndim}."
            )
        if queue_embeddings.shape[1] != image_embeds.shape[1]:
            raise ValueError(
                f"Queue embedding dimension must match image/text dimension, got "
                f"queue_embeddings.shape[1]={queue_embeddings.shape[1]}, "
                f"image_embeds.shape[1]={image_embeds.shape[1]}."
            )
