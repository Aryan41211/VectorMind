"""Training loop infrastructure: AMP, gradient accumulation, metrics.

Purpose: provide the core training machinery — a ``train_one_step()``
function that handles mixed precision, gradient accumulation,
contrastive loss computation, and metric collection. This is
infrastructure only; the multi-epoch loop that drives it lives
in ``vectormind.training.trainer`` (one shared implementation used by
scripts/train.py).

Design decisions (locked in ARCHITECTURE.md §6):
- Mixed precision (``torch.cuda.amp``) is default, not optional —
  roughly halves activation memory on 6GB VRAM.
- Gradient accumulation simulates larger effective batch size for
  optimizer statistics (but does not increase negatives per
  contrastive comparison — that's the memory queue's job).
- Memory queue enqueued after each forward pass, not per accumulation
  step, to keep the implementation simple and correct.

Input:
  - train_one_step(model, batch, optimizer, scaler, memory_queue,
    config) → dict[str, float]

Output:
  - Metrics dict: loss, temperature, embedding norms/variance,
    GPU memory usage.

This module does NOT call the real training loop against the full
dataset — that's Phase 3.5/4.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from vectormind.models.vectormind_model import VectorMindModel
from vectormind.training.losses import symmetric_infonce
from vectormind.training.memory_queue import MemoryQueue
from vectormind.training.uniformity import combined_uniformity_loss

logger = logging.getLogger(__name__)


def train_one_step(
    model: VectorMindModel,
    batch: dict[str, torch.Tensor | list[str]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    memory_queue: MemoryQueue,
    accumulation_steps: int = 1,
    device: torch.device | None = None,
    enqueue_text: bool = True,
    uniformity_weight: float = 0.0,
) -> dict[str, float]:
    """Execute one training step with AMP and gradient accumulation.

    Performs the forward pass, loss computation, and backward pass.
    The optimizer step and scaler update are handled by the caller
    at the appropriate accumulation boundary.

    Args:
        model: The VectorMindModel to train.
        batch: A batch dict from the DataLoader with keys ``"image"``,
            ``"input_ids"``, ``"attention_mask"``.
        optimizer: The optimizer instance.
        scaler: GradScaler for mixed precision training.
        memory_queue: MoCo-style memory queue for extra negatives.
        accumulation_steps: Number of steps to accumulate gradients
            before optimizer.step(). Default 1 (no accumulation).
        device: Device to move batch tensors to. If None, uses the
            model's parameter device.
        enqueue_text: If True (default), push this step's text
            embeddings into ``memory_queue`` before returning. Set
            False only when the caller manages the queue itself.
        uniformity_weight: Weight on the Wang & Isola uniformity term,
            which spreads embeddings over the hypersphere. **0.0 by
            default, which reproduces InfoNCE-only training exactly** —
            so enabling it is a single config value and an A/B against
            the current model is a clean comparison.

    Returns:
        Dictionary of metrics:
            ``"loss"``: total loss value (divided by accumulation_steps)
            ``"loss_i2t"``: image→text loss component
            ``"loss_t2i"``: text→image loss component
            ``"temperature"``: current temperature value
            ``"image_embed_norm"``: mean L2 norm of image embeddings
            ``"text_embed_norm"``: mean L2 norm of text embeddings
            ``"image_embed_std"``: std of image embedding norms
            ``"text_embed_std"``: std of text embedding norms
            ``"gpu_memory_gb"``: peak GPU memory allocated (GB)

    Raises:
        RuntimeError: If the model is not in training mode.
        RuntimeError: If batch tensors cannot be moved to device.

    Assumptions:
        The caller handles optimizer.zero_grad(), optimizer.step(),
        and scaler.update() at the correct accumulation boundaries,
        and calls ``model.clamp_log_temperature()`` after each
        optimizer step. This function computes the loss, scales it for
        accumulation, and maintains the memory queue.

    Limitations:
        Gradient clipping is not applied here — the caller can
        handle it if needed.
    """
    if device is None:
        device = next(model.parameters()).device

    # Move batch tensors to device
    images = batch["image"].to(device, non_blocking=True)  # type: ignore[union-attr]
    input_ids = batch["input_ids"].to(device, non_blocking=True)  # type: ignore[union-attr]
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)  # type: ignore[union-attr]

    # Forward pass with AMP
    with torch.autocast(device_type="cuda", enabled=True):
        # Encode
        image_embeds = model.encode_image(images)
        text_embeds = model.encode_text(input_ids, attention_mask)

        # Get queue negatives
        queue_embeds = memory_queue.get_embeddings()
        queue_for_loss = queue_embeds if queue_embeds.numel() > 0 else None

        # Compute loss
        contrastive = symmetric_infonce(
            image_embeds, text_embeds, model.temperature, queue_for_loss
        )

        # Optional uniformity regularizer (docs/KNOWN_ISSUES.md §12).
        # Skipped entirely at weight 0 rather than multiplied by zero, so
        # the default path costs nothing — the term is O(B^2).
        if uniformity_weight > 0.0:
            uniformity = combined_uniformity_loss(image_embeds, text_embeds)
            loss = contrastive + uniformity_weight * uniformity
        else:
            uniformity = torch.zeros((), device=contrastive.device)
            loss = contrastive

        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps

    # Backward pass (scaled by GradScaler)
    scaler.scale(scaled_loss).backward()

    with torch.no_grad():
        # Enqueue the text embeddings this step already produced.
        #
        # This used to be the caller's job, which meant every training
        # step ran the text encoder a second time purely to fill the
        # queue — roughly a 20% throughput cost for an identical result.
        # Enqueueing here also matches MoCo semantics: the queue holds
        # the embeddings the loss actually saw, from before this step's
        # optimizer update.
        if enqueue_text:
            memory_queue.enqueue(text_embeds)

        metrics = _collect_metrics(
            loss=loss,
            model=model,
            image_embeds=image_embeds,
            text_embeds=text_embeds,
        )
        # Reported separately so the two terms can be watched against
        # each other — a uniformity weight that is winning shows up as
        # contrastive loss rising while total loss falls.
        metrics["loss_contrastive"] = contrastive.item()
        metrics["loss_uniformity"] = uniformity.item()

    return metrics


def _collect_metrics(
    loss: torch.Tensor,
    model: VectorMindModel,
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> dict[str, float]:
    """Collect training metrics for logging.

    Args:
        loss: The computed loss tensor.
        model: The model (for temperature).
        image_embeds: Image embeddings [B, D].
        text_embeds: Text embeddings [B, D].

    Returns:
        Dictionary of metric name to float value.
    """
    # Temperature (exp of log_temperature)
    temp_val = model.temperature.item()

    # Embedding norms (should be ~1.0 given L2 normalization)
    img_norms = image_embeds.norm(p=2, dim=-1)
    txt_norms = text_embeds.norm(p=2, dim=-1)

    img_norm_mean = img_norms.mean().item()
    txt_norm_mean = txt_norms.mean().item()
    img_norm_std = img_norms.std().item()
    txt_norm_std = txt_norms.std().item()

    # GPU memory usage
    gpu_memory_gb: float = 0.0
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

    metrics = {
        "loss": loss.item(),
        "temperature": temp_val,
        "image_embed_norm": img_norm_mean,
        "text_embed_norm": txt_norm_mean,
        "image_embed_std": img_norm_std,
        "text_embed_std": txt_norm_std,
        "gpu_memory_gb": gpu_memory_gb,
    }

    return metrics


def create_optimizer(
    model: VectorMindModel,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Create an AdamW optimizer for the model.

    Uses different weight decay for different parameter groups:
    - No weight decay on biases and LayerNorm parameters
    - No weight decay on the learnable logit scale
    - Standard weight decay on everything else

    ``log_temperature`` is excluded because weight decay on it is not a
    regularizer, it is a fixed pull toward ``log_temperature = 0``, i.e.
    a logit scale of 1.0. That is an arbitrary target unrelated to
    generalization, and it fights the gradient signal rather than
    constraining it. CLIP excludes the scalar for the same reason; the
    real ceiling is the clamp in
    :meth:`VectorMindModel.clamp_log_temperature`.

    Args:
        model: The model to optimize.
        lr: Learning rate.
        weight_decay: Weight decay coefficient.

    Returns:
        Configured AdamW optimizer.
    """
    # Separate parameters into decay/no-decay groups
    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            "bias" in name
            or "norm" in name
            or "ln" in name
            or "log_temperature" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    logger.info(
        "Optimizer created: AdamW, lr=%.2e, weight_decay=%.4f, "
        "decay_params=%d, no_decay_params=%d",
        lr,
        weight_decay,
        len(decay_params),
        len(no_decay_params),
    )

    return optimizer


def create_scaler() -> torch.amp.GradScaler:
    """Create a GradScaler for mixed precision training.

    Returns:
        A GradScaler instance.
    """
    return torch.amp.GradScaler("cuda")
