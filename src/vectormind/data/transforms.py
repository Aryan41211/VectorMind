"""Image transforms for the VectorMind data pipeline.

Purpose: provide train and evaluation image transform pipelines that
resize, crop, normalize, and convert Flickr30k images into tensors
suitable for the image encoder (ARCHITECTURE.md §2). All settings
come from configs/data.yaml — no hardcoded values.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torchvision.transforms import v2

logger = logging.getLogger(__name__)


def get_train_transforms(config: dict[str, Any]) -> v2.Compose:
    """Build the training augmentations pipeline.

    Args:
        config: The full data config dict (from ``configs/data.yaml``).
            Must contain ``transforms.image_mean``, ``transforms.image_std``,
            ``transforms.resize_size``, ``transforms.random_horizontal_flip_p``,
            and ``dataset.image_size``.

    Returns:
        A ``torchvision.transforms.v2.Compose`` pipeline for training.

    Raises:
        KeyError: If any required key is missing from ``config``.

    Assumptions:
        Input images are PIL Images or tensors of arbitrary reasonable
        size; the pipeline handles resizing internally.

    Limitations:
        Augmentation is deliberately light — RandomCrop plus an optional
        horizontal flip and color jitter. Color jitter is gated behind
        ``transforms.color_jitter_strength`` (0.0 = off, the value the
        shipped checkpoint trained with) because raising it changes the
        input distribution and would only be meaningful after a retrain.
    """
    image_size = config["dataset"]["image_size"]
    resize_size = config["transforms"]["resize_size"]
    mean = config["transforms"]["image_mean"]
    std = config["transforms"]["image_std"]
    flip_p = config["transforms"]["random_horizontal_flip_p"]
    jitter = float(config["transforms"].get("color_jitter_strength", 0.0))

    pipeline: list[Any] = [
        v2.Resize(resize_size, interpolation=v2.InterpolationMode.BILINEAR),
        v2.RandomCrop(image_size),
        v2.RandomHorizontalFlip(p=flip_p),
    ]
    if jitter > 0.0:
        # torchvision clamps hue to [-0.5, 0.5] and rejects anything
        # above; map the configured strength onto it rather than
        # smuggling a second knob into the config.
        pipeline.append(
            v2.ColorJitter(
                brightness=jitter,
                contrast=jitter,
                saturation=jitter,
                hue=min(jitter, 0.5),
            )
        )
    pipeline.extend(
        [
            v2.ToImage(),  # PIL Image -> Tensor (uint8)
            v2.ToDtype(dtype=torch.float32, scale=True),  # scales [0,255] -> [0,1]
            v2.Normalize(mean=mean, std=std),
        ]
    )
    transforms = v2.Compose(pipeline)

    logger.info(
        "Train transforms: Resize(%d) -> RandomCrop(%d) -> "
        "RandomHorizontalFlip(p=%.1f)%s -> ToFloat -> Normalize",
        resize_size,
        image_size,
        flip_p,
        f" -> ColorJitter(s={jitter:g})" if jitter > 0.0 else "",
    )
    return transforms


def get_eval_transforms(config: dict[str, Any]) -> v2.Compose:
    """Build the evaluation (val/test) transforms pipeline.

    Args:
        config: The full data config dict (from ``configs/data.yaml``).
            Must contain ``transforms.image_mean``, ``transforms.image_std``,
            ``transforms.resize_size``, and ``dataset.image_size``.

    Returns:
        A ``torchvision.transforms.v2.Compose`` pipeline for evaluation.

    Raises:
        KeyError: If any required key is missing from ``config``.

    Assumptions:
        No randomness — a deterministic center crop ensures reproducible
        evaluation across runs.

    Limitations:
        Single-crop only; multi-crop evaluation is not implemented
        (not needed at this project's scale).
    """
    image_size = config["dataset"]["image_size"]
    resize_size = config["transforms"]["resize_size"]
    mean = config["transforms"]["image_mean"]
    std = config["transforms"]["image_std"]

    transforms = v2.Compose(
        [
            v2.Resize(resize_size, interpolation=v2.InterpolationMode.BILINEAR),
            v2.CenterCrop(image_size),
            v2.ToImage(),  # PIL Image -> Tensor (uint8)
            v2.ToDtype(dtype=torch.float32, scale=True),  # scales [0,255] -> [0,1]
            v2.Normalize(mean=mean, std=std),
        ]
    )

    logger.info(
        "Eval transforms: Resize(%d) -> CenterCrop(%d) -> " "ToFloat -> Normalize",
        resize_size,
        image_size,
    )
    return transforms
