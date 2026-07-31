"""DataLoader factory for the VectorMind data pipeline.

Purpose: create train/val/test DataLoaders from Flickr30k split data,
with appropriate transforms, batching, and worker configuration — all
driven by configs/data.yaml.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from vectormind.data.dataset import Flickr30kDataset

logger = logging.getLogger(__name__)


def _collate_fn(
    batch: list[dict[str, Tensor | str]],
) -> dict[str, Tensor | list[str]]:
    """Collate a list of dataset items into a batched dictionary.

    Pads ``input_ids`` and ``attention_mask`` tensors to the maximum
    length in the batch (or to ``max_text_length``, whichever is
    smaller), and stacks image tensors.

    Args:
        batch: A list of dicts, each from ``Flickr30kDataset.__getitem__``.

    Returns:
        A dictionary with batched tensors:
        - ``"image"``: ``[B, 3, H, W]``
        - ``"input_ids"``: ``[B, max_length]``
        - ``"attention_mask"``: ``[B, max_length]``
        - ``"caption_text"``: list of B strings
    """
    images = torch.stack([item["image"] for item in batch], dim=0)
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.stack([item["attention_mask"] for item in batch], dim=0)
    caption_texts = [item["caption_text"] for item in batch]

    return {
        "image": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "caption_text": caption_texts,
    }


def create_dataloaders(
    config: dict[str, Any],
    train_pairs: list[tuple[Path, str]],
    val_pairs: list[tuple[Path, str]],
    test_pairs: list[tuple[Path, str]],
    train_transform: Callable[[Any], Tensor],
    eval_transform: Callable[[Any], Tensor],
    tokenizer: Any,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from split data.

    Args:
        config: The full data config dict (from ``configs/data.yaml``).
        train_pairs: List of ``(image_path, caption)`` for training.
        val_pairs: List of ``(image_path, caption)`` for validation.
        test_pairs: List of ``(image_path, caption)`` for testing.
        train_transform: Transform pipeline for training images.
        eval_transform: Transform pipeline for eval images.
        tokenizer: A ``CaptionTokenizer`` instance.

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)``.

    Raises:
        ValueError: If any split is empty.

    Assumptions:
        All image paths in the pairs exist and are readable.
        The config contains all required keys.

    Limitations:
        ``drop_last=True`` for training is intentional — avoids the
        noisy gradient from a tiny last batch. Validation/test loaders
        use ``drop_last=False`` so all samples are evaluated.
    """
    batch_size = config["dataset"]["batch_size"]
    num_workers = config["dataset"]["num_workers"]
    pin_memory = config["dataset"]["pin_memory"]
    drop_last = config["dataset"]["drop_last"]
    max_text_length = config["dataset"]["max_text_length"]

    if not train_pairs:
        raise ValueError("train_pairs must be non-empty.")
    if not val_pairs:
        raise ValueError("val_pairs must be non-empty.")
    if not test_pairs:
        raise ValueError("test_pairs must be non-empty.")

    # Unpack pairs into separate lists.
    train_paths, train_caps = zip(*train_pairs)
    val_paths, val_caps = zip(*val_pairs)
    test_paths, test_caps = zip(*test_pairs)

    # Create datasets.
    train_dataset = Flickr30kDataset(
        image_paths=list(train_paths),
        captions=list(train_caps),
        transform=train_transform,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
    )
    val_dataset = Flickr30kDataset(
        image_paths=list(val_paths),
        captions=list(val_caps),
        transform=eval_transform,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
    )
    test_dataset = Flickr30kDataset(
        image_paths=list(test_paths),
        captions=list(test_caps),
        transform=eval_transform,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
    )

    # Build DataLoaders.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_fn,
    )

    logger.info(
        "DataLoaders created — Train: %d batches, Val: %d batches, "
        "Test: %d batches (batch_size=%d, num_workers=%d)",
        len(train_loader),
        len(val_loader),
        len(test_loader),
        batch_size,
        num_workers,
    )

    return train_loader, val_loader, test_loader
