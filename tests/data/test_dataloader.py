"""Unit tests for vectormind.data.dataloader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from vectormind.data.dataloader import _collate_fn, create_dataloaders
from vectormind.data.tokenizer import CaptionTokenizer


@pytest.fixture
def data_config() -> dict[str, Any]:
    """Minimal config for dataloader tests."""
    return {
        "dataset": {
            "image_size": 224,
            "tokenizer_name": "bert-base-uncased",
            "max_text_length": 77,
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": True,
            "random_seed": 42,
        },
        "transforms": {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "resize_size": 256,
            "random_horizontal_flip_p": 0.5,
        },
    }


@pytest.fixture
def tokenizer() -> CaptionTokenizer:
    return CaptionTokenizer(tokenizer_name="bert-base-uncased", max_length=77)


@pytest.fixture
def sample_pairs(tmp_path: Path) -> list[tuple[Path, str]]:
    """Create 16 small test images with captions."""

    pairs = []
    for i in range(16):
        img = Image.new("RGB", (64, 64), color=(i * 15, 100, 200 - i * 10))
        path = tmp_path / f"img_{i:04d}.jpg"
        img.save(path)
        pairs.append((path, f"Caption for image {i}"))
    return pairs


@pytest.fixture
def train_transform():
    """A simple eval-like transform for testing."""
    import torch
    from torchvision.transforms.v2 import (
        Compose,
        Resize,
        CenterCrop,
        ToImage,
        ToDtype,
        Normalize,
    )

    return Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@pytest.fixture
def eval_transform():
    import torch
    from torchvision.transforms.v2 import (
        Compose,
        Resize,
        CenterCrop,
        ToImage,
        ToDtype,
        Normalize,
    )

    return Compose(
        [
            Resize(256),
            CenterCrop(224),
            ToImage(),
            ToDtype(dtype=torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def test_collate_fn_output_keys() -> None:
    """_collate_fn should return a dict with expected keys."""
    batch = [
        {
            "image": torch.randn(3, 224, 224),
            "input_ids": torch.ones(77, dtype=torch.long),
            "attention_mask": torch.ones(77, dtype=torch.long),
            "caption_text": "hello",
        },
        {
            "image": torch.randn(3, 224, 224),
            "input_ids": torch.ones(77, dtype=torch.long),
            "attention_mask": torch.ones(77, dtype=torch.long),
            "caption_text": "world",
        },
    ]
    result = _collate_fn(batch)

    assert "image" in result
    assert "input_ids" in result
    assert "attention_mask" in result
    assert "caption_text" in result


def test_collate_fn_tensor_shapes() -> None:
    """_collate_fn should produce correctly shaped tensors."""
    batch = [
        {
            "image": torch.randn(3, 224, 224),
            "input_ids": torch.ones(77, dtype=torch.long),
            "attention_mask": torch.ones(77, dtype=torch.long),
            "caption_text": "a",
        },
        {
            "image": torch.randn(3, 224, 224),
            "input_ids": torch.ones(77, dtype=torch.long),
            "attention_mask": torch.ones(77, dtype=torch.long),
            "caption_text": "b",
        },
    ]
    result = _collate_fn(batch)

    assert result["image"].shape == (2, 3, 224, 224)
    assert result["input_ids"].shape == (2, 77)
    assert result["attention_mask"].shape == (2, 77)
    assert isinstance(result["caption_text"], list)
    assert len(result["caption_text"]) == 2


def test_create_dataloaders_returns_three_loaders(
    data_config: dict[str, Any],
    sample_pairs: list[tuple[Path, str]],
    train_transform,
    eval_transform,
    tokenizer: CaptionTokenizer,
) -> None:
    """create_dataloaders should return (train, val, test) loaders."""
    # Split the 16 pairs into 3 groups.
    train_pairs = sample_pairs[:8]
    val_pairs = sample_pairs[8:12]
    test_pairs = sample_pairs[12:16]

    train_loader, val_loader, test_loader = create_dataloaders(
        config=data_config,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_transform=train_transform,
        eval_transform=eval_transform,
        tokenizer=tokenizer,
    )

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None


def test_dataloader_batch_shapes(
    data_config: dict[str, Any],
    sample_pairs: list[tuple[Path, str]],
    train_transform,
    eval_transform,
    tokenizer: CaptionTokenizer,
) -> None:
    """Batches from the dataloader should have correct shapes."""
    train_pairs = sample_pairs[:8]
    val_pairs = sample_pairs[8:12]
    test_pairs = sample_pairs[12:16]

    train_loader, _, _ = create_dataloaders(
        config=data_config,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_transform=train_transform,
        eval_transform=eval_transform,
        tokenizer=tokenizer,
    )

    batch = next(iter(train_loader))

    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["input_ids"].shape == (4, 77)
    assert batch["attention_mask"].shape == (4, 77)
    assert len(batch["caption_text"]) == 4


def test_dataloader_no_nan(
    data_config: dict[str, Any],
    sample_pairs: list[tuple[Path, str]],
    train_transform,
    eval_transform,
    tokenizer: CaptionTokenizer,
) -> None:
    """Batches should contain no NaN or Inf values."""
    train_pairs = sample_pairs[:8]
    val_pairs = sample_pairs[8:12]
    test_pairs = sample_pairs[12:16]

    train_loader, _, _ = create_dataloaders(
        config=data_config,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        train_transform=train_transform,
        eval_transform=eval_transform,
        tokenizer=tokenizer,
    )

    batch = next(iter(train_loader))
    assert not batch["image"].isnan().any()
    assert not batch["image"].isinf().any()


def test_dataloader_raises_on_empty_train() -> None:
    """create_dataloaders should raise ValueError if train is empty."""
    config = {
        "dataset": {
            "image_size": 224,
            "max_text_length": 77,
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "drop_last": True,
        },
    }
    with pytest.raises(ValueError, match="non-empty"):
        create_dataloaders(
            config=config,
            train_pairs=[],
            val_pairs=[(Path("a.jpg"), "cap")],
            test_pairs=[(Path("b.jpg"), "cap")],
            train_transform=None,
            eval_transform=None,
            tokenizer=None,
        )
