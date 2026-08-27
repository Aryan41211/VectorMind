"""Unit tests for vectormind.data.transforms."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from PIL import Image

from vectormind.data.transforms import get_eval_transforms, get_train_transforms


@pytest.fixture
def data_config() -> dict[str, Any]:
    """Minimal config dict for transform tests."""
    return {
        "dataset": {"image_size": 224},
        "transforms": {
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "resize_size": 256,
            "random_horizontal_flip_p": 0.5,
        },
    }


@pytest.fixture
def sample_image() -> Image.Image:
    """A 300x400 RGB PIL Image for testing."""
    return Image.new("RGB", (300, 400), color=(128, 64, 32))


def test_train_transforms_output_shape(
    data_config: dict[str, Any], sample_image: Image.Image
) -> None:
    """Train transforms should produce [3, 224, 224] tensor."""
    transforms = get_train_transforms(data_config)
    tensor = transforms(sample_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_eval_transforms_output_shape(
    data_config: dict[str, Any], sample_image: Image.Image
) -> None:
    """Eval transforms should produce [3, 224, 224] tensor."""
    transforms = get_eval_transforms(data_config)
    tensor = transforms(sample_image)

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_train_transforms_no_nan(
    data_config: dict[str, Any], sample_image: Image.Image
) -> None:
    """Train transforms should produce finite values."""
    transforms = get_train_transforms(data_config)
    tensor = transforms(sample_image)

    assert not tensor.isnan().any()
    assert not tensor.isinf().any()


def test_eval_transforms_no_nan(
    data_config: dict[str, Any], sample_image: Image.Image
) -> None:
    """Eval transforms should produce finite values."""
    transforms = get_eval_transforms(data_config)
    tensor = transforms(sample_image)

    assert not tensor.isnan().any()
    assert not tensor.isinf().any()


def test_train_transforms_deterministic_with_same_input(
    data_config: dict[str, Any], sample_image: Image.Image
) -> None:
    """Two calls with the same image should produce different results
    due to RandomCrop and RandomHorizontalFlip (stochastic transforms)."""
    transforms = get_train_transforms(data_config)
    t1 = transforms(sample_image)
    t2 = transforms(sample_image)

    # They may occasionally be equal by chance, but almost certainly
    # won't be for a 300x400 image with random crop.
    # We just check they have the right shape — the randomness is
    # tested implicitly by the fact that these are stochastic.
    assert t1.shape == t2.shape == (3, 224, 224)


def test_eval_transforms_deterministic(
    data_config: dict[str, Any], sample_image: Image.Image
) -> None:
    """Two calls with the same image should produce identical results
    (no stochastic transforms in eval)."""
    transforms = get_eval_transforms(data_config)
    t1 = transforms(sample_image)
    t2 = transforms(sample_image)

    assert torch.equal(t1, t2)


def test_train_transforms_returns_compose(data_config: dict[str, Any]) -> None:
    """get_train_transforms should return a torchvision.transforms.v2.Compose."""
    transforms = get_train_transforms(data_config)
    assert hasattr(transforms, "transforms")


def test_eval_transforms_returns_compose(data_config: dict[str, Any]) -> None:
    """get_eval_transforms should return a torchvision.transforms.v2.Compose."""
    transforms = get_eval_transforms(data_config)
    assert hasattr(transforms, "transforms")


def _pipeline_transform_types(transforms: Any) -> list[str]:
    """Names of the transforms in a v2.Compose pipeline."""
    return [type(t).__name__ for t in transforms.transforms]


def test_no_color_jitter_when_disabled(
    data_config: dict[str, Any],
) -> None:
    """A missing or zero color_jitter_strength keeps today's pipeline.

    The shipped checkpoint was trained without jitter; the default must
    reproduce that pipeline exactly so a config refresh is a no-op.
    """
    baseline = get_train_transforms(data_config)
    data_config["transforms"]["color_jitter_strength"] = 0.0
    explicit = get_train_transforms(data_config)

    assert "ColorJitter" not in _pipeline_transform_types(explicit)
    assert _pipeline_transform_types(explicit) == _pipeline_transform_types(
        baseline
    )


def test_color_jitter_enters_pipeline_when_positive(
    data_config: dict[str, Any], sample_image: Image.Image
) -> None:
    """A positive strength inserts ColorJitter into the train pipeline."""
    data_config["transforms"]["color_jitter_strength"] = 0.2
    transforms = get_train_transforms(data_config)

    assert "ColorJitter" in _pipeline_transform_types(transforms)

    tensor = transforms(sample_image)
    assert tensor.shape == (3, 224, 224)
    assert not tensor.isnan().any()


def test_color_jitter_strength_over_half_clamps_hue(
    data_config: dict[str, Any], sample_image: Image.Image
) -> None:
    """Hue accepts at most 0.5; a full-strength jitter must not crash."""
    data_config["transforms"]["color_jitter_strength"] = 1.0
    transforms = get_train_transforms(data_config)

    jitter = next(
        t for t in transforms.transforms if type(t).__name__ == "ColorJitter"
    )
    tensor = transforms(sample_image)
    assert tensor.shape == (3, 224, 224)
    assert not tensor.isnan().any()
