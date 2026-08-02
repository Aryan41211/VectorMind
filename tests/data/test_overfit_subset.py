"""Tests for overfit_subset.py — deterministic subset builder."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from vectormind.data.overfit_subset import (
    CAPTIONS_PER_IMAGE,
    create_overfit_subset,
    load_subset_metadata,
    save_subset_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data() -> tuple[list[str], list[str]]:
    """Create sample image paths and captions (20 unique images, 100 pairs)."""
    image_paths: list[str] = []
    captions: list[str] = []
    for img_idx in range(20):
        img_path = f"/fake/images/{img_idx:06d}.jpg"
        for cap_idx in range(5):
            image_paths.append(img_path)
            captions.append(f"Caption {cap_idx} for image {img_idx}")
    return image_paths, captions


# ---------------------------------------------------------------------------
# Tests: create_overfit_subset
# ---------------------------------------------------------------------------


class TestCreateOverfitSubset:
    """Tests for create_overfit_subset function."""

    def test_returns_correct_number_of_pairs(
        self, sample_data: tuple[list[str], list[str]]
    ) -> None:
        """Subset should contain subset_size * 5 pairs."""
        image_paths, captions = sample_data
        pairs = create_overfit_subset(image_paths, captions, subset_size=10, seed=42)
        assert len(pairs) == 10 * CAPTIONS_PER_IMAGE

    def test_all_captions_per_image_included(
        self, sample_data: tuple[list[str], list[str]]
    ) -> None:
        """Each selected image should have all 5 captions."""
        image_paths, captions = sample_data
        pairs = create_overfit_subset(image_paths, captions, subset_size=5, seed=42)
        # Count pairs per image
        image_counts: dict[str, int] = {}
        for img_path, _ in pairs:
            image_counts[img_path] = image_counts.get(img_path, 0) + 1
        for count in image_counts.values():
            assert count == CAPTIONS_PER_IMAGE

    def test_deterministic_with_same_seed(
        self, sample_data: tuple[list[str], list[str]]
    ) -> None:
        """Same seed should produce identical subsets."""
        image_paths, captions = sample_data
        pairs1 = create_overfit_subset(image_paths, captions, subset_size=10, seed=42)
        pairs2 = create_overfit_subset(image_paths, captions, subset_size=10, seed=42)
        assert pairs1 == pairs2

    def test_different_seeds_produce_different_subsets(
        self, sample_data: tuple[list[str], list[str]]
    ) -> None:
        """Different seeds should (almost certainly) produce different subsets."""
        image_paths, captions = sample_data
        pairs1 = create_overfit_subset(image_paths, captions, subset_size=10, seed=42)
        pairs2 = create_overfit_subset(image_paths, captions, subset_size=10, seed=123)
        # Extract unique images from each subset
        imgs1 = set(p[0] for p in pairs1)
        imgs2 = set(p[0] for p in pairs2)
        # With 20 images and selecting 10, different seeds should
        # produce different selections (extremely high probability)
        assert imgs1 != imgs2

    def test_subset_images_are_from_source(
        self, sample_data: tuple[list[str], list[str]]
    ) -> None:
        """All images in subset should exist in the source data."""
        image_paths, captions = sample_data
        pairs = create_overfit_subset(image_paths, captions, subset_size=10, seed=42)
        source_images = set(image_paths)
        for img_path, _ in pairs:
            assert img_path in source_images

    def test_subset_captions_match_source(
        self, sample_data: tuple[list[str], list[str]]
    ) -> None:
        """All captions in subset should match the source captions."""
        image_paths, captions = sample_data
        pairs = create_overfit_subset(image_paths, captions, subset_size=10, seed=42)
        # Build a lookup: (image_path, caption) -> count in source
        source_pairs = set(zip(image_paths, captions))
        for img_path, cap in pairs:
            assert (img_path, cap) in source_pairs

    def test_subset_size_exceeds_unique_images_raises(
        self, sample_data: tuple[list[str], list[str]]
    ) -> None:
        """Requesting more images than available should raise ValueError."""
        image_paths, captions = sample_data
        with pytest.raises(ValueError, match="subset_size.*exceeds"):
            create_overfit_subset(image_paths, captions, subset_size=25, seed=42)

    def test_empty_input_raises(self) -> None:
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            create_overfit_subset([], [], subset_size=10, seed=42)

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="must equal"):
            create_overfit_subset(
                ["/fake/a.jpg", "/fake/b.jpg"],
                ["cap1"],
                subset_size=1,
                seed=42,
            )

    def test_subset_size_one(self, sample_data: tuple[list[str], list[str]]) -> None:
        """subset_size=1 should select exactly one image with 5 captions."""
        image_paths, captions = sample_data
        pairs = create_overfit_subset(image_paths, captions, subset_size=1, seed=42)
        assert len(pairs) == 5
        unique_images = set(p[0] for p in pairs)
        assert len(unique_images) == 1


# ---------------------------------------------------------------------------
# Tests: save_subset_metadata / load_subset_metadata
# ---------------------------------------------------------------------------


class TestSubsetMetadata:
    """Tests for saving and loading subset metadata."""

    def test_save_load_roundtrip(
        self, sample_data: tuple[list[str], list[str]]
    ) -> None:
        """Saved metadata should load back with identical pairs."""
        image_paths, captions = sample_data
        pairs = create_overfit_subset(image_paths, captions, subset_size=5, seed=42)

        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = Path(tmp_dir) / "subset.json"
            save_subset_metadata(
                pairs=pairs,
                output_path=metadata_path,
                subset_size=5,
                seed=42,
                total_images=20,
                total_pairs=100,
            )

            loaded_pairs = load_subset_metadata(metadata_path)
            assert loaded_pairs == pairs

    def test_metadata_file_is_valid_json(
        self, sample_data: tuple[list[str], list[str]]
    ) -> None:
        """Saved file should be valid JSON with expected structure."""
        image_paths, captions = sample_data
        pairs = create_overfit_subset(image_paths, captions, subset_size=5, seed=42)

        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = Path(tmp_dir) / "subset.json"
            save_subset_metadata(
                pairs=pairs,
                output_path=metadata_path,
                subset_size=5,
                seed=42,
                total_images=20,
                total_pairs=100,
            )

            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            assert "config" in metadata
            assert "source" in metadata
            assert "subset" in metadata
            assert metadata["config"]["subset_size"] == 5
            assert metadata["config"]["seed"] == 42
            assert metadata["subset"]["num_images"] == 5
            assert metadata["subset"]["num_pairs"] == 25

    def test_load_nonexistent_file_raises(self) -> None:
        """Loading from a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_subset_metadata("/nonexistent/path/subset.json")

    def test_metadata_creates_parent_dirs(
        self, sample_data: tuple[list[str], list[str]]
    ) -> None:
        """save_subset_metadata should create parent directories."""
        image_paths, captions = sample_data
        pairs = create_overfit_subset(image_paths, captions, subset_size=3, seed=42)

        with tempfile.TemporaryDirectory() as tmp_dir:
            metadata_path = Path(tmp_dir) / "nested" / "dir" / "subset.json"
            save_subset_metadata(
                pairs=pairs,
                output_path=metadata_path,
                subset_size=3,
                seed=42,
                total_images=20,
                total_pairs=100,
            )
            assert metadata_path.exists()
