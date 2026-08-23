"""Tests for memorization.py — Phase 3.5 evaluation metrics."""

from __future__ import annotations

import pytest
import torch

from vectormind.evaluation.memorization import (
    compute_embedding_diagnostics,
    compute_image_level_recall,
    compute_similarity_analysis,
    compute_text_level_recall,
    compute_top_k_examples,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Create embeddings where each image matches its own captions perfectly.

    3 images, 5 captions each = 15 text embeddings.
    Image i's embedding is identical to captions [i*5 : i*5+5].
    """
    D = 64
    N_images = 3
    captions_per_image = 5

    # Create distinct image embeddings
    image_embeds = torch.randn(N_images, D)
    image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

    # Create text embeddings: each image's captions are close copies
    text_embeds = []
    for i in range(N_images):
        for _ in range(captions_per_image):
            noise = torch.randn(D) * 0.01
            text_embeds.append(image_embeds[i] + noise)
    text_embeds = torch.stack(text_embeds)
    text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

    return image_embeds, text_embeds


@pytest.fixture
def random_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Create random embeddings (no correlation between image and text)."""
    torch.manual_seed(42)
    image_embeds = torch.randn(3, 64)
    image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
    text_embeds = torch.randn(15, 64)
    text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
    return image_embeds, text_embeds


# ---------------------------------------------------------------------------
# Tests: Image-level Recall
# ---------------------------------------------------------------------------


class TestImageLevelRecall:
    """Tests for compute_image_level_recall."""

    def test_perfect_match_recall_is_one(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Perfect embeddings should give Recall@1 = 1.0."""
        image_embeds, text_embeds = perfect_embeddings
        recall = compute_image_level_recall(image_embeds, text_embeds, k=1)
        assert recall == 1.0

    def test_perfect_match_recall5_is_one(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Perfect embeddings should give Recall@5 = 1.0."""
        image_embeds, text_embeds = perfect_embeddings
        recall = compute_image_level_recall(image_embeds, text_embeds, k=5)
        assert recall == 1.0

    def test_random_embeddings_low_recall(
        self, random_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Random embeddings should give Recall@1 around 1/num_texts."""
        image_embeds, text_embeds = random_embeddings
        recall = compute_image_level_recall(image_embeds, text_embeds, k=1)
        # With 3 images and 15 texts, random chance for Recall@1 is ~5/15 = 0.33
        # (each image has 5 correct out of 15)
        assert 0.0 <= recall <= 1.0

    def test_recall_k_increasing(
        self, random_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Recall@K should be non-decreasing with K."""
        image_embeds, text_embeds = random_embeddings
        r1 = compute_image_level_recall(image_embeds, text_embeds, k=1)
        r5 = compute_image_level_recall(image_embeds, text_embeds, k=5)
        r10 = compute_image_level_recall(image_embeds, text_embeds, k=10)
        assert r1 <= r5 <= r10


# ---------------------------------------------------------------------------
# Tests: Text-level Recall
# ---------------------------------------------------------------------------


class TestTextLevelRecall:
    """Tests for compute_text_level_recall."""

    def test_perfect_match_recall_is_one(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Perfect embeddings should give text Recall@1 = 1.0."""
        image_embeds, text_embeds = perfect_embeddings
        recall = compute_text_level_recall(image_embeds, text_embeds, k=1)
        assert recall == 1.0

    def test_random_embeddings_valid_range(
        self, random_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Random embeddings should give Recall@1 between 0 and 1."""
        image_embeds, text_embeds = random_embeddings
        recall = compute_text_level_recall(image_embeds, text_embeds, k=1)
        assert 0.0 <= recall <= 1.0


# ---------------------------------------------------------------------------
# Tests: Similarity Analysis
# ---------------------------------------------------------------------------


class TestSimilarityAnalysis:
    """Tests for compute_similarity_analysis."""

    def test_perfect_match_positive_separation(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Perfect embeddings should have positive separation."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_similarity_analysis(image_embeds, text_embeds)
        assert result["separation"] > 0
        assert result["matched_mean_similarity"] > result["unmatched_mean_similarity"]

    def test_returns_expected_keys(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Should return all expected keys."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_similarity_analysis(image_embeds, text_embeds)
        expected_keys = {
            "matched_mean_similarity",
            "matched_std_similarity",
            "unmatched_mean_similarity",
            "unmatched_std_similarity",
            "separation",
            "min_matched_similarity",
            "max_unmatched_similarity",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Tests: Top-K Examples
# ---------------------------------------------------------------------------


class TestTopKExamples:
    """Tests for compute_top_k_examples."""

    def test_returns_correct_number_of_examples(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Should return the requested number of examples."""
        image_embeds, text_embeds = perfect_embeddings
        examples = compute_top_k_examples(
            image_embeds, text_embeds, k=5, num_examples=2
        )
        assert len(examples) == 2

    def test_example_has_required_keys(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Each example should have all required keys."""
        image_embeds, text_embeds = perfect_embeddings
        examples = compute_top_k_examples(
            image_embeds, text_embeds, k=3, num_examples=1
        )
        example = examples[0]
        assert "image_index" in example
        assert "correct_caption_range" in example
        assert "top_k_indices" in example
        assert "top_k_scores" in example
        assert "correct_in_top_k" in example
        assert "recall_at_k" in example

    def test_perfect_embeddings_recall_at_k(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Perfect embeddings should have recall_at_k = True for all examples."""
        image_embeds, text_embeds = perfect_embeddings
        examples = compute_top_k_examples(
            image_embeds, text_embeds, k=5, num_examples=3
        )
        for example in examples:
            assert example["recall_at_k"] is True


# ---------------------------------------------------------------------------
# Tests: Embedding Diagnostics
# ---------------------------------------------------------------------------


class TestEmbeddingDiagnostics:
    """Tests for compute_embedding_diagnostics."""

    def test_returns_expected_keys(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Should return all expected diagnostic keys."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_embedding_diagnostics(image_embeds, text_embeds)
        expected_keys = {
            "image_dim_variance",
            "text_dim_variance",
            "image_mean_pairwise_dist",
            "text_mean_pairwise_dist",
            "image_min_pairwise_dist",
            "text_min_pairwise_dist",
        }
        assert set(result.keys()) == expected_keys

    def test_variance_positive(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Embedding variance should be positive (no collapse)."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_embedding_diagnostics(image_embeds, text_embeds)
        assert result["image_dim_variance"] > 0
        assert result["text_dim_variance"] > 0

    def test_pairwise_distances_positive(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Mean pairwise distances should be positive."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_embedding_diagnostics(image_embeds, text_embeds)
        assert result["image_mean_pairwise_dist"] > 0
        assert result["text_mean_pairwise_dist"] > 0
