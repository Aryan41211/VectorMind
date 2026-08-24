"""Tests for retrieval.py — Phase 5 evaluation metrics."""

from __future__ import annotations

import pytest
import torch

from vectormind.evaluation.retrieval import (
    compute_alignment,
    compute_bidirectional_recall,
    compute_comprehensive_embedding_diagnostics,
    compute_failure_analysis,
    compute_retrieval_examples,
    compute_uniformity,
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

    image_embeds = torch.randn(N_images, D)
    image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

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


@pytest.fixture
def collapsed_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Create collapsed embeddings (all similar)."""
    D = 64
    N_images = 3
    captions_per_image = 5

    base = torch.randn(1, D)
    base = base / base.norm(dim=1, keepdim=True)

    noise = torch.randn(N_images, D) * 0.001
    image_embeds = base + noise
    image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)

    text_noise = torch.randn(N_images * captions_per_image, D) * 0.001
    text_embeds = base + text_noise
    text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

    return image_embeds, text_embeds


# ---------------------------------------------------------------------------
# Tests: Bidirectional Recall
# ---------------------------------------------------------------------------


class TestComputeBidirectionalRecall:
    """Tests for compute_bidirectional_recall."""

    def test_returns_expected_keys(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Should return all 6 recall keys."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_bidirectional_recall(image_embeds, text_embeds)
        expected_keys = [
            "image_to_text_recall@1",
            "image_to_text_recall@5",
            "image_to_text_recall@10",
            "text_to_image_recall@1",
            "text_to_image_recall@5",
            "text_to_image_recall@10",
        ]
        assert all(k in result for k in expected_keys)

    def test_perfect_embeddings_high_recall(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Perfect embeddings should give Recall@1 close to 1.0."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_bidirectional_recall(image_embeds, text_embeds)
        assert result["image_to_text_recall@1"] == 1.0
        assert result["text_to_image_recall@1"] == 1.0

    def test_random_embeddings_baseline(
        self, random_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Random embeddings should give valid recall values."""
        image_embeds, text_embeds = random_embeddings
        result = compute_bidirectional_recall(image_embeds, text_embeds)
        for value in result.values():
            assert 0.0 <= value <= 1.0

    def test_recall_k_non_decreasing(
        self, random_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Recall@K should be non-decreasing with K."""
        image_embeds, text_embeds = random_embeddings
        result = compute_bidirectional_recall(image_embeds, text_embeds)
        assert result["image_to_text_recall@1"] <= result["image_to_text_recall@5"]
        assert result["image_to_text_recall@5"] <= result["image_to_text_recall@10"]
        assert result["text_to_image_recall@1"] <= result["text_to_image_recall@5"]
        assert result["text_to_image_recall@5"] <= result["text_to_image_recall@10"]


# ---------------------------------------------------------------------------
# Tests: Uniformity
# ---------------------------------------------------------------------------


class TestComputeUniformity:
    """Tests for compute_uniformity."""

    def test_returns_float(self) -> None:
        """Should return a float value."""
        embeddings = torch.randn(10, 64)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        result = compute_uniformity(embeddings)
        assert isinstance(result, float)

    def test_uniform_embeddings_low_value(self) -> None:
        """Uniformly distributed embeddings should have lower uniformity."""
        torch.manual_seed(42)
        uniform = torch.randn(100, 64)
        uniform = uniform / uniform.norm(dim=1, keepdim=True)
        result = compute_uniformity(uniform)
        assert result < 0.0

    def test_collapsed_embeddings_high_value(self) -> None:
        """Collapsed embeddings should have higher uniformity."""
        base = torch.randn(1, 64)
        collapsed = base.repeat(100, 1) + torch.randn(100, 64) * 0.01
        collapsed = collapsed / collapsed.norm(dim=1, keepdim=True)
        result = compute_uniformity(collapsed)
        assert result > -1.0


# ---------------------------------------------------------------------------
# Tests: Alignment
# ---------------------------------------------------------------------------


class TestComputeAlignment:
    """Tests for compute_alignment."""

    def test_returns_float(self) -> None:
        """Should return a float value."""
        img = torch.randn(10, 64)
        txt = torch.randn(10, 64)
        result = compute_alignment(img, txt)
        assert isinstance(result, float)

    def test_perfect_alignment_low_value(self) -> None:
        """Perfectly aligned embeddings should have low alignment."""
        embeddings = torch.randn(10, 64)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
        result = compute_alignment(embeddings, embeddings)
        assert result < 0.01

    def test_misaligned_embeddings_high_value(self) -> None:
        """Misaligned embeddings should have higher alignment score."""
        torch.manual_seed(42)
        img = torch.randn(10, 64)
        img = img / img.norm(dim=1, keepdim=True)
        txt = torch.randn(10, 64)
        txt = txt / txt.norm(dim=1, keepdim=True)
        result = compute_alignment(img, txt)
        assert result > 0.5


# ---------------------------------------------------------------------------
# Tests: Comprehensive Embedding Diagnostics
# ---------------------------------------------------------------------------


class TestComprehensiveEmbeddingDiagnostics:
    """Tests for compute_comprehensive_embedding_diagnostics."""

    def test_returns_expected_keys(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Should return all diagnostic keys."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_comprehensive_embedding_diagnostics(
            image_embeds, text_embeds
        )
        expected_keys = [
            "image_dim_variance",
            "text_dim_variance",
            "image_mean_pairwise_dist",
            "text_mean_pairwise_dist",
            "image_min_pairwise_dist",
            "text_min_pairwise_dist",
            "image_uniformity",
            "text_uniformity",
            "alignment",
        ]
        assert all(k in result for k in expected_keys)

    def test_variance_positive(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Variance should be positive."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_comprehensive_embedding_diagnostics(
            image_embeds, text_embeds
        )
        assert result["image_dim_variance"] > 0
        assert result["text_dim_variance"] > 0

    def test_pairwise_distances_positive(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Pairwise distances should be positive."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_comprehensive_embedding_diagnostics(
            image_embeds, text_embeds
        )
        assert result["image_mean_pairwise_dist"] > 0
        assert result["text_mean_pairwise_dist"] > 0

    def test_collapsed_embeddings_low_variance(
        self, collapsed_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Collapsed embeddings should have lower variance."""
        image_embeds, text_embeds = collapsed_embeddings
        result = compute_comprehensive_embedding_diagnostics(
            image_embeds, text_embeds
        )
        assert result["image_dim_variance"] < 0.001


# ---------------------------------------------------------------------------
# Tests: Retrieval Examples
# ---------------------------------------------------------------------------


class TestComputeRetrievalExamples:
    """Tests for compute_retrieval_examples."""

    def test_returns_successes_and_failures(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Should return both successes and failures."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_retrieval_examples(
            image_embeds, text_embeds, num_successes=2, num_failures=2
        )
        assert "successes" in result
        assert "failures" in result
        assert len(result["successes"]) > 0

    def test_examples_contain_required_fields(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Each example should contain all required fields."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_retrieval_examples(
            image_embeds, text_embeds, num_successes=1, num_failures=1
        )
        required_fields = [
            "image_index",
            "image_path",
            "correct_caption_range",
            "query_caption",
            "top_k_indices",
            "top_k_scores",
            "top_k_captions",
            "correct_in_top_k",
            "recall_at_k",
        ]
        for example in result["successes"] + result["failures"]:
            assert all(f in example for f in required_fields)

    def test_handles_metadata_none(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Should handle None metadata gracefully."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_retrieval_examples(
            image_embeds, text_embeds, image_paths=None, captions=None
        )
        assert len(result["successes"]) >= 0

    def test_stride_spreads_examples_across_the_split(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """A stride should draw examples from beyond the first rows."""
        image_embeds, text_embeds = perfect_embeddings
        strided = compute_retrieval_examples(
            image_embeds, text_embeds, num_successes=2, num_failures=0, stride=2
        )
        indices = [e["image_index"] for e in strided["successes"]]
        assert indices == [0, 2]

    def test_stride_of_one_keeps_the_original_behaviour(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """The default must remain a contiguous scan from row 0."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_retrieval_examples(
            image_embeds, text_embeds, num_successes=2, num_failures=0
        )
        assert [e["image_index"] for e in result["successes"]] == [0, 1]

    def test_rejects_a_non_positive_stride(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """A stride of 0 would loop forever; reject it explicitly."""
        image_embeds, text_embeds = perfect_embeddings
        with pytest.raises(ValueError, match="stride must be"):
            compute_retrieval_examples(image_embeds, text_embeds, stride=0)


# ---------------------------------------------------------------------------
# Tests: Failure Analysis
# ---------------------------------------------------------------------------


class TestComputeFailureAnalysis:
    """Tests for compute_failure_analysis."""

    def test_returns_expected_keys(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Should return all failure analysis keys."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_failure_analysis(image_embeds, text_embeds)
        expected_keys = [
            "total_images",
            "total_failures",
            "failure_rate",
            "success_rate",
            "hit_rank_distribution",
        ]
        assert all(k in result for k in expected_keys)

    def test_perfect_embeddings_zero_failures(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Perfect embeddings should have zero failures."""
        image_embeds, text_embeds = perfect_embeddings
        result = compute_failure_analysis(image_embeds, text_embeds)
        assert result["total_failures"] == 0
        assert result["failure_rate"] == 0.0
        assert result["success_rate"] == 1.0

    def test_failure_rate_between_zero_and_one(
        self, random_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Failure rate should be between 0 and 1."""
        image_embeds, text_embeds = random_embeddings
        result = compute_failure_analysis(image_embeds, text_embeds)
        assert 0.0 <= result["failure_rate"] <= 1.0
        assert 0.0 <= result["success_rate"] <= 1.0

    def test_rates_sum_to_one(
        self, random_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Failure rate and success rate should sum to 1.0."""
        image_embeds, text_embeds = random_embeddings
        result = compute_failure_analysis(image_embeds, text_embeds)
        assert abs(result["failure_rate"] + result["success_rate"] - 1.0) < 1e-6

    def test_hit_ranks_account_for_every_success(
        self, random_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """The histogram must sum to the number of successful queries."""
        image_embeds, text_embeds = random_embeddings
        result = compute_failure_analysis(image_embeds, text_embeds)
        successes = result["total_images"] - result["total_failures"]
        assert sum(result["hit_rank_distribution"]) == successes

    def test_perfect_embeddings_all_hit_at_rank_one(
        self, perfect_embeddings: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Perfect retrieval puts every correct caption first.

        The previous implementation incremented bucket 0 on *failure*,
        so a perfect run reported an empty histogram and a total failure
        run reported every image at rank 1 — the reverse of the truth.
        """
        image_embeds, text_embeds = perfect_embeddings
        result = compute_failure_analysis(image_embeds, text_embeds)
        distribution = result["hit_rank_distribution"]
        assert distribution[0] == result["total_images"]
        assert sum(distribution[1:]) == 0
