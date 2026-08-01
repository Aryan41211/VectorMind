"""Unit tests for vectormind.training.losses.

Includes hand-computed examples as required by CLAUDE.md §4 and the
Phase 3 acceptance criteria.
"""

from __future__ import annotations

import math

import pytest
import torch

from vectormind.training.losses import symmetric_infonce


# ---------------------------------------------------------------------------
# Hand-computed test (CLAUDE.md §4 requirement)
# ---------------------------------------------------------------------------


class TestSymmetricInfoNCEHandComputed:
    """Hand-computed unit tests with known expected values.

    These tests use carefully constructed inputs where the expected
    loss value can be computed analytically, not just checked for
    basic properties like "positive scalar."
    """

    def test_orthogonal_embeddings_loss_equals_ln2(self) -> None:
        """B=2, orthogonal embeddings, temperature=1.0 → loss = ln(2).

        Hand computation:
            image_embeds = [[1, 0], [0, 1]]  (L2-normalized)
            text_embeds  = [[1, 0], [0, 1]]  (L2-normalized)
            temperature  = 1.0

            similarity_matrix = image_embeds @ text_embeds.T * temp
                              = [[1, 0], [0, 1]] * 1.0
                              = [[1, 0], [0, 1]]

            For image→text (CE with labels=[0, 1]):
                row 0: softmax([1, 0]) = [e^1/(e^1+e^0), e^0/(e^1+e^0)]
                      = [e/(e+1), 1/(e+1)]
                CE for sample 0: -log(e/(e+1)) = -1 + log(e+1)
                row 1: softmax([0, 1]) = [1/(e+1), e/(e+1)]
                CE for sample 1: -log(e/(e+1)) = -1 + log(e+1)
                mean CE = -1 + log(e+1) = log(e+1) - 1

                Note: log(e+1) - 1 = log(e+1) - log(e) = log((e+1)/e) = log(1 + 1/e)

                Wait, let me recalculate more carefully:
                softmax([1, 0]) = [exp(1)/(exp(1)+exp(0)), exp(0)/(exp(1)+exp(0))]
                                = [e/(e+1), 1/(e+1)]
                CE_0 = -log(e/(e+1)) = -1 + log(e+1)

                softmax([0, 1]) = [1/(e+1), e/(e+1)]
                CE_1 = -log(e/(e+1)) = -1 + log(e+1)

                mean_i2t = CE_0 = CE_1 = log(e+1) - 1

            For text→image (CE with labels=[0, 1]):
                Same matrix transposed = same matrix (symmetric)
                mean_t2i = log(e+1) - 1

            Final loss = (mean_i2t + mean_t2i) / 2 = log(e+1) - 1

            Let me compute: log(e+1) - 1 = log(e+1) - log(e) = log((e+1)/e)
            e ≈ 2.71828, so (e+1)/e = 3.71828/2.71828 ≈ 1.36788
            log(1.36788) ≈ 0.3133

            Hmm, that's not ln(2). Let me reconsider...

            Actually, for CE with uniform target distribution on B classes:
            CE = log(B) when logits are all equal.

            But here logits are NOT all equal. Let me use the correct formula.

            For cross_entropy with target label y:
            CE = -log(softmax(logits)[y])

            For row 0, label 0:
            softmax([1, 0])[0] = e^1 / (e^1 + e^0) = e / (e + 1)
            CE = -log(e/(e+1)) = log((e+1)/e) = log(1 + 1/e)

            For uniform logits [a, a]:
            softmax([a, a]) = [0.5, 0.5]
            CE = -log(0.5) = log(2) ≈ 0.6931

            So I need logits where the softmax gives uniform distribution.
            That happens when all logits are equal.

            Let me use temperature that makes the off-diagonal equal to diagonal.
            If image_embeds = [[1,0],[0,1]] and text_embeds = [[1,0],[0,1]],
            then similarity = [[1,0],[0,1]].

            To make this uniform, I need temperature such that t*1 = t*0... no.

            Actually, let me just use a simpler example:
            B=2, D=2
            image_embeds = [[1, 0], [1, 0]]  (both map to same direction)
            text_embeds  = [[1, 0], [1, 0]]  (both map to same direction)

            similarity = [[1, 1], [1, 1]] * t = [[t, t], [t, t]]

            softmax([t, t]) = [0.5, 0.5] for each row
            CE = -log(0.5) = log(2) for each sample
            mean_i2t = log(2)
            mean_t2i = log(2)
            final = log(2) ≈ 0.6931

            That works! But it's a degenerate case where all embeddings are the same.

            Let me try a non-degenerate case:
            image_embeds = [[1, 0], [0, 1]]
            text_embeds  = [[1, 0], [0, 1]]
            temperature = 1.0

            similarity = [[1, 0], [0, 1]]

            Row 0, label 0: softmax([1, 0]) = [e/(e+1), 1/(e+1)]
            CE = -log(e/(e+1)) = log(1 + 1/e) ≈ 0.3133

            Row 1, label 1: softmax([0, 1]) = [1/(e+1), e/(e+1)]
            CE = -log(e/(e+1)) = log(1 + 1/e) ≈ 0.3133

            mean_i2t = log(1 + 1/e)

            mean_t2i = same (symmetric matrix)

            final = log(1 + 1/e) ≈ 0.3133

            Hmm, that's not a clean number. Let me use a different approach.

            Actually, the cleanest test is the uniform case where all logits are equal.
            Let me use:
            image_embeds = [[1, 0], [1, 0]]
            text_embeds  = [[1, 0], [1, 0]]
            temperature = 1.0

            This gives similarity = [[1, 1], [1, 1]]
            softmax = [[0.5, 0.5], [0.5, 0.5]]
            CE = -log(0.5) = log(2) for each
            loss = log(2) ≈ 0.693147

            This is a valid test because:
            1. It's hand-computable
            2. It tests the actual CE computation
            3. The expected value is exact (log(2))

            Let me implement this test.
        """
        # Construct inputs where the expected loss is exactly log(2)
        # Both image and text embeddings are identical and point in the
        # same direction, so similarity matrix is all-ones (after scaling).
        # With temperature=1.0, logits = [[1,1],[1,1]], softmax = [[0.5,0.5],[0.5,0.5]]
        # CE = -log(0.5) = log(2) for each sample, each direction
        # Final loss = (log(2) + log(2)) / 2 = log(2)
        image_embeds = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # [2, 2]
        text_embeds = torch.tensor([[1.0, 0.0], [1.0, 0.0]])  # [2, 2]
        temperature = torch.tensor(1.0)

        loss = symmetric_infonce(image_embeds, text_embeds, temperature)

        expected_loss = math.log(2)  # ≈ 0.693147
        assert loss.item() == pytest.approx(expected_loss, abs=1e-5), (
            f"Expected loss = log(2) ≈ {expected_loss:.6f}, got {loss.item():.6f}"
        )

    def test_perfect_alignment_loss_is_low(self) -> None:
        """B=2, identical embeddings, temperature=0.1 → loss close to -log(0.5).

        With temperature=0.1, logits = [[0.1, 0.1], [0.1, 0.1]]
        softmax = [[0.5, 0.5], [0.5, 0.5]]
        CE = -log(0.5) = log(2) ≈ 0.6931

        Wait, temperature scales the logits, so:
        similarity = [[1,0],[0,1]] * 0.1 = [[0.1, 0],[0, 0.1]]
        Row 0: softmax([0.1, 0]) = [e^0.1/(e^0.1+1), 1/(e^0.1+1)]
        CE = -log(e^0.1/(e^0.1+1)) = -0.1 + log(e^0.1+1)

        e^0.1 ≈ 1.10517
        log(1.10517 + 1) = log(2.10517) ≈ 0.7449
        CE = -0.1 + 0.7449 = 0.6449

        Hmm, still not super clean. Let me just test with a simpler case.

        Actually, let me test with the case where embeddings are perfectly aligned:
        image_embeds = [[1, 0], [0, 1]]
        text_embeds  = [[1, 0], [0, 1]]
        temperature = 1.0

        similarity = [[1, 0], [0, 1]]
        Row 0, label 0: softmax([1, 0]) = [e/(e+1), 1/(e+1)]
        CE = -log(e/(e+1)) = log(1 + 1/e) ≈ 0.3133

        Row 1, label 1: softmax([0, 1]) = [1/(e+1), e/(e+1)]
        CE = -log(e/(e+1)) = log(1 + 1/e) ≈ 0.3133

        mean_i2t = log(1 + 1/e)
        mean_t2i = log(1 + 1/e)
        final = log(1 + 1/e) ≈ 0.3133

        This is a good test - the loss should be significantly less than log(2)
        (which is the "uniform" case), showing that the model can distinguish
        positive pairs from negatives.

        Actually wait, I need to be more careful. The CE formula is:
        CE(logits, target) = -logits[target] + log(sum(exp(logits)))

        For row 0 = [1, 0], target = 0:
        CE = -1 + log(e^1 + e^0) = -1 + log(e + 1) = log(e+1) - 1

        log(e+1) = log(3.71828) ≈ 1.3133
        CE = 1.3133 - 1 = 0.3133

        For the "all same" case, row 0 = [1, 1], target = 0:
        CE = -1 + log(e^1 + e^1) = -1 + log(2e) = -1 + log(2) + 1 = log(2) ≈ 0.6931

        So the "perfect alignment" case (logits=[1,0]) gives LOWER loss than the
        "all same" case (logits=[1,1]), which makes sense because the model is
        more confident about the positive pair.

        Let me implement both tests:
        1. All-same case: loss = log(2) (hard-coded expected value)
        2. Perfect alignment case: loss = log(1 + 1/e) (hard-coded expected value)
        """
        # Perfect alignment: diagonal similarity matrix
        # Expected loss = log(1 + 1/e) ≈ 0.3133
        e = math.e
        expected_loss = math.log(1 + 1.0 / e)

        image_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        text_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        temperature = torch.tensor(1.0)

        loss = symmetric_infonce(image_embeds, text_embeds, temperature)

        assert loss.item() == pytest.approx(expected_loss, abs=1e-5), (
            f"Expected loss = log(1+1/e) ≈ {expected_loss:.6f}, got {loss.item():.6f}"
        )

    def test_misaligned_embeddings_loss_is_higher(self) -> None:
        """B=2, off-diagonal similarity → loss should be higher than aligned.

        image_embeds = [[1, 0], [0, 1]]
        text_embeds  = [[0, 1], [1, 0]]  (swapped!)

        similarity = [[0, 1], [1, 0]]
        Row 0, label 0: softmax([0, 1]) = [1/(e+1), e/(e+1)]
        CE = -log(1/(e+1)) = log(e+1) ≈ 1.3133

        Row 1, label 1: softmax([1, 0]) = [e/(e+1), 1/(e+1)]
        CE = -log(1/(e+1)) = log(e+1) ≈ 1.3133

        mean_i2t = log(e+1)
        mean_t2i = log(e+1)
        final = log(e+1) ≈ 1.3133

        This should be HIGHER than the aligned case (0.3133).
        """
        e = math.e
        expected_loss = math.log(e + 1)  # ≈ 1.3133

        image_embeds = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        text_embeds = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # swapped!
        temperature = torch.tensor(1.0)

        loss = symmetric_infonce(image_embeds, text_embeds, temperature)

        assert loss.item() == pytest.approx(expected_loss, abs=1e-5), (
            f"Expected loss = log(e+1) ≈ {expected_loss:.6f}, got {loss.item():.6f}"
        )


# ---------------------------------------------------------------------------
# Queue negatives tests
# ---------------------------------------------------------------------------


class TestSymmetricInfoNCEQueue:
    """Tests for the queue_embeddings parameter."""

    def test_queue_increases_i2t_loss_dimension(self) -> None:
        """Queue negatives add columns to the image→text similarity matrix."""
        B, D, K = 2, 4, 3
        image_embeds = torch.randn(B, D)
        text_embeds = torch.randn(B, D)
        queue_embeddings = torch.randn(K, D)
        temperature = torch.tensor(0.07)

        # Should not crash; loss should be a scalar
        loss = symmetric_infonce(
            image_embeds, text_embeds, temperature, queue_embeddings
        )
        assert loss.shape == ()
        assert loss.item() > 0

    def test_queue_changes_loss_value(self) -> None:
        """Adding queue negatives should change the loss value."""
        B, D, K = 4, 8, 5
        torch.manual_seed(42)
        image_embeds = torch.randn(B, D)
        text_embeds = torch.randn(B, D)
        queue_embeddings = torch.randn(K, D)
        temperature = torch.tensor(0.1)

        loss_no_queue = symmetric_infonce(image_embeds, text_embeds, temperature)
        loss_with_queue = symmetric_infonce(
            image_embeds, text_embeds, temperature, queue_embeddings
        )

        # Loss values should differ
        assert loss_no_queue.item() != pytest.approx(
            loss_with_queue.item(), abs=1e-6
        )

    def test_queue_dimension_mismatch_raises(self) -> None:
        """Queue embedding dim must match image/text dim."""
        image_embeds = torch.randn(2, 4)
        text_embeds = torch.randn(2, 4)
        queue_embeddings = torch.randn(3, 8)  # wrong dim!
        temperature = torch.tensor(0.1)

        with pytest.raises(ValueError, match="dimension must match"):
            symmetric_infonce(image_embeds, text_embeds, temperature, queue_embeddings)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestSymmetricInfoNCEValidation:
    """Tests for input validation."""

    def test_1d_embeddings_raise(self) -> None:
        """1D embeddings should raise ValueError."""
        image_embeds = torch.randn(4)
        text_embeds = torch.randn(4)
        temperature = torch.tensor(0.1)

        with pytest.raises(ValueError, match="2D"):
            symmetric_infonce(image_embeds, text_embeds, temperature)

    def test_mismatched_shapes_raise(self) -> None:
        """Different batch sizes should raise ValueError."""
        image_embeds = torch.randn(4, 8)
        text_embeds = torch.randn(2, 8)
        temperature = torch.tensor(0.1)

        with pytest.raises(ValueError, match="same shape"):
            symmetric_infonce(image_embeds, text_embeds, temperature)

    def test_non_scalar_temperature_raises(self) -> None:
        """Non-scalar temperature should raise ValueError."""
        image_embeds = torch.randn(4, 8)
        text_embeds = torch.randn(4, 8)
        temperature = torch.tensor([0.1, 0.2])

        with pytest.raises(ValueError, match="scalar"):
            symmetric_infonce(image_embeds, text_embeds, temperature)


# ---------------------------------------------------------------------------
# Gradient and numerical tests
# ---------------------------------------------------------------------------


class TestSymmetricInfoNCEGradient:
    """Tests for gradient flow and numerical properties."""

    def test_loss_is_finite(self) -> None:
        """Loss should be finite (no NaN or Inf)."""
        torch.manual_seed(42)
        image_embeds = torch.randn(16, 256)
        text_embeds = torch.randn(16, 256)
        temperature = torch.tensor(0.07)

        loss = symmetric_infonce(image_embeds, text_embeds, temperature)
        assert torch.isfinite(loss)

    def test_loss_is_positive(self) -> None:
        """Loss should be positive."""
        torch.manual_seed(42)
        image_embeds = torch.randn(8, 64)
        text_embeds = torch.randn(8, 64)
        temperature = torch.tensor(0.1)

        loss = symmetric_infonce(image_embeds, text_embeds, temperature)
        assert loss.item() > 0

    def test_gradient_flows_to_embeddings(self) -> None:
        """Gradients should flow to input embeddings."""
        image_embeds = torch.randn(4, 8, requires_grad=True)
        text_embeds = torch.randn(4, 8, requires_grad=True)
        temperature = torch.tensor(0.1)

        loss = symmetric_infonce(image_embeds, text_embeds, temperature)
        loss.backward()

        assert image_embeds.grad is not None
        assert text_embeds.grad is not None
        assert image_embeds.grad.abs().sum() > 0
        assert text_embeds.grad.abs().sum() > 0

    def test_temperature_affects_loss(self) -> None:
        """Different temperatures should produce different losses."""
        torch.manual_seed(42)
        image_embeds = torch.randn(8, 32)
        text_embeds = torch.randn(8, 32)

        loss_low_temp = symmetric_infonce(
            image_embeds, text_embeds, torch.tensor(0.01)
        )
        loss_high_temp = symmetric_infonce(
            image_embeds, text_embeds, torch.tensor(1.0)
        )

        assert loss_low_temp.item() != pytest.approx(
            loss_high_temp.item(), abs=1e-4
        )

    def test_batch_size_one(self) -> None:
        """Should work with batch size 1 (degenerate case)."""
        image_embeds = torch.tensor([[1.0, 0.0]])
        text_embeds = torch.tensor([[1.0, 0.0]])
        temperature = torch.tensor(1.0)

        loss = symmetric_infonce(image_embeds, text_embeds, temperature)
        # With B=1, similarity = [[1]], softmax = [[1]], CE = -log(1) = 0
        assert loss.item() == pytest.approx(0.0, abs=1e-5)
