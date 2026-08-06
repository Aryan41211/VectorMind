"""Evaluation: Recall@K metrics and embedding-space diagnostics.

Populated in Phase 4/5 (see ROADMAP.md). Covers image->text and
text->image retrieval metrics on held-out splits, plus embedding
collapse/uniformity checks.
"""

from vectormind.evaluation.memorization import (
    compute_embedding_diagnostics,
    compute_image_level_recall,
    compute_similarity_analysis,
    compute_text_level_recall,
    compute_top_k_examples,
)
from vectormind.evaluation.retrieval import (
    compute_alignment,
    compute_bidirectional_recall,
    compute_comprehensive_embedding_diagnostics,
    compute_failure_analysis,
    compute_retrieval_examples,
    compute_uniformity,
)

__all__ = [
    "compute_alignment",
    "compute_bidirectional_recall",
    "compute_comprehensive_embedding_diagnostics",
    "compute_embedding_diagnostics",
    "compute_failure_analysis",
    "compute_image_level_recall",
    "compute_retrieval_examples",
    "compute_similarity_analysis",
    "compute_text_level_recall",
    "compute_top_k_examples",
    "compute_uniformity",
]
