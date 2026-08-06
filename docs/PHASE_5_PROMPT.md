# Phase 5 Evaluation — Detailed Execution Prompt

## Overview

Phase 5 evaluates the trained VectorMind model on the held-out test set,
quantitatively and qualitatively. The best checkpoint is at Epoch 7,
Step 7944 (`checkpoints/train/best_model.pt`) with val Recall@10 of 20.26%.

**Critical Context:**
- Model trained from scratch (no pretrained CLIP weights)
- Flickr30k dataset (~30k images, 5 captions each)
- RTX 4050 laptop GPU, 6GB VRAM
- Embedding collapse occurred after Epoch 7 — Epoch 7 is the true best

---

## MANDATORY INCREMENTAL GIT WORKFLOW

Every completed logical implementation MUST immediately be:
1. Validated
2. Committed
3. Pushed to origin/main

before continuing.

### Commit Structure

**COMMIT 1 — Evaluation Infrastructure**
- Create `src/vectormind/evaluation/retrieval.py` with full evaluation functions
- Create `scripts/evaluate_test_set.py` for test set evaluation
- Create `tests/evaluation/test_retrieval.py` with unit tests
- Run tests, commit, push

**COMMIT 2 — Test Set Evaluation**
- Run evaluation on test set
- Generate metrics report
- Create `reports/phase5_test_metrics.json`
- Commit, push

**COMMIT 3 — Embedding Diagnostics**
- Compute embedding space diagnostics on test set
- Analyze collapse/uniformity
- Create `reports/phase5_embedding_diagnostics.json`
- Commit, push

**COMMIT 4 — Qualitative Analysis**
- Generate retrieval examples (successes and failures)
- Analyze patterns in failures
- Create `reports/phase5_qualitative_analysis.md`
- Create visualizations if possible
- Commit, push

**COMMIT 5 — Documentation Update**
- Update ROADMAP.md with Phase 5 results
- Update PROJECT_STATUS.md
- Update TRAINING_LOG.md
- Create `reports/phase5_final_report.md`
- Commit, push

**COMMIT 6 — Final Verification**
- Run all tests
- Verify no regressions
- Clean repository
- Final engineering report
- Commit, push

---

## DETAILED TASK SPECIFICATIONS

### Task 1: Evaluation Infrastructure

#### 1.1 Create `src/vectormind/evaluation/retrieval.py`

This module provides comprehensive retrieval evaluation functions.

**Required Functions:**

```python
def compute双向_recall(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    captions_per_image: int = 5,
) -> dict[str, float]:
    """Compute Recall@1/5/10 for both image→text and text→image directions.
    
    Returns:
        Dictionary with keys:
        - "image_to_text_recall@1"
        - "image_to_text_recall@5"
        - "image_to_text_recall@10"
        - "text_to_image_recall@1"
        - "text_to_image_recall@5"
        - "text_to_image_recall@10"
    """
```

```python
def compute_comprehensive_embedding_diagnostics(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> dict[str, Any]:
    """Compute comprehensive embedding space diagnostics.
    
    Returns:
        Dictionary with:
        - Per-dimension variance (image and text)
        - Mean pairwise distances
        - Min/max pairwise distances
        - Uniformity metric (optional)
        - Alignment metric (optional)
    """
```

```python
def compute_retrieval_examples(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    image_paths: list[str],
    captions: list[str],
    captions_per_image: int = 5,
    k: int = 10,
    num_successes: int = 5,
    num_failures: int = 5,
) -> dict[str, Any]:
    """Generate retrieval examples for qualitative analysis.
    
    Returns:
        Dictionary with:
        - "successes": list of successful retrieval examples
        - "failures": list of failed retrieval examples
        - Each example contains: image_path, query_caption, retrieved_captions, scores
    """
```

```python
def compute_retrieval_accuracy_by_category(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    image_categories: list[str],
    captions_per_image: int = 5,
    k: int = 10,
) -> dict[str, float]:
    """Compute retrieval accuracy broken down by image category.
    
    Useful for understanding which types of images the model handles well.
    """
```

#### 1.2 Create `scripts/evaluate_test_set.py`

Main evaluation script that:
1. Loads the best checkpoint (Epoch 7)
2. Loads the test set
3. Computes all metrics
4. Saves results to `reports/phase5_test_metrics.json`
5. Prints summary to console

**Script Structure:**
```python
"""Phase 5: Evaluate trained model on test set.

Usage:
    python scripts/evaluate_test_set.py
    python scripts/evaluate_test_set.py --checkpoint checkpoints/train/best_model.pt
"""

def main():
    # 1. Load configs
    # 2. Load model from checkpoint
    # 3. Load test set
    # 4. Compute embeddings for all test images and captions
    # 5. Compute Recall@1/5/10 for both directions
    # 6. Compute embedding diagnostics
    # 7. Save results to JSON
    # 8. Print summary
```

#### 1.3 Create `tests/evaluation/test_retrieval.py`

Unit tests for the new evaluation functions:

```python
"""Tests for retrieval.py — Phase 5 evaluation metrics."""

class TestComputeBidirectionalRecall:
    """Tests for compute双向_recall."""
    
    def test_perfect_embeddings_perfect_recall(self):
        """Perfect embeddings should give Recall@1 = 1.0 in both directions."""
        
    def test_random_embeddings_baseline_recall(self):
        """Random embeddings should give expected baseline recall."""
        
    def test_recall_k_is_non_decreasing(self):
        """Recall@K should be non-decreasing with K."""
        
    def test_symmetric_recall_values(self):
        """Both directions should produce valid values."""
        
    def test_handles_single_image(self):
        """Should handle edge case of single image."""

class TestComprehensiveEmbeddingDiagnostics:
    """Tests for compute_comprehensive_embedding_diagnostics."""
    
    def test_returns_expected_keys(self):
        """Should return all required diagnostic keys."""
        
    def test_variance_positive(self):
        """Variance should be positive for non-collapsed embeddings."""
        
    def test_pairwise_distances_positive(self):
        """Pairwise distances should be positive."""

class TestRetrievalExamples:
    """Tests for compute_retrieval_examples."""
    
    def test_returns_successes_and_failures(self):
        """Should return both successes and failures."""
        
    def test_examples_contain_required_fields(self):
        """Each example should contain all required fields."""
```

**Acceptance Criteria for COMMIT 1:**
- [ ] All functions implemented with type hints and docstrings
- [ ] All unit tests pass (`pytest tests/evaluation/test_retrieval.py -v`)
- [ ] Script runs without errors on a small subset
- [ ] Code follows project style (no comments unless asked, proper imports)

---

### Task 2: Test Set Evaluation

#### 2.1 Run Full Evaluation

Execute `scripts/evaluate_test_set.py` with the best checkpoint.

**Expected Output:**
```
Phase 5 Test Set Evaluation Results
====================================

Image-to-Text Retrieval:
  Recall@1:  X.XX% (X.Xx random baseline)
  Recall@5:  X.XX% (X.Xx random baseline)
  Recall@10: X.XX% (X.Xx random baseline)

Text-to-Image Retrieval:
  Recall@1:  X.XX% (X.Xx random baseline)
  Recall@5:  X.XX% (X.Xx random baseline)
  Recall@10: X.XX% (X.Xx random baseline)

Embedding Diagnostics:
  Image dim variance: X.XXXXXX
  Text dim variance:  X.XXXXXX
  Image mean pairwise dist: X.XXXX
  Text mean pairwise dist:  X.XXXX

Random Baseline Context:
  Recall@1:  ~1% (for 100 candidate captions)
  Recall@5:  ~5%
  Recall@10: ~10%
```

#### 2.2 Generate Metrics Report

Save results to `reports/phase5_test_metrics.json`:
```json
{
  "checkpoint": "checkpoints/train/best_model.pt",
  "epoch": 7,
  "step": 7944,
  "test_set_size": {
    "images": 3179,
    "captions": 15895
  },
  "image_to_text": {
    "recall@1": 0.0422,
    "recall@5": 0.1403,
    "recall@10": 0.2026
  },
  "text_to_image": {
    "recall@1": 0.0422,
    "recall@5": 0.1403,
    "recall@10": 0.2026
  },
  "embedding_diagnostics": {
    "image_dim_variance": 0.000746,
    "text_dim_variance": 0.000471,
    "image_mean_pairwise_dist": 0.6024,
    "text_mean_pairwise_dist": 0.4744
  },
  "random_baseline": {
    "recall@1": 0.01,
    "recall@5": 0.05,
    "recall@10": 0.10
  }
}
```

**Acceptance Criteria for COMMIT 2:**
- [ ] Evaluation script runs to completion
- [ ] Metrics saved to JSON file
- [ ] Results are reproducible (run twice, get same results)
- [ ] Comparison to random baseline documented

---

### Task 3: Embedding Diagnostics

#### 3.1 Comprehensive Analysis

Analyze the embedding space for:
1. **Collapse detection:** Is variance healthy? Are embeddings spreading?
2. **Uniformity:** Are embeddings uniformly distributed on the hypersphere?
3. **Alignment:** Are matched pairs closer than unmatched pairs?
4. **Dimensionality utilization:** Are all dimensions being used?

#### 3.2 Generate Diagnostics Report

Save to `reports/phase5_embedding_diagnostics.json`:
```json
{
  "collapse_analysis": {
    "image_variance": 0.000746,
    "text_variance": 0.000471,
    "variance_healthy": true,
    "notes": "Variance slightly below 0.001 threshold but pairwise distances healthy"
  },
  "uniformity_analysis": {
    "image_uniformity": 0.85,
    "text_uniformity": 0.82,
    "notes": "Moderate uniformity, not perfectly uniform but acceptable"
  },
  "alignment_analysis": {
    "matched_mean_similarity": 0.45,
    "unmatched_mean_similarity": 0.12,
    "separation": 0.33,
    "notes": "Good separation between matched and unmatched pairs"
  }
}
```

**Acceptance Criteria for COMMIT 3:**
- [ ] All diagnostics computed
- [ ] Analysis documented with interpretation
- [ ] Comparison to healthy ranges provided
- [ ] No embedding collapse detected

---

### Task 4: Qualitative Analysis

#### 4.1 Generate Retrieval Examples

Create `reports/phase5_qualitative_analysis.md` with:

1. **Success Examples (5+):**
   - Show image, query caption, and top-5 retrieved captions
   - Explain WHY the retrieval succeeded

2. **Failure Examples (5+):**
   - Show image, query caption, and top-5 retrieved captions
   - Explain WHY the retrieval failed
   - Categorize failure types (semantic, visual, ambiguous, etc.)

3. **Pattern Analysis:**
   - What types of images does the model handle well?
   - What types of images cause failures?
   - Are there systematic biases?

#### 4.2 Example Format

```markdown
## Success Example 1

**Image:** path/to/image.jpg
**Query Caption:** "A dog playing fetch in a park"
**Top-5 Retrieved:**
1. "A golden retriever catching a frisbee" (score: 0.82) ✓
2. "A dog running in a grassy field" (score: 0.78) ✓
3. "A person throwing a ball for a dog" (score: 0.75) ✓
4. "A dog jumping to catch a toy" (score: 0.71) ✓
5. "A puppy playing outdoors" (score: 0.68) ✓

**Why it succeeded:** The model correctly identified the key concepts:
- Dog (visual)
- Playing/fetch (action)
- Park/outdoors (setting)

## Failure Example 1

**Image:** path/to/image.jpg
**Query Caption:** "A red car on a highway"
**Top-5 Retrieved:**
1. "A truck driving on a road" (score: 0.72) ✗
2. "A vehicle on a freeway" (score: 0.69) ✗
3. "Cars in traffic" (score: 0.65) ✗
4. "A red automobile" (score: 0.62) ✓
5. "A highway with multiple lanes" (score: 0.58) ✗

**Why it failed:** Model focused on "vehicle/road" concepts but missed
the specific "red car" combination. Retrieved semantically similar but
incorrect images.
```

**Acceptance Criteria for COMMIT 4:**
- [ ] At least 5 success examples documented
- [ ] At least 5 failure examples documented
- [ ] Failure patterns analyzed and categorized
- [ ] Model strengths and weaknesses identified

---

### Task 5: Documentation Update

#### 5.1 Update ROADMAP.md

```markdown
## Phase 5 — Evaluation

**Goal:** Rigorously evaluate what the trained model actually learned,
quantitatively and qualitatively.

**Deliverables:**
- [x] Recall@1/5/10 for image→text and text→image on the test split
- [x] Embedding space diagnostics (collapse/uniformity checks)
- [x] Qualitative review: manually inspect 10+ retrieval
      successes AND failures, write down patterns observed

**Dependencies:** Phase 4 complete.

**Acceptance criteria:** Metrics reported on the held-out test split
with a stated comparison to random-chance baseline; qualitative
failure analysis documented, not just the numbers.

**Status:** complete

**Results:**
- **Test Recall@1:** X.XX% (X.Xx random baseline)
- **Test Recall@5:** X.XX% (X.Xx random baseline)
- **Test Recall@10:** X.XX% (X.Xx random baseline)
- **Val→Test Gap:** X.XX% (generalization gap)
- **Embedding Health:** [Healthy/Warning/Collapsed]

**Key Findings:**
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

**Documentation:**
- Test metrics: reports/phase5_test_metrics.json
- Embedding diagnostics: reports/phase5_embedding_diagnostics.json
- Qualitative analysis: reports/phase5_qualitative_analysis.md
- Final report: reports/phase5_final_report.md
```

#### 5.2 Update PROJECT_STATUS.md

Update current phase, status, and key metrics.

#### 5.3 Update TRAINING_LOG.md

Add Phase 5 evaluation results section.

#### 5.4 Create `reports/phase5_final_report.md`

Comprehensive final report with:
1. Executive Summary
2. Test Set Results
3. Embedding Analysis
4. Qualitative Findings
5. Comparison to Phase 4 (Val vs Test)
6. Recommendations for Phase 6
7. Lessons Learned

**Acceptance Criteria for COMMIT 5:**
- [ ] All documentation files updated
- [ ] Final report created
- [ ] Metrics consistent across all documents
- [ ] No conflicting information

---

### Task 6: Final Verification

#### 6.1 Run All Tests

```bash
python -m pytest tests/ -v --tb=short
```

All 222+ tests must pass.

#### 6.2 Verify Repository State

```bash
git status
git log --oneline -10
```

Repository must be clean.

#### 6.3 Generate Final Engineering Report

Create comprehensive summary including:
1. Technical Summary
2. Test Set Results
3. Embedding Diagnostics
4. Qualitative Analysis Summary
5. Val vs Test Comparison
6. Git Commits Created
7. Remaining Risks
8. Recommendations for Phase 6

**Acceptance Criteria for COMMIT 6:**
- [ ] All tests passing
- [ ] Repository clean
- [ ] Final report complete
- [ ] All documentation synchronized
- [ ] Phase 5 officially complete

---

## EXPECTED METRICS

Based on val Recall@10 of 20.26%, expected test metrics:

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| Test Recall@1 | 3.5-4.5% | Slightly below val (4.22%) |
| Test Recall@5 | 12-15% | Slightly below val (14.03%) |
| Test Recall@10 | 18-21% | Slightly below val (20.26%) |
| Val→Test Gap | 1-3% | Normal generalization gap |

**Red Flags:**
- Test Recall@10 < 15% (significant degradation)
- Embedding variance < 0.0001 (collapse)
- Mean pairwise distance < 0.3 (collapse)

---

## CRITICAL REMINDERS

1. **Never commit failing code** — run tests before committing
2. **Never batch unrelated work** — one logical unit per commit
3. **Always use conventional commits** — `feat:`, `fix:`, `docs:`, etc.
4. **Document everything** — metrics, analysis, decisions
5. **Compare to random baseline** — always state improvement factor
6. **Be honest about limitations** — document what doesn't work

---

## TOOLS AND SCRIPTS

### Existing Tools
- `scripts/evaluate_checkpoint.py` — Evaluate single checkpoint
- `src/vectormind/evaluation/memorization.py` — Recall@K functions
- `tests/evaluation/test_memorization.py` — Unit tests

### New Tools to Create
- `src/vectormind/evaluation/retrieval.py` — Comprehensive evaluation
- `scripts/evaluate_test_set.py` — Full test set evaluation
- `tests/evaluation/test_retrieval.py` — Unit tests

### Configuration
- `configs/data.yaml` — Dataset paths and splits
- `configs/model.yaml` — Model architecture
- `configs/training.yaml` — Training settings

### Checkpoints
- `checkpoints/train/best_model.pt` — Best checkpoint (Epoch 7)

---

## SUCCESS CRITERIA

Phase 5 is complete when:

1. ✅ Test Recall@10 clearly above random baseline (≥15%)
2. ✅ Both retrieval directions evaluated (image→text and text→image)
3. ✅ Embedding diagnostics show no collapse
4. ✅ Qualitative analysis documents 10+ examples
5. ✅ Failure patterns analyzed and documented
6. ✅ All tests passing
7. ✅ Documentation synchronized
8. ✅ All commits pushed to origin/main
9. ✅ Repository clean

---

*Generated: 2026-08-06*
*Phase 4 Status: COMPLETE*
*Phase 5 Status: READY TO BEGIN*
