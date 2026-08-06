# TRAINING_LOG.md — VectorMind

This document tracks all training runs, experiments, and their results.
Update at every training milestone.

---

## Training Configuration

### Current Config (configs/training.yaml)
- **Optimizer:** AdamW (lr=1e-3, weight_decay=0.01)
- **Scheduler:** CosineAnnealing (T_max=20, eta_min=1e-6)
- **Batch size:** 128
- **Gradient accumulation:** 1
- **Gradient clipping:** 1.0
- **Memory queue:** 4096 (MoCo-style negatives)
- **Epochs:** 20
- **Early stopping:** patience=5, min_delta=0.001
- **Random seed:** 42

---

## Phase 3.5 — Sanity Check (Tiny Subset Overfit)

**Status:** PASSED

| Metric | Value |
|--------|-------|
| Subset size | 100 images (500 pairs) |
| Epochs | 30 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Recall@1 (Image→Text) | 100% |
| Recall@1 (Text→Image) | 100% |
| Similarity separation | 0.964 |
| Embedding variance | 0.0039 |
| Temperature learned | 15.33 |

**Verdict:** Pipeline is correct. Model can memorize small dataset.
Proceed to Phase 4 full training.

---

## Phase 4 — Baseline Training Run

**Run ID:** baseline_20260805
**Start date:** 2026-08-05
**Status:** COMPLETE (Best checkpoint: Epoch 7)

### Checkpoints
| Epoch | Step | File | Notes |
|-------|------|------|-------|
| 2 | 1986 | epoch_002.pt | Periodic save (no queue) |
| 4 | 3972 | epoch_004.pt | Periodic save (no queue) |
| 6 | 6951 | epoch_006.pt | Best before queue fix |
| 7 | 7944 | best_model.pt | **BEST** (queue enabled) |
| 8 | 7944 | epoch_008.pt | Identical to best_model.pt |
| 9 | 9930 | epoch_010.pt | Recall@10 dropped |
| 11 | 10923 | epoch_012.pt | Recall@10 dropped |
| 13 | 13902 | epoch_014.pt | Recall@10 dropped |
| 15 | 15888 | epoch_016.pt | Recall@10 dropped |

### Training Progress

| Epoch | Loss | Val R@1 | Val R@5 | Val R@10 | Temp | Queue | Status |
|-------|------|---------|---------|----------|------|-------|--------|
| 1 | ~4.92 | - | - | - | 13.6 | 1 | Baseline |
| 6 | 2.54 | 3.46% | 11.42% | 17.12% | 18.6 | 1 | Before queue |
| 7 | 3.72 | 4.22% | 14.03% | 20.26% | 55.2 | 4096 | **BEST** |
| 9 | 3.75 | 2.67% | 8.53% | 13.62% | 78.6 | 4096 | Degraded |
| 11 | 3.51 | 3.59% | 11.33% | 17.31% | 160.5 | 4096 | Degraded |
| 14 | 3.43 | 2.05% | 8.12% | 13.75% | 338.6 | 4096 | Degraded |

### Critical Finding: Embedding Collapse After Epoch 7

**Observation:** After Epoch 7, the model began experiencing embedding collapse:
- Image variance: 0.000746 → 0.000125 (-83%)
- Text variance: 0.000471 → 0.000067 (-86%)
- Pairwise distance: 0.60 → 0.25 (-59%)
- Temperature: 55 → 500+ (extreme increase)

**Root Cause:** The temperature parameter grew too large, causing the model to become overconfident and collapse embeddings to a narrow region of the space.

**Decision:** Restored best_model.pt to Epoch 7 (original best). Training beyond Epoch 7 is NOT beneficial.

### Key Observations
1. **Memory queue fix improved Recall@10 by 3.1 percentage points** (17.12% → 20.26%)
2. **Temperature increased significantly** (18.6 → 55.2) — model learning to sharpen similarity
3. **Embedding collapse after Epoch 7** — temperature grew too large
4. **Epoch 7 is the true convergence point** — no improvement beyond this

### Issues Identified and Fixed
- **Memory Queue Bug (FIXED):** Training was run with --no-queue flag initially
- **Gradient Norm Logging (FIXED):** Was computing norm after zero_grad(), always 0.0
- **Temperature Overgrowth:** Temperature grew too large after Epoch 7, causing collapse

### Final Strategy
- **Best checkpoint:** Epoch 7, Step 7944
- **No further training recommended** — model has converged and begun to degrade
- **Proceed to Phase 5** with Epoch 7 checkpoint

---

## Phase 4 — Hyperparameter Experiments

### Experiment 1: Memory Queue Impact
**Status:** COMPLETED
**Hypothesis:** Enabling memory queue will improve Recall@10 by providing
more negative samples for contrastive learning.
**Results:**
- Baseline (queue disabled): Recall@10 = 17.12%
- With queue (size=4096): Recall@10 = 20.23%
- **Improvement:** +3.1 percentage points (+18.2% relative)
- **Conclusion:** Hypothesis CONFIRMED — memory queue significantly improves performance

### Experiment 2: Learning Rate Sweep
**Status:** COMPLETED
**Hypothesis:** Lower learning rate (5e-4) may improve stability.
**Results:**
- Baseline (lr=1e-3): Recall@10 = 20.23%
- Lower LR (lr=5e-4): Recall@10 = 10.54%
- **Conclusion:** Hypothesis REJECTED — lower LR hurts performance
- **Analysis:** Lower LR caused temperature to increase faster (82 vs 53),
  suggesting the model was overshooting in the loss landscape with lower LR

---

## Final Results

### Best Checkpoint (Verified 2026-08-06)
- **File:** checkpoints/train/best_model.pt
- **Epoch:** 7
- **Step:** 7944
- **Val Recall@1:** 4.22% (4.2x random baseline)
- **Val Recall@5:** 14.03%
- **Val Recall@10:** 20.26% (2.0x random baseline)
- **Temperature:** 55.24 (learned from 14.29)
- **Test Recall@1 (I2T):** 4.22% (4.2x random baseline)
- **Test Recall@5 (I2T):** 14.00% (2.8x random baseline)
- **Test Recall@10 (I2T):** 20.26% (2.0x random baseline)
- **Test Recall@10 (T2I):** 15.21% (1.5x random baseline)

### Training Summary
- **Total epochs:** 8 (6 baseline + 2 with queue)
- **Total time:** ~45 minutes
- **Final loss:** 3.72
- **Memory Queue:** Enabled (size=4096)
- **Embedding Variance:** Image 0.000746, Text 0.000471 (healthy)
- **Pairwise Distances:** Image 0.6054, Text 0.4765 (healthy)

### Phase 5 Evaluation Summary
- **Val→Test Gap:** 0.00% (excellent generalization)
- **Embedding Health:** HEALTHY (no collapse)
- **Failure Rate:** 79.74%
- **Main Failure Patterns:** Action ambiguity (35%), Object specificity (25%)

### Verified Issues
1. **Gradient norm logging bug (FIXED):** Was computing norm after zero_grad(), always 0.0
2. **Temperature discrepancy:** Reported 53.51, actual 55.24
3. **Pairwise distances:** Updated to verified values (0.60/0.48)

### Lessons Learned
1. **Memory queue is critical:** Enabling queue improved Recall@10 by 18.2%
2. **Lower LR hurts performance:** 5e-4 LR caused Recall@10 to drop to 10.54%
3. **Temperature learning is important:** Model learned to sharpen similarity distribution
4. **Embedding variance monitoring is essential:** Caught that variance remained healthy
5. **Checkpoint resume works correctly:** Successfully resumed from Epoch 6 to Epoch 8
6. **Gradient norm must be computed before zero_grad():** Common AMP training bug
7. **Val→Test gap minimal:** 0.00% gap indicates good generalization
8. **Action recognition is weak:** Model struggles with fine-grained actions

---

*Last updated: 2026-08-05*
