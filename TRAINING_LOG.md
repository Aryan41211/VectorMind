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
**Status:** In Progress (paused at Epoch 6)

### Checkpoints
| Epoch | Step | File | Notes |
|-------|------|------|-------|
| 2 | 1986 | epoch_002.pt | Periodic save |
| 4 | 3972 | epoch_004.pt | Periodic save |
| 6 | 6951 | best_model.pt | Best val Recall@10 |

### Training Progress

| Epoch | Loss | Val R@1 | Val R@5 | Val R@10 | Temp | LR |
|-------|------|---------|---------|----------|------|-----|
| 1 | ~4.92 | - | - | - | 13.6 | 1e-3 |
| 6 | 2.54 | 3.46% | 11.42% | 17.12% | 18.6 | 7.3e-4 |

### Key Observations
1. Loss decreased 50% (4.92 → 2.54)
2. Recall@10 at 17.12% (1.7x random baseline)
3. Embedding variance healthy (0.0023)
4. Memory queue not being used (size=1)
5. Gradient norm logging shows 0.0 (investigation needed)

### Issues Identified
- **Memory Queue Bug:** Queue size remains at 1 despite enqueue calls
- **Gradient Logging:** All values are 0.000000
- **Embedding Std Logging:** All values are 0.000000

### Resume Strategy
- **Decision:** Resume from Epoch 6 (best checkpoint)
- **Fix:** Memory queue bug before resuming
- **Budget:** 14 more epochs (total 20)
- **Convergence criteria:** Early stopping (patience=5)

---

## Phase 4 — Hyperparameter Experiments

### Experiment 1: Memory Queue Impact
**Status:** Pending
**Hypothesis:** Enabling memory queue will improve Recall@10 by providing
more negative samples for contrastive learning.
**Plan:**
1. Fix memory queue bug
2. Train with queue_size=4096 for 10 epochs
3. Compare against baseline (queue disabled)

### Experiment 2: Learning Rate Sweep
**Status:** Pending
**Hypothesis:** Lower learning rate (5e-4) may improve stability.
**Plan:**
1. Resume from best checkpoint
2. Reduce lr to 5e-4
3. Train for 5 epochs
4. Compare against baseline

---

## Final Results (To Be Completed)

### Best Checkpoint
- **File:** TBD
- **Epoch:** TBD
- **Val Recall@1:** TBD
- **Val Recall@5:** TBD
- **Val Recall@10:** TBD
- **Test Recall@1:** TBD
- **Test Recall@5:** TBD
- **Test Recall@10:** TBD

### Training Summary
- **Total epochs:** TBD
- **Total time:** TBD
- **Final loss:** TBD
- **Temperature:** TBD

### Lessons Learned
- TBD (to be filled after training completion)

---

*Last updated: 2026-08-05*
