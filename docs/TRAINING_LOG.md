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
| 7 | 3.72 | 4.22% | 14.03% | 20.23% | 55.2 | 4096 | **BEST** |
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
1. ~~**Memory queue fix improved Recall@10 by 3.1 percentage points** (17.12% → 20.23%)~~ — **retracted, see Experiment 5.** The controlled re-run shows the queue *halves* R@10 from the same checkpoint.
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
**Status:** COMPLETED — **RETRACTED 2026-08-24, see Experiment 5**
**Hypothesis:** Enabling memory queue will improve Recall@10 by providing
more negative samples for contrastive learning.
**Results as originally recorded:**
- Baseline (queue disabled): Recall@10 = 17.12%
- With queue (size=4096): Recall@10 = 20.23%
- **Improvement:** +3.1 percentage points (+18.2% relative)
- **Conclusion at the time:** Hypothesis CONFIRMED

**Why this conclusion does not hold.** Three problems, none visible at
the time:

1. **One epoch, measured before the damage showed.** 20.23% is epoch 7,
   the first epoch after activation. Epochs 9-15 fell to 13.62%. That
   decline was logged separately as "embedding collapse from temperature
   overgrowth" and treated as an unrelated failure. It was not unrelated
   — it was this experiment's own effect arriving.
2. **Never a controlled A/B.** The two arms did not share a starting
   point: `--no-queue` substituted a size-1 stub queue, and
   `load_checkpoint` rejects a size-1 queue against a 4096-entry
   checkpoint, so the baseline arm could only ever run from scratch.
   The comparison was epoch 6 of one run against epoch 7 of another.
3. **No embedding-health metric existed.** Recall@10 was the only signal,
   and it is exactly the signal that lags a collapse.

All three are fixed. Experiment 5 is the controlled re-run.

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

### Best Checkpoint (Verified 2026-08-07)
- **File:** checkpoints/train/best_model.pt
- **Epoch:** 7
- **Step:** 7944
- **Val Recall@1:** 4.22% (134x chance)
- **Val Recall@5:** 14.00%
- **Val Recall@10:** 20.23% (64x chance)
- **Temperature:** 55.24 (learned from 14.29)
- **Test Recall@1 (I2T):** 4.62% (147x chance) [CORRECTED]
- **Test Recall@5 (I2T):** 13.43% (85x chance) [CORRECTED]
- **Test Recall@10 (I2T):** 19.63% (62x chance) [CORRECTED]
- **Test Recall@10 (T2I):** 15.09% (48x chance) [CORRECTED]
- **Val→Test Gap (R@10):** -0.60% (reasonable generalization) [CORRECTED]

### Training Summary
- **Total epochs:** 8 (6 baseline + 2 with queue)
- **Total time:** ~45 minutes
- **Final loss:** 3.72
- **Memory Queue:** Enabled (size=4096)
- **Embedding Variance:** Image 0.000746, Text 0.000471 (healthy)
- **Pairwise Distances:** Image 0.6054, Text 0.4765 (healthy)

### Phase 5 Evaluation Summary (Corrected 2026-08-07)
- **Val→Test Gap (R@10):** -0.60% (reasonable generalization)
- **Embedding Health:** HEALTHY (no collapse)
- **Failure Rate:** 80.37%
- **Main Failure Patterns:** Action ambiguity (35%), Object specificity (25%)

### Bug Fix (2026-08-07)
- **Test evaluation bug:** `evaluate_test_set.py` had `_, eval_loader, _` destructuring
  that always used val_loader regardless of `--split` argument. Fixed to properly select
  test_loader when `--split test`. Re-run confirmed real test R@10 = 19.63% (vs val 20.23%).
- **Data helpers import bug:** `_data_helpers.py` imported `datasets` at function entry
  before cache check, causing ImportError even when cache existed. Fixed to lazy-import
  only when download is needed.

### Verified Issues
1. **Gradient norm logging bug (FIXED):** Was computing norm after zero_grad(), always 0.0
2. **Temperature discrepancy:** Reported 53.51, actual 55.24
3. **Pairwise distances:** Updated to verified values (0.60/0.48)
4. **Test eval bug (FIXED 2026-08-07):** Always evaluated on val split due to destructuring error

### Lessons Learned
1. ~~**Memory queue is critical:** Enabling queue improved Recall@10 by 18.2%~~ — **retracted.** The queue is the cause of the collapse this log attributed to temperature overgrowth. See Experiment 5.
2. **Lower LR hurts performance:** 5e-4 LR caused Recall@10 to drop to 10.54%
3. **Temperature learning is important:** Model learned to sharpen similarity distribution
4. **Embedding variance monitoring is essential:** Caught that variance remained healthy
5. **Checkpoint resume works correctly:** Successfully resumed from Epoch 6 to Epoch 8
6. **Gradient norm must be computed before zero_grad():** Common AMP training bug
7. **Val→Test gap reasonable:** ~0.6pp gap indicates decent generalization
8. **Action recognition is weak:** Model struggles with fine-grained actions
9. **Test eval destructuring bug (FIXED):** `_, loader, _` always used val_loader — always verify which split is actually being evaluated by printing filenames

---

*Last updated: 2026-08-05*

---

## Phase 4b — Clamped Re-run (2026-08-23/24)

**Run ID:** clamped_20260823
**Why:** the Phase 4 checkpoint shipped with a matched-vs-unmatched
separation of 0.094 against 0.964 for the Phase 3.5 overfit — a
collapsed embedding space that the reports of the time called HEALTHY
(docs/KNOWN_ISSUES.md §1). Changes under test: the logit scale is
clamped at CLIP's 100, `log_temperature` is excluded from weight decay,
and embedding health is logged beside Recall@K every epoch.

### Training progress (queue inactive throughout)

| Epoch | Loss | Val R@1 | Val R@5 | Val R@10 | Separation | mean cos | ‖mean‖ | Logit scale |
|---|---|---|---|---|---|---|---|---|
| 1 | 4.51 | 0.41% | 1.66% | 2.99% | 0.102 | 0.792 | 0.890 | 14.3 |
| 2 | 3.99 | — | — | 5.03% | 0.166 | 0.682 | 0.826 | ~15 |
| 3 | 3.61 | — | — | 9.60% | 0.244 | 0.538 | 0.734 | ~15 |
| 4 | 3.25 | — | — | 11.26% | 0.263 | 0.511 | 0.715 | ~16 |
| 6 | 2.72 | 3.43% | 11.26% | 17.46% | 0.329 | 0.388 | 0.623 | 17.4 |
| 7 | 2.51 | 3.81% | 13.09% | **19.63%** | 0.322 | 0.409 | 0.639 | 18.6 |

Every metric improves monotonically, and the logit scale stays near its
initialization instead of running to 500+. Compare the Phase 4 table
above, where R@10 oscillated while the scale climbed past 300.

### Interruptions

The run survived three stops, none of them modelling failures. They are
recorded because a 6GB laptop GPU that also drives a display is the
project's real operating environment, and pretending training is
uninterrupted would misrepresent it.

| Epoch | Cause | Resolution |
|---|---|---|
| 5 | `CUDA error: out of memory` at ~4.6GB of 6GB — a neighbouring desktop process took the headroom | `src/vectormind/training/oom.py`: release the allocator cache and retry the step |
| 7 | `CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED` — **system** RAM, not VRAM, with 4.5GB free | Detection broadened to host failures; `num_workers`/`prefetch_factor` 4→2, cutting pinned buffers from 1.15GB to 0.29GB |
| 7 | Deliberate stop after the queue A/B below | — |

---

### Experiment 3: Logit-scale clamp

**Hypothesis:** clamping the learnable logit scale at 100 prevents the
Phase 4 collapse.
**Result:** CONFIRMED. Across 7 epochs the scale rose 14.3 → 18.6 and
never approached the ceiling; separation rose 0.102 → 0.322 rather than
falling to 0.094. The clamp never actually engaged — bounding the
parameter changed the trajectory the optimizer took toward it.
**Caveat:** this is confounded with Experiment 5 (the queue was inactive
for these epochs). The clamp is retained on the CLIP precedent and
because it costs nothing; the queue removal is doing more of the work.

### Experiment 4: Memory-queue warmup

**Hypothesis:** the queue fails early because its entries are stale
relative to a fast-moving encoder, so holding it inactive until the
encoder stabilizes should let it help.
**Result:** REJECTED.
- Queue active from step 1: val R@10 stuck at 0.35% (chance) after 2 epochs, separation 0.000.
- Warmed up for 6 epochs, then activated: collapse arrived in full at the first epoch after activation (below).

Warmup delays the damage rather than preventing it. Staleness is not a
startup transient — without a momentum encoder the queue always holds
embeddings from an encoder that has since moved.

### Experiment 5: Memory queue, controlled A/B

**Hypothesis:** re-test Experiment 1 properly — same starting
checkpoint, one variable, embedding health measured alongside recall.
**Setup:** both arms resume from `epoch_006.pt`; the only difference is
whether the queue serves negatives. Made possible by fixing `--no-queue`,
which previously substituted a size-1 stub that `load_checkpoint`
rejected against a real checkpoint.

| Epoch 7 from `epoch_006.pt` | Queue active | Queue inactive |
|---|---|---|
| Train loss | 3.86 | **2.51** |
| Val R@1 | 1.73% | **3.81%** |
| Val R@10 | 10.51% | **19.63%** |
| Separation | 0.062 | **0.322** |
| Mean image–image cosine | 0.872 | **0.409** |
| Logit scale | 67.6 | **18.6** |

**Result:** Experiment 1's conclusion is REVERSED. The queue costs 87%
of R@10 and 81% of separation from an identical starting state.

**Mechanism.** The queue is what drives the logit scale up, which Phase 4
logged as a separate "temperature overgrowth" problem. Minimising loss
against 4096 mismatched stale negatives is easier by sharpening the
similarity distribution than by improving the representation, and an
unbounded scale is the cheapest way to sharpen. Collapse follows.

**Decision:** train with in-batch negatives only. A momentum encoder is
the correct way to make a queue work at this scale and is recorded in
docs/FUTURE_IDEAS.md.

### Bugs found during this run

1. **A resumed run overwrote a better checkpoint.** `best_val_recall10` reset to 0.0 on resume, so the first epoch always won the comparison — a 17.46% checkpoint was replaced by a 10.51% one. `save_checkpoint` now records the score that earned it.
2. **`--no-queue` could not resume.** The size-1 stub was rejected by `load_checkpoint`, so the baseline arm could only start from scratch. This is why Experiment 1 was never a clean A/B.
3. **`expandable_segments` is unsupported on Windows** — it was being passed for nothing.

---

## Phase 4b — Convergence (epochs 12–14)

Training was resumed past epoch 12 to test whether the trajectory still
had room. It did not.

| Epoch | Train loss | Val R@10 | Separation | ‖mean‖ |
|---|---|---|---|---|
| 11 | 1.83 | 25.77% | 0.351 | 0.588 |
| 12 | 1.70 | **29.30%** | 0.356 | 0.613 |
| 13 | 1.57 | no improvement | 0.356 | 0.568 |
| 14 | 1.46 | no improvement | 0.344 | 0.568 |

**Training loss fell 1.70 → 1.46 while validation R@10 stopped moving.**
That divergence is the definition of the model beginning to fit the
training split rather than learn transferable structure. Separation
flattened and then fell slightly, which says the same thing from the
representation side.

**Conclusion: epoch 12 is the converged checkpoint.** It is the shipped
model. Early stopping (patience 5) would have terminated the run at
epoch 17 and selected the same weights.

**What this rules out.** "Train longer" was the cheapest remaining lever
on accuracy and it is now exhausted. Further gains need a change of
kind, not degree:

- a momentum encoder, which would make the memory queue usable instead of collapsing the space (docs/FUTURE_IDEAS.md)
- ~~a uniformity term targeting the residual anisotropy~~ — **done, and it shipped** (2026-08-25). At weight 0.2 it took ‖mean image embedding‖ 0.577 → 0.154 and separation 0.356 → 0.490, moving the grade to HEALTHY, for -0.13pp image→text R@10 and **+1.30pp text→image**. So this one was not a change of kind after all: the accuracy lever was exhausted, the *representation* lever was not. EXPERIMENTS.md 009, KNOWN_ISSUES.md §12
- more data, or a larger encoder, both bounded by the 6GB VRAM constraint that defines the project

**Run interruptions.** This run was stopped six times across two days,
by memory pressure and by manual interruption, on a laptop whose GPU
also drives the display. It survived because `--resume` restores the
best-so-far score from the checkpoint rather than restarting the
comparison at zero — without that fix, any one of those six stops would
have overwritten the best weights with whatever the next epoch produced.
