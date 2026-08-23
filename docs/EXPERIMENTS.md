# EXPERIMENTS.md — VectorMind

Experiment tracking log. Three runs have been executed and are recorded
below. This file defines the template every future experiment must
follow, so the log stays consistent from the first real run onward.

Complements, not duplicates: Weights & Biases/TensorBoard
(TECH_STACK.md) captures continuous metrics (loss curves, gradient
norms) automatically. This file captures the discrete, human-written
summary of each experiment — the *decision* context a metrics
dashboard alone doesn't record.

---

## Template

Copy this block for every new experiment. Fill in every field — use
"N/A" explicitly rather than leaving a field blank, so it's clear the
field was considered, not forgotten.

```markdown
### Experiment [ID]

**Date:**
**Phase:** (which ROADMAP.md phase this experiment belongs to, e.g. "3.5 sanity check", "4 full run")
**Git commit SHA:**
**Config file(s) used:**

**Dataset:**
- Split(s) used:
- Subset size (if not full dataset, e.g. Phase 3.5's tiny subset):

**Model:**
- Image encoder config:
- Text encoder config:
- Shared embedding dim:

**Hyperparameters:**
- Batch size:
- Learning rate:
- Optimizer:
- Scheduler:
- Temperature init:
- Memory queue size:
- Gradient accumulation steps:
- Mixed precision (Y/N):

**Training Time:**
- Wall-clock duration:
- Hardware (should be the RTX 4050 unless explicitly noted otherwise):

**Evaluation Metrics:**
- Recall@1 (image→text):
- Recall@5 (image→text):
- Recall@10 (image→text):
- Recall@1 (text→image):
- Recall@5 (text→image):
- Recall@10 (text→image):
- Embedding variance/collapse check result:
- Random-chance baseline (for comparison):

**Observations:**
(What actually happened — surprising behavior, instability, anything
that didn't match expectations from ARCHITECTURE.md's design.)

**Conclusions:**
(What this experiment tells us. Did it pass/fail its acceptance
criteria per ROADMAP.md's phase definition?)

**Future Improvements:**
(What to try differently next time, and why — link to
PROJECT_MEMORY.md if this becomes a recorded decision.)
```

---

## Experiment Log

Newest last. Per CLAUDE.md §4 and PROJECT_RULES.md rule #11, the Phase
3.5 sanity check must appear — and must have passed — before any Phase 4
entry. It does.

Entries below were reconstructed on 2026-08-23 from TRAINING_LOG.md,
ROADMAP.md, reports/checkpoint_summary.json, and
reports/overfit/phase3_5_evaluation.json, because this log was never
filled in while the runs were happening. Fields that were not recorded
at the time are marked "not recorded" rather than guessed.

---

### Experiment 001 — Phase 3.5 tiny-subset overfit sanity check

**Date:** 2026-08-04
**Phase:** 3.5 sanity check
**Git commit SHA:** not recorded
**Config file(s) used:** `configs/overfit.yaml`, `configs/model.yaml`

**Dataset:**
- Split(s) used: fixed 100-image subset drawn from train (`data/processed/overfit_subset.json`)
- Subset size: 100 images / 500 image-caption pairs

**Model:**
- Image encoder config: ResNet-18-style CNN from scratch, output_dim 512
- Text encoder config: 6-layer Transformer from scratch, embed_dim 256, 8 heads
- Shared embedding dim: 256

**Hyperparameters:**
- Batch size: 32
- Learning rate: 3e-4
- Optimizer: AdamW (weight_decay 0.01)
- Scheduler: N/A (short run)
- Temperature init: log(1/0.07) → 14.29
- Memory queue size: 0 (disabled — extra negatives work against memorization)
- Gradient accumulation steps: 1
- Mixed precision (Y/N): Y

**Training Time:**
- Wall-clock duration: not recorded
- Hardware: RTX 4050 Laptop, 6GB VRAM

**Evaluation Metrics:** (evaluated on the memorized subset itself)
- Recall@1 (image→text): 100.0%
- Recall@5 / @10 (image→text): 100.0%
- Recall@1 (text→image): 100.0%
- Recall@5 / @10 (text→image): 100.0%
- Embedding variance/collapse check: 0.0039 — healthy; similarity separation 0.964 (matched 0.955, unmatched −0.009)
- Random-chance baseline: 1% @ R@1 over 100 images

**Observations:**
Temperature moved 14.29 → 15.33 over 30 epochs — a small, stable change.
The similarity separation of 0.964 is the most useful single number this
project produced: it shows the architecture and loss *can* produce a
well-spread embedding space. It is the correct baseline against which
Phase 4's separation of 0.094 should be read (docs/KNOWN_ISSUES.md §1).

**Conclusions:**
PASSED. Acceptance criterion (near-perfect retrieval on the memorized
subset) met with margin. Phase 4 unblocked.

**Future Improvements:**
Log separation and the norm of the mean embedding from this run onward,
so a Phase 4 regression shows up in the metrics rather than only in
retrospect.

---

### Experiment 002 — Phase 4 baseline, memory-queue ablation

**Date:** 2026-08-05
**Phase:** 4 full run
**Git commit SHA:** not recorded
**Config file(s) used:** `configs/training.yaml`, `configs/data.yaml`, `configs/model.yaml`

**Dataset:**
- Split(s) used: Flickr30k train (80%) / val (10%), split by image
- Subset size: full split

**Model:** as Experiment 001.

**Hyperparameters:**
- Batch size: 128 (below the profiled 256 ceiling, to leave headroom for validation and the queue)
- Learning rate: 1e-3
- Optimizer: AdamW (weight_decay 0.01)
- Scheduler: CosineAnnealing, T_max 20, eta_min 1e-6
- Temperature init: log(1/0.07) → 14.29
- Memory queue size: 0 for epochs 1–6, then 4096 for epochs 7–8
- Gradient accumulation steps: 1
- Mixed precision (Y/N): Y

**Training Time:**
- Wall-clock duration: ~45 minutes for 8 epochs, including the continuation run
- Hardware: RTX 4050 Laptop, 6GB VRAM

**Evaluation Metrics:** (val split; best checkpoint = epoch 7, step 7944)

| Metric | queue off (ep 6) | queue on (ep 7) |
|---|---|---|
| Recall@1 (image→text) | 3.46% | **4.22%** |
| Recall@5 (image→text) | 11.42% | **14.03%** |
| Recall@10 (image→text) | 17.12% | **20.23%** |
| Learned temperature | 18.6 | 55.2 |

- Embedding variance/collapse check: image 0.000746, text 0.000471 — **recorded as healthy at the time; that reading was wrong, see below**
- Random-chance baseline: ~1% @ R@1, ~10% @ R@10

**Observations:**
The memory queue is worth +3.1pp R@10 (+18.2% relative) — the clearest
positive result in the project, and it surfaced only because the first
six epochs accidentally ran with `--no-queue`.

Everything after epoch 7 degraded. R@10 fell to 13.62% by epoch 9 while
the learned logit scale ran 55 → 78 → 160 → 338 → 500+. Image variance
fell 83%, text variance 86%, mean pairwise distance 0.60 → 0.25.

This was logged at the time as "embedding collapse after epoch 7", with
epoch 7 itself judged healthy. The 2026-08-23 audit measured the epoch-7
embeddings directly and found mean off-diagonal image–image cosine
**0.810** and matched-vs-unmatched separation **0.094**, against 0.964 in
Experiment 001. Epoch 7 is not a healthy space — it is an earlier point
on the same collapse curve, caught before the retrieval metric noticed.

Two bugs surfaced during this run: gradient-norm logging computed the
norm *after* `zero_grad()` (always 0.0), and the checkpoint-resume path
mishandled a queue-size mismatch.

**Conclusions:**
Acceptance criterion met — a checkpoint exists with val R@10 (20.23%)
reproducibly above the ~10% random baseline. The *interpretation* of
embedding health recorded at the time did not survive re-measurement.

**Future Improvements:**
Clamp the logit scale at `ln(100)`, as CLIP does. Its absence is the
most likely single cause of the collapse trajectory, and it is a
one-line change. Re-run before drawing further conclusions from this
checkpoint.

---

### Experiment 003 — Learning-rate ablation

**Date:** 2026-08-05
**Phase:** 4 hyperparameter iteration
**Git commit SHA:** not recorded
**Config file(s) used:** `configs/training.yaml` with `optimizer.lr` overridden

**Dataset / Model:** as Experiment 002.

**Hyperparameters:** as Experiment 002 except **lr = 5e-4** (vs 1e-3).

**Training Time:**
- Wall-clock duration: not recorded (run to epoch 9)
- Hardware: RTX 4050 Laptop, 6GB VRAM

**Evaluation Metrics:** (val split)
- Recall@1 (image→text): 2.08% (vs 4.22% baseline)
- Recall@10 (image→text): 10.54% (vs 20.23% baseline)
- Learned temperature: 82 (vs 53 at the comparable baseline point)

**Observations:**
Halving the LR halved R@10. The stated hypothesis — that a lower LR
would improve stability — was rejected.

The analysis recorded at the time ("lower LR caused temperature to
increase faster, suggesting the model was overshooting") does not
follow: a lower LR producing a *faster*-growing logit scale points at
the logit scale being the unconstrained degree of freedom the optimizer
exploits when the representation itself is learning more slowly. That
reading is consistent with Experiment 002 and reinforces the clamp
recommendation.

**Conclusions:**
Hypothesis REJECTED. lr=1e-3 retained.

**Future Improvements:**
Re-run this ablation *after* clamping the logit scale. The current
result may be measuring the clamp's absence rather than the LR.

---

## Notes on Using This Log

- One entry per actual training run, not per config-file edit — if a
  run crashes before producing any evaluation metrics, still log it
  under "Observations" (e.g. "OOM at step N, see Phase 0.2 batch-size
  numbers for context") rather than silently discarding the attempt.
  Failed/aborted runs are part of the honest engineering record
  (PROJECT_CONTEXT.md §5's "the process is part of the deliverable").
- Reference the checkpoint metadata sidecar (ARCHITECTURE.md §12) by
  its path/commit SHA so an experiment log entry and its actual saved
  weights stay traceable to each other.
- If an experiment leads to an architecture or engineering decision
  (not just a hyperparameter tweak), record that decision in
  PROJECT_MEMORY.md as well — this file is the raw experimental
  record; PROJECT_MEMORY.md is the distilled "why we decided X" record.
