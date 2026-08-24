# EXPERIMENTS.md — VectorMind

Experiment tracking log. Seven runs have been executed and are recorded
below. Experiments 004-006 supersede 002's memory-queue conclusion —
see 006 for the controlled re-run that reverses it. This file defines the template every future experiment must
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

---

### Experiment 004 — Clamped re-run with embedding health monitoring

**Date:** 2026-08-23 / 2026-08-24
**Phase:** 4b re-run
**Git commit SHA:** see `logit-scale clamp` commit on `main`
**Config file(s) used:** `configs/training.yaml`, `configs/data.yaml`, `configs/model.yaml`

**Dataset:**
- Split(s) used: Flickr30k train (80%) / val (10%), split by image
- Subset size: full split — 127,130 train pairs, 15,890 val pairs

**Model:** as Experiment 001 — ResNet-18-style CNN, 6-layer Transformer, shared dim 256, 23.9M parameters.

**Hyperparameters:**
- Batch size: 128
- Learning rate: 1e-3
- Optimizer: AdamW (weight_decay 0.01, **excluding `log_temperature`**)
- Scheduler: CosineAnnealing, T_max 20, eta_min 1e-6
- Temperature init: log(1/0.07) → 14.29, **clamped at 100**
- Memory queue size: 4096, **inactive** (see Experiment 006)
- Gradient accumulation steps: 1
- Mixed precision (Y/N): Y

**Training Time:**
- Wall-clock duration: ~10 min/epoch, interrupted twice by memory failures
- Hardware: RTX 4050 Laptop, 6GB VRAM, 16GB system RAM shared with a desktop

**Evaluation Metrics:** (val split, epoch 7)
- Recall@1 (image→text): 3.81%
- Recall@5 (image→text): 13.09%
- Recall@10 (image→text): 19.63%
- Embedding variance/collapse check: separation **0.322**, mean image–image cosine 0.409, ‖mean embedding‖ 0.639
- Random-chance baseline: 0.031% @ R@1, 0.314% @ R@10 — see the note below

**Observations:**
Every metric improved monotonically for seven epochs, and the logit
scale rose only 14.3 → 18.6 rather than to 500+. The clamp never
engaged; bounding the parameter changed the trajectory the optimizer
took toward it.

Separation reached 0.322 against 0.094 for the Phase 4 checkpoint — the
same retrieval quality on a space 3.4× better separated. It is still
short of Experiment 001's 0.964, but that was an overfit of 100 images
and is an upper bound, not a target.

**On the baseline.** Every earlier entry in this project stated recall
"× random baseline" using 1% and 10%, which are correct for Experiment
001's 100-image subset and were carried over unchanged to the
3,179-image test split. Real chance for R@10 there is 0.314%, so the
multiples were understated by roughly 30×. See docs/KNOWN_ISSUES.md §1b.

**Conclusions:**
The clamp holds, and the collapse documented in Phase 4 does not recur.
Confounded with Experiment 006 — the queue was inactive throughout — so
the clamp cannot be credited alone.

**Future Improvements:**
Re-run with the queue disabled from epoch 1 to separate the two effects
cleanly, rather than resuming mid-run.

---

### Experiment 005 — Memory-queue warmup

**Date:** 2026-08-23
**Phase:** 4b
**Config file(s) used:** `configs/training.yaml` with `memory_queue.warmup_epochs`

**Hypothesis:** the queue fails early because its entries are stale
relative to a fast-moving encoder. Holding it inactive while it fills,
then activating it once the encoder has stabilized, should let it
deliver the extra negatives it was added for.

**Setup:** two arms. Queue active from step 1, versus inactive for six
epochs and then activated already full of recent embeddings.

**Evaluation Metrics:**

| Arm | Val R@10 | Separation |
|---|---|---|
| Active from step 1 | 0.35% after 2 epochs (chance) | 0.000 |
| Warmed up 6 epochs, then activated | 10.51% at epoch 7, down from 17.46% | 0.062, down from 0.329 |

**Observations:**
Activating from step 1 prevented the model from learning at all — 4096
stale negatives against 128 in-batch ones is a 32:1 ratio, and the
gradient is dominated by noise. Warmup fixed that, but the collapse
arrived in full at the first epoch after activation.

**Conclusions:**
Hypothesis REJECTED. Staleness is not a startup transient. Without a
momentum encoder the queue always holds embeddings from an encoder that
has since moved, so there is no point at which activating it is safe.

**Future Improvements:**
`warmup_epochs` is retained for anyone re-running this. The real fix is
a momentum encoder — see docs/FUTURE_IDEAS.md.

---

### Experiment 006 — Memory queue, controlled A/B

**Date:** 2026-08-24
**Phase:** 4b
**Config file(s) used:** `configs/training.yaml`, `--no-queue` for the baseline arm

**Hypothesis:** re-test Experiment 002's memory-queue result under a
controlled comparison — same starting checkpoint, one variable, and
embedding health measured alongside recall rather than recall alone.

**Setup:** both arms resume from `checkpoints/train/epoch_006.pt`. Only
possible after fixing `--no-queue`, which previously substituted a
size-1 stub queue that `load_checkpoint` rejected against a 4096-entry
checkpoint — which is why the original comparison was never controlled.

**Evaluation Metrics:** (val split, epoch 7, identical starting state)

| Metric | Queue active | Queue inactive |
|---|---|---|
| Train loss | 3.86 | **2.51** |
| Recall@1 (I→T) | 1.73% | **3.81%** |
| Recall@10 (I→T) | 10.51% | **19.63%** |
| Separation | 0.062 | **0.322** |
| Mean image–image cosine | 0.872 | **0.409** |
| Logit scale | 67.6 | **18.6** |

**Observations:**
The queue costs 87% of R@10 and 81% of separation from an identical
starting state. One epoch of it undid six epochs of improvement.

The mechanism explains Phase 4's "temperature overgrowth" as a symptom
rather than a cause. Minimising contrastive loss against thousands of
mismatched stale negatives is easier by sharpening the similarity
distribution than by improving the representation, and an unbounded
logit scale is the cheapest way to sharpen. The scale runs away, the
space collapses, and recall follows a few epochs later.

**Conclusions:**
Experiment 002's conclusion is REVERSED. The memory queue is not a
mitigation at this scale; it is the primary cause of the collapse this
project spent Phase 4 documenting. Training proceeds with in-batch
negatives only.

This is the most useful negative result the project has produced, and it
was only visible because embedding health is now measured beside recall.
Recall alone lags a collapse by several epochs, which is exactly how the
original conclusion survived.

**Future Improvements:**
Implement a momentum encoder and re-run this A/B a third time. That is
the difference between "a queue does not work here" and "a queue does
not work here *without the mechanism that makes it work elsewhere*",
and only the second statement is actually about MoCo.

---

### Experiment 007 — Does training past epoch 12 help?

**Date:** 2026-08-24
**Phase:** 4b
**Config file(s) used:** `configs/training.yaml`, `--no-queue`

**Hypothesis:** the epoch-12 checkpoint was reached by a run that had
been interrupted, not by convergence, and validation was still climbing
when it stopped. More epochs should improve it further.

**Setup:** resume from `best_model.pt` (epoch 12) and continue toward
the configured 20 epochs, with the queue still disabled.

**Evaluation Metrics:**

| Epoch | Train loss | Val R@10 | Separation | ‖mean embedding‖ |
|---|---|---|---|---|
| 12 (start) | 1.70 | **29.30%** | 0.356 | 0.613 |
| 13 | 1.57 | no improvement | 0.356 | 0.568 |
| 14 | 1.46 | no improvement | 0.344 | 0.568 |

**Observations:**
Training loss fell by 14% across the two epochs while validation R@10
did not improve at all. Separation held and then declined slightly.

That combination — training loss falling, validation flat — is the
standard signature of a model that has started fitting its training
split. At 25,426 training images and 24M parameters it is unsurprising
that this arrives around epoch 12.

Worth noting what did *not* happen: no collapse. The logit scale sat
around 26 against its ceiling of 100, and separation stayed near 0.35.
The failure mode of the original Phase 4 run is genuinely gone — this is
ordinary convergence, which is a much better problem to have.

**Conclusions:**
Hypothesis REJECTED. Epoch 12 is the converged checkpoint and is the
shipped model. Early stopping at patience 5 would have selected the same
weights at epoch 17.

The practical value of this result is negative-but-useful: it closes off
"train longer" as an option, so any further accuracy work has to be
architectural.

**Future Improvements:**
Three candidates, in order of expected value:
1. A momentum encoder, making the memory queue usable rather than harmful (Experiment 006, FUTURE_IDEAS.md).
2. A uniformity term targeting ‖mean embedding‖ 0.621 — the one health threshold the model still fails.
3. More data. Flickr30k is 31k images; the constraint is the project's premise rather than an oversight.
