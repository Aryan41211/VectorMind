# EXPERIMENTS.md — VectorMind

Experiment tracking log. **No entries exist yet** — no training has
occurred in this project as of this document's creation (see
PROJECT_STATUS.md: Phase 2/3 have not started). This file defines the
template every future experiment must follow, so the log stays
consistent from the very first real run onward.

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

_(empty — no experiments have been run yet)_

The first entry in this section should be the **Phase 3.5 tiny-subset
overfit sanity check** (ROADMAP.md), since CLAUDE.md §4 and
PROJECT_RULES.md rule #11 both require it to pass and be recorded
*before* any Phase 4 full run. Do not add a Phase 4 entry to this log
without a preceding, passing Phase 3.5 entry directly above it.

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
