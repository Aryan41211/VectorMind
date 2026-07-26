# ROADMAP.md — VectorMind

Living project-tracking document. Update status and notes at every
milestone. This is the single source of truth for project state.

---

## Constraints

- **GPU:** RTX 4050 laptop, 6GB VRAM — governs batch size strategy;
  contrastive learning quality is highly sensitive to negative sample
  count, so in-batch-negative limitations must be compensated for
  architecturally (see ARCHITECTURE.md).
- **Framework:** PyTorch
- **Dataset:** Flickr30k (public; ~30k images, 5 captions each) —
  public data only, no scraped/private data
- **Model:** Trained from scratch — no pretrained CLIP/OpenCLIP
  weights loaded anywhere in the pipeline# ROADMAP.md — VectorMind

Living project-tracking document. Update status and notes at every
milestone. This is the single source of truth for project state.

---

## Constraints

- **GPU:** RTX 4050 laptop, 6GB VRAM — governs batch size strategy;
  contrastive learning quality is highly sensitive to negative sample
  count, so in-batch-negative limitations must be compensated for
  architecturally (see ARCHITECTURE.md, §6).
- **Framework:** PyTorch
- **Dataset:** Flickr30k (public; ~30k images, 5 captions each) —
  public data only, no scraped/private data
- **Model:** Trained from scratch — no pretrained CLIP/OpenCLIP
  weights loaded anywhere in the pipeline

## Realistic Success Definition

This project will not, and is not trying to, match published CLIP
numbers. Success looks like: a model that demonstrably learns
non-trivial cross-modal retrieval on Flickr30k (Recall@10 clearly
above random chance, with sensible qualitative results), built
through a pipeline that is correct, debuggable, and well-documented
end to end. The engineering process is as much the deliverable as
the final metric.

## Known Risks (named upfront, not discovered mid-training)

| Risk | Likely cause | Mitigation / where it's addressed |
|---|---|---|
| Embedding collapse (all embeddings converge to near-identical vectors, loss looks fine but retrieval is random) | Batch too small, learning rate too high, no negative diversity | Phase 3.5 sanity check catches this before a full run; embedding variance logged from the first run onward |
| OOM on 6GB VRAM | Batch size / model size too large for available memory | Phase 0.2 profiling; AMP + gradient accumulation + memory queue (ARCHITECTURE.md §6) |
| Loss decreases but retrieval doesn't improve | Overfitting to training pairs, or metric/eval bug | Held-out val set from Phase 1; qualitative spot-checks in Phase 5, not just the number |
| Training is unstable / diverges | Temperature or LR misconfigured | Phase 3.5 tiny-scale run before any long run; log temperature value every step |
| Tokenizer/text pipeline bug silently produces garbage batches | Off-by-one in padding/truncation, wrong special tokens | Phase 1 sanity checks include manually inspecting decoded batches, not just shapes |

---

## Phase 0 — Project Setup & Architecture Decisions

**Goal:** Establish a reproducible project skeleton and lock in all
core architecture decisions before any model/data code is written.

**Sub-phases:**
- **0.1 Environment & repo setup** — structure, venv, PyTorch+CUDA
  verified, `.gitignore`, core docs committed.
- **0.2 VRAM profiling** — empirically determine the max feasible
  batch size on the RTX 4050 with the planned encoder sizes, under
  AMP. Record the real number in ARCHITECTURE.md (replacing the
  placeholder). This determines memory-queue size in Phase 2.

**Deliverables:**
- [x] Repository structure created
- [ ] Virtual environment + PyTorch (GPU) verified working
- [ ] `CLAUDE.md`, `ROADMAP.md`, `ARCHITECTURE.md` committed
- [ ] Empirical batch-size ceiling measured and documented

**Dependencies:** None — this is the foundation.

**Acceptance criteria:** Repo builds; CUDA verified on RTX 4050; a
documented, measured (not guessed) max batch size exists in
ARCHITECTURE.md.

**Status:** in progress

---

## Phase 1 — Data Pipeline

**Goal:** Acquire, clean, and structure Flickr30k into a
training-ready, verified-correct paired image-text dataset.

**Deliverables:**
- [ ] Flickr30k downloaded and verified (checksums / count matches
      expected)
- [ ] Image transforms + tokenization pipeline
- [ ] Paired `Dataset`/`DataLoader` implementation
- [ ] Train/val/test split with zero image leakage across splits
- [ ] Sanity checks: visualize N batches with decoded captions
      side-by-side with images (manual inspection, not just shape
      checks) to catch pairing/tokenizer bugs

**Dependencies:** Phase 0 complete.

**Acceptance criteria:** A `DataLoader` yields correctly paired,
correctly shaped tensors; manual inspection of 10+ samples confirms
image-caption pairing is correct and captions decode to sensible text.

**Status:** not started

---

## Phase 2 — Model Architecture

**Goal:** Implement the dual-encoder model and confirm it runs
end-to-end, before any real training.

**Deliverables:**
- [ ] Image encoder (small CNN, from scratch)
- [ ] Text encoder (small Transformer, from scratch)
- [ ] Projection heads + L2 normalization into shared embedding space
- [ ] Learnable temperature parameter
- [ ] Forward-pass smoke test: one batch through the full model,
      correct output shapes, no NaNs

**Dependencies:** Phase 0 architecture decisions + measured batch
size finalized.

**Acceptance criteria:** Forward pass runs end-to-end on a real batch
from the Phase 1 pipeline (not synthetic data), producing
L2-normalized embeddings of the agreed shared dimension for both
modalities, with no NaN/Inf values.

**Status:** not started

---

## Phase 3 — Training Infrastructure

**Goal:** Build the training loop machinery itself — separate from
actually training a real model (that's Phase 3.5/4). This phase is
"does the machine run," not "does the model learn."

**Deliverables:**
- [ ] Symmetric InfoNCE loss implementation (unit tested against a
      hand-computed small example)
- [ ] MoCo-style memory queue implementation (unit tested for
      correct enqueue/dequeue behavior)
- [ ] Mixed precision + gradient accumulation wired in
- [ ] Checkpointing (save/resume, including optimizer state)
- [ ] Logging to W&B/TensorBoard: loss, temperature, embedding norm,
      embedding variance, GPU memory usage

**Dependencies:** Phases 1 and 2 complete.

**Acceptance criteria:** Loss function has a passing unit test with
a known expected value; a training loop can run for a few steps on
real data without crashing, and a checkpoint can be saved and
successfully reloaded to resume.

**Status:** not started

---

## Phase 3.5 — Sanity Check: Overfit a Tiny Subset

**Goal:** Prove the entire pipeline can actually learn something
before spending real compute on a full run. This is the single most
important risk-reduction step in the project and is not optional.

**Deliverables:**
- [ ] Train on a tiny fixed subset (e.g. 50–100 image-caption pairs)
      for enough steps to memorize it
- [ ] Confirm near-perfect Recall@1 on that same tiny subset
- [ ] Confirm embedding variance stays healthy (not collapsing to a
      single point) throughout

**Dependencies:** Phase 3 complete.

**Acceptance criteria:** The model achieves near-perfect retrieval on
the tiny memorized subset. If it cannot, training does not proceed to
Phase 4 — debug here first, since nothing downstream can be trusted
otherwise.

**Status:** not started

---

## Phase 4 — Full Training Run(s)

**Goal:** Train the real model on the full Flickr30k training split,
iterating on hyperparameters as needed.

**Deliverables:**
- [ ] At least one full training run to convergence (loss plateaued,
      val metrics stopped improving)
- [ ] Training curves documented (loss, embedding stats, val Recall@K
      over time)
- [ ] At least one hyperparameter iteration informed by the first run
      (e.g. adjusted LR, queue size, or temperature init)

**Dependencies:** Phase 3.5 passed.

**Acceptance criteria:** A checkpoint exists with val Recall@10
clearly and reproducibly above random-chance baseline (documented
with the actual number, not just "it works").

**Status:** not started

---

## Phase 5 — Evaluation

**Goal:** Rigorously evaluate what the trained model actually learned,
quantitatively and qualitatively.

**Deliverables:**
- [ ] Recall@1/5/10 for image→text and text→image on the test split
- [ ] Embedding space diagnostics (collapse/uniformity checks)
- [ ] Qualitative review: manually inspect 10+ retrieval
      successes AND failures, write down patterns observed

**Dependencies:** Phase 4 complete.

**Acceptance criteria:** Metrics reported on the held-out test split
with a stated comparison to random-chance baseline; qualitative
failure analysis documented, not just the numbers.

**Status:** not started

---

## Phase 6 — Serving / Retrieval Infrastructure

**Goal:** Build a queryable retrieval system on top of the trained
embeddings.

**Deliverables:**
- [ ] Vector index (FAISS or equivalent) over the full embedding set
- [ ] API layer for query → retrieval (image or text in, ranked
      results out)

**Dependencies:** Phase 5 complete, model quality validated as
meaningfully above baseline.

**Acceptance criteria:** A live query (image or text) returns ranked,
correctly-shaped results from the index via the API, matching what
Phase 5's offline evaluation predicted.

**Status:** not started

---

## Phase 7 — Portfolio Polish

**Goal:** Package the project for interviews/portfolio presentation.

**Deliverables:**
- [ ] Demo interface
- [ ] Write-up of design decisions and tradeoffs
- [ ] Write-up of the debugging story (what broke, how it was found,
      how it was fixed — including the Phase 3.5 sanity check result)

**Dependencies:** Phases 0–6 complete.

**Acceptance criteria:** A reader unfamiliar with the project can
understand what it does, how it was built, why key decisions were
made, and what went wrong along the way, from the docs alone.

**Status:** not started

---

## Phase 0 — Project Setup & Architecture Decisions

**Goal:** Establish a reproducible project skeleton and lock in all
core architecture decisions before any model/data code is written.

**Deliverables:**
- [x] Repository structure created
- [ ] Virtual environment + PyTorch (GPU) verified working
- [ ] `CLAUDE.md`, `ROADMAP.md`, `ARCHITECTURE.md` committed
- [ ] Architecture decisions locked (encoders, embedding dim, batch
      strategy for 6GB VRAM)

**Dependencies:** None — this is the foundation.

**Acceptance criteria:** Repo builds, CUDA verified on RTX 4050, all
three core docs exist and are pushed to `origin main`.

**Status:** in progress

---

## Phase 1 — Data Pipeline

**Goal:** Acquire, clean, and structure Flickr30k into a
training-ready paired image-text dataset.

**Deliverables:**
- [ ] Flickr30k downloaded and verified
- [ ] Image transforms + tokenization pipeline
- [ ] Paired `Dataset`/`DataLoader` implementation
- [ ] Train/val/test split
- [ ] Sanity checks: visualized batches, pairing integrity confirmed

**Dependencies:** Phase 0 complete.

**Acceptance criteria:** A `DataLoader` yields correctly paired,
correctly shaped image/text tensors; no pairing leakage between
splits.

**Status:** not started

---

## Phase 2 — Model Architecture

**Goal:** Implement the dual-encoder model: image tower, text tower,
projection heads into a shared embedding space.

**Deliverables:**
- [ ] Image encoder (small CNN, from scratch)
- [ ] Text encoder (small Transformer, from scratch)
- [ ] Projection heads + L2 normalization into shared embedding space
- [ ] Learnable temperature parameter

**Dependencies:** Phase 0 architecture decisions finalized.

**Acceptance criteria:** Forward pass runs end-to-end on a sample
batch, producing L2-normalized embeddings of the agreed shared
dimension for both modalities.

**Status:** not started

---

## Phase 3 — Contrastive Training Loop

**Goal:** Implement and run the contrastive training loop.

**Deliverables:**
- [ ] Symmetric InfoNCE loss implementation
- [ ] Mixed precision + gradient accumulation
- [ ] Checkpointing
- [ ] Training/validation logging (W&B or TensorBoard)
- [ ] First full training run completed

**Dependencies:** Phases 1 and 2 complete.

**Acceptance criteria:** Loss decreases and stabilizes over training;
embeddings do not collapse (verified via embedding norm/variance
diagnostics).

**Status:** not started

---

## Phase 4 — Evaluation

**Goal:** Quantitatively evaluate retrieval quality.

**Deliverables:**
- [ ] Recall@1/5/10 for image→text and text→image
- [ ] Embedding space diagnostics (collapse/uniformity checks)

**Dependencies:** Phase 3 complete (at least one trained checkpoint).

**Acceptance criteria:** Metrics computed on held-out test split,
documented with interpretation (not just raw numbers).

**Status:** not started

---

## Phase 5 — Serving / Retrieval Infrastructure

**Goal:** Build a queryable retrieval system on top of the trained
embeddings.

**Deliverables:**
- [ ] Vector index (FAISS or equivalent) over the embedding set
- [ ] API layer for query → retrieval

**Dependencies:** Phase 4 complete, model quality validated.

**Acceptance criteria:** A query (image or text) returns ranked,
correct-shape results from the index via the API.

**Status:** not started

---

## Phase 6 — Portfolio Polish

**Goal:** Package the project for interviews/portfolio presentation.

**Deliverables:**
- [ ] Demo interface
- [ ] Write-up of design decisions and tradeoffs
- [ ] Write-up of the debugging story (what broke, how it was fixed)

**Dependencies:** Phases 0–5 complete.

**Acceptance criteria:** A reader unfamiliar with the project can
understand what it does, how it was built, and why key decisions were
made, from the docs alone.

**Status:** not started