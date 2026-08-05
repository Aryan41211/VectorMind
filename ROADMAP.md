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
- [x] Virtual environment + PyTorch (GPU) verified working
- [x] `CLAUDE.md`, `ROADMAP.md`, `ARCHITECTURE.md` committed
- [x] Empirical batch-size ceiling measured and documented

**Dependencies:** None — this is the foundation.

**Acceptance criteria:** Repo builds; CUDA verified on RTX 4050; a
documented, measured (not guessed) max batch size exists in
ARCHITECTURE.md.

**Status:** complete

---

## Phase 1 — Data Pipeline

**Goal:** Acquire, clean, and structure Flickr30k into a
training-ready, verified-correct paired image-text dataset.

**Deliverables:**
- [x] Flickr30k downloaded and verified (checksums / count matches
      expected)
- [x] Image transforms + tokenization pipeline
- [x] Paired `Dataset`/`DataLoader` implementation
- [x] Train/val/test split with zero image leakage across splits
- [x] Sanity checks: visualize N batches with decoded captions
      side-by-side with images (manual inspection, not just shape
      checks) to catch pairing/tokenizer bugs

**Dependencies:** Phase 0 complete.

**Acceptance criteria:** A `DataLoader` yields correctly paired,
correctly shaped tensors; manual inspection of 10+ samples confirms
image-caption pairing is correct and captions decode to sensible text.

**Status:** complete

---

## Phase 2 — Model Architecture

**Goal:** Implement the dual-encoder model and confirm it runs
end-to-end, before any real training.

**Deliverables:**
- [x] Image encoder (small CNN, from scratch)
- [x] Text encoder (small Transformer, from scratch)
- [x] Projection heads + L2 normalization into shared embedding space
- [x] Learnable temperature parameter
- [x] Forward-pass smoke test: one batch through the full model,
      correct output shapes, no NaNs

**Dependencies:** Phase 0 architecture decisions + measured batch
size finalized.

**Acceptance criteria:** Forward pass runs end-to-end on a real batch
from the Phase 1 pipeline (not synthetic data), producing
L2-normalized embeddings of the agreed shared dimension for both
modalities, with no NaN/Inf values.

**Status:** complete

---

## Phase 3 — Training Infrastructure

**Goal:** Build the training loop machinery itself — separate from
actually training a real model (that's Phase 3.5/4). This phase is
"does the machine run," not "does the model learn."

**Deliverables:**
- [x] Symmetric InfoNCE loss implementation (unit tested against a
      hand-computed small example)
- [x] MoCo-style memory queue implementation (unit tested for
      correct enqueue/dequeue behavior)
- [x] Mixed precision + gradient accumulation wired in
- [x] Checkpointing (save/resume, including optimizer state)
- [x] Logging to W&B/TensorBoard: loss, temperature, embedding norm,
      embedding variance, GPU memory usage

**Dependencies:** Phases 1 and 2 complete.

**Acceptance criteria:** Loss function has a passing unit test with
a known expected value; a training loop can run for a few steps on
real data without crashing, and a checkpoint can be saved and
successfully reloaded to resume.

**Status:** complete

---

## Phase 3.5 — Sanity Check: Overfit a Tiny Subset

**Goal:** Prove the entire pipeline can actually learn something
before spending real compute on a full run. This is the single most
important risk-reduction step in the project and is not optional.

**Deliverables:**
- [x] Train on a tiny fixed subset (e.g. 50–100 image-caption pairs)
      for enough steps to memorize it
- [x] Confirm near-perfect Recall@1 on that same tiny subset
- [x] Confirm embedding variance stays healthy (not collapsing to a
      single point) throughout

**Dependencies:** Phase 3 complete.

**Acceptance criteria:** The model achieves near-perfect retrieval on
the tiny memorized subset. If it cannot, training does not proceed to
Phase 4 — debug here first, since nothing downstream can be trusted
otherwise.

**Status:** done — VERIFIED PASSED
- Image->Text Recall@1: 100.0% (100x random baseline of 1%)
- Text->Image Recall@1: 100.0% (100x random baseline of 1%)
- Similarity separation: 0.964 (matched: 0.955, unmatched: -0.009)
- Embedding variance: 0.0039 (healthy, no collapse)
- Temperature learned: 15.33 (from CLIP init of 14.29)
- 100 images, 500 pairs, 30 epochs, batch_size=32, lr=3e-4
- Evaluation report: reports/overfit/phase3_5_evaluation.json

---

## Phase 4 — Full Training Run(s)

**Goal:** Train the real model on the full Flickr30k training split,
iterating on hyperparameters as needed.

**Deliverables:**
- [x] At least one full training run to convergence (loss plateaued,
      val metrics stopped improving)
- [x] Training curves documented (loss, embedding stats, val Recall@K
      over time)
- [x] At least one hyperparameter iteration informed by the first run
      (e.g. adjusted LR, queue size, or temperature init)

**Dependencies:** Phase 3.5 passed.

**Acceptance criteria:** A checkpoint exists with val Recall@10
clearly and reproducibly above random-chance baseline (documented
with the actual number, not just "it works").

**Status:** complete

**Results:**
- **Best Checkpoint:** Epoch 7, Step 7944 (checkpoints/train/best_model.pt)
- **Val Recall@1:** 4.22% (4.2x random baseline)
- **Val Recall@5:** 14.00%
- **Val Recall@10:** 20.23% (2.0x random baseline)
- **Training Time:** ~8 minutes (2 epochs with memory queue)
- **Memory Queue:** Enabled (size=4096)
- **Temperature:** Learned from 14.29 to 53.51

**Key Findings:**
1. Memory queue fix improved Recall@10 from 17.12% to 20.23% (+18.2% relative)
2. Lower learning rate (5e-4) hurt performance (Recall@10 dropped to 10.54%)
3. Embedding variance remained healthy (no collapse)
4. Temperature increased significantly, indicating model learned to sharpen similarity

**Documentation:**
- Baseline analysis: reports/baseline_analysis.md
- Training curves: reports/figures/training_curves.png
- Checkpoint summary: reports/checkpoint_summary.json
- Training log: TRAINING_LOG.md

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
embeddings. See ARCHITECTURE.md §9 for the full design.

**Deliverables:**
- [ ] FAISS `IndexFlatIP` built offline from the Phase 5 embedding set
      (`backend/index_builder.py`)
- [ ] FastAPI app with `/search/text` and `/search/image` endpoints,
      Pydantic request/response schemas (`backend/schemas.py`)
- [ ] Model + index loaded once at app startup, not per-request
- [ ] Unit tests for the API layer (request validation, response
      shape) and the index builder
- [ ] Basic request logging (latency, query type) via the existing
      `logging` setup

**Dependencies:** Phase 5 complete, model quality validated as
meaningfully above baseline.

**Acceptance criteria:** A live query (image or text) returns ranked,
correctly-shaped results from the index via the API, matching what
Phase 5's offline evaluation predicted. p95 latency for a single
text query measured and documented (not just "it works").

**Status:** not started

---

## Phase 6.5 — Frontend Demo Interface

**Goal:** A minimal interactive UI over the Phase 6 API. See
ARCHITECTURE.md §10.

**Deliverables:**
- [ ] React + TypeScript app: text search bar, image drag-and-drop
      upload, ranked result grid
- [ ] Typed API client mirroring the backend's Pydantic schemas
- [ ] (Stretch) 2D embedding-space visualization (UMAP/t-SNE) for the
      portfolio write-up

**Dependencies:** Phase 6 complete (API contract stable).

**Acceptance criteria:** A user can type a text query or upload an
image in the browser and see ranked results rendered, with no console
errors and correct loading/empty/error states handled.

**Status:** not started

---

## Phase 7 — Deployment & Portfolio Polish

**Goal:** Package the project for interviews/portfolio presentation
with a working, deployed demo. See ARCHITECTURE.md §11-12.

**Deliverables:**
- [ ] `backend.Dockerfile` / `frontend.Dockerfile` + `docker-compose.yml`
- [ ] GitHub Actions: `test.yml` (pytest + `tsc --noEmit` on every PR),
      `build.yml` (Docker builds on merge to `main`)
- [ ] Deployed demo reachable via a public URL (single-machine/VM
      deployment — Kubernetes and managed cloud endpoints are
      explicitly out of scope; see FUTURE_IDEAS.md)
- [ ] Write-up of design decisions and tradeoffs
- [ ] Write-up of the debugging story (what broke, how it was found,
      how it was fixed — including the Phase 3.5 sanity check result)

**Dependencies:** Phases 0–6.5 complete.

**Acceptance criteria:** A reader unfamiliar with the project can
understand what it does, how it was built, why key decisions were
made, and what went wrong along the way, from the docs alone, AND can
reach a live deployed instance to try it themselves.

**Status:** not started

---

## Stretch / Research Goals (not on the critical path)

These are explicitly optional. Pursuing them should never block or
delay Phases 0-7 above. See FUTURE_IDEAS.md for full detail on each.

- Multilingual caption retrieval (would require a multilingual
  tokenizer + dataset beyond Flickr30k)
- Knowledge distillation into a smaller/faster inference model
- Quantization (int8) for cheaper CPU inference
- LoRA-style fine-tuning experiments once a base checkpoint exists
- Approximate nearest-neighbor index (HNSW) benchmarked against the
  current flat FAISS index, once/if corpus size grows

## Benchmark Goals

- Document Recall@1/5/10 against the random-chance baseline (Phase 5,
  already required) AND, as a stretch, against a small pretrained
  CLIP checkpoint run in inference-only mode on the same test split —
  purely as a labeled reference point for the write-up, never as a
  target this from-scratch model is expected to hit. This does not
  change the from-scratch training constraint (ARCHITECTURE.md §8);
  no pretrained weights are loaded into VectorMind's own model at any
  point.

## Production Goals (Phase 7 scope, restated for clarity)

- Live deployed demo (single machine/VM, Docker Compose)
- CI enforcing tests + type-checking on every change
- p95 API latency documented
- Traceable checkpoints (metadata sidecar per ARCHITECTURE.md §12)

