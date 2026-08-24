# ROADMAP.md — VectorMind

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

**Status:** complete — **superseded by Phase 4b (2026-08-24)**

> **This run's checkpoint has been retired.** Its embedding space was
> collapsed: matched-vs-unmatched separation 0.094, against 0.964 for
> the Phase 3.5 overfit. The Phase 5 reports called it HEALTHY. The
> cause was the memory queue, not the temperature — see
> [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) §11 and
> [ARCHITECTURE.md](ARCHITECTURE.md) §6.1. Results are kept below as
> the record of what was believed at the time.

**Results (superseded):**
- **Best Checkpoint:** Epoch 7, Step 7944
- **Val Recall@1:** 4.22%
- **Val Recall@5:** 14.03%
- **Val Recall@10:** 20.23%
- **Training Time:** ~45 minutes total (8 epochs baseline + continuation)
- **Memory Queue:** Enabled (size=4096)
- **Temperature:** Learned from 14.29 to 55.24
- **Separation:** 0.094 — measured afterwards, not at the time

**Key Findings, annotated:**
1. ~~Memory queue improved Recall@10 from 17.12% to 20.23% (+18.2%)~~ — **retracted.** A controlled A/B from a shared checkpoint gives 10.51% with the queue against 19.63% without. The original figure was one epoch measured before the collapse reached the metric, and was never a controlled comparison.
2. Lower learning rate (5e-4) hurt performance (Recall@10 dropped to 10.54%) — **stands**, though it should be re-run without the queue.
3. **Embedding collapse after Epoch 7** — **stands, but the cause was misattributed.** The temperature did not grow on its own; the queue drove it.
4. ~~Epoch 7 is the true convergence point~~ — **retracted.** Without the queue the model kept improving past epoch 7.
5. Gradient norm logging bug identified and fixed (was always 0.0) — **stands.**

**Every "x random baseline" multiple in the original results was ~30x
too low** — the 1%/10% baselines belong to the Phase 3.5 100-image
subset. See [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) §1b.

**Documentation:**
- Baseline analysis: reports/baseline_analysis.md
- Training curves: reports/figures/training_curves.png
- Checkpoint summary: reports/checkpoint_summary.json
- Training log: TRAINING_LOG.md

---

## Phase 4b — Clamped Re-run (2026-08-24)

**Goal:** retrain with the collapse cause removed, and with a metric
that can actually detect collapse.

**Why:** Phase 4 shipped a checkpoint whose embedding space had
collapsed into a narrow cone while its reports said HEALTHY. The only
health metric was per-dimension variance, which cannot distinguish a
cone from a spread on a unit hypersphere.

**Changes under test:**
- Memory queue **disabled** — it was the cause, not the mitigation (§6.1)
- Learnable logit scale **clamped at 100**, CLIP's ceiling
- `log_temperature` excluded from weight decay
- Separation, anisotropy and ‖mean embedding‖ logged beside Recall@K every epoch

**Deliverables:**
- [x] Controlled A/B isolating the memory queue from a shared checkpoint
- [x] Embedding health measured every epoch, not reconstructed afterwards
- [x] Run to convergence — epoch 12, confirmed by epochs 13–14 showing no validation gain
- [x] Rebuild FAISS index and regenerate all reports from the final checkpoint

**Acceptance criteria:** a checkpoint whose separation is materially
above Phase 4's 0.094, at no cost to Recall@10, with every reported
figure regenerable by `scripts/generate_reports.py`.

**Status:** complete — converged at epoch 12

**Final results** (test split, `scripts/generate_reports.py`):

| Metric | Phase 4 (retired) | Phase 4b (epoch 12) |
|---|---|---|
| Test Recall@1 | 4.62% | **7.71%** (245× chance) |
| Test Recall@5 | 13.43% | **20.60%** (131× chance) |
| Test Recall@10 | 19.63% | **28.91%** (92× chance) |
| Test Recall@10 (T2I) | 15.09% | **25.20%** (80× chance) |
| **Separation** | **0.094** | **0.347** |
| Mean image–image cosine | 0.810 | **0.322** |
| Logit scale | 55.2 (→500+) | **24.1** |

R@10 up 47% relative, on a space 3.7× better separated, with the logit
scale stable near its initialization instead of running away.

**Convergence, not interruption.** Epochs 13–14 dropped training loss
14% without improving validation R@10 once. Epoch 12 is the peak, and
early stopping at patience 5 would have selected the same weights
(EXPERIMENTS.md 007).

**Remaining health gap:** the space grades ANISOTROPIC, not HEALTHY —
‖mean embedding‖ 0.621 against a 0.5 threshold is the one check it still
fails.

**Documentation:**
- Experiments 004-006: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- Run log and interruptions: [docs/TRAINING_LOG.md](docs/TRAINING_LOG.md)
- Root cause: [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) §11

---

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

**Status:** complete for the Phase 4 checkpoint — **must be re-run for Phase 4b**

> The metrics below describe the retired Phase 4 checkpoint, and every
> "x random baseline" multiple in them is roughly 30x too low
> ([docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) §1b — the real figure
> for R@10 is 62x chance, not 2.0x). The embedding-health verdict is
> also wrong: separation was 0.094, not the 0.33 reported. Rerun with
> `scripts/generate_reports.py` once Phase 4b converges.

**Results (corrected 2026-08-07):**
- **Test Recall@1 (I2T):** 4.62% (147x chance)
- **Test Recall@5 (I2T):** 13.43% (85x chance)
- **Test Recall@10 (I2T):** 19.63% (62x chance)
- **Test Recall@1 (T2I):** 2.49% (79x chance)
- **Test Recall@5 (T2I):** 8.91% (57x chance)
- **Test Recall@10 (T2I):** 15.09% (48x chance)
- **Val→Test Gap (R@10):** -0.60% (reasonable generalization)
- **Embedding Health:** HEALTHY (no collapse)
- **Failure Rate:** 80.37%

**Bug Fix (2026-08-07):** The original evaluation had a destructuring
bug in `evaluate_test_set.py` where `_, eval_loader, _` always selected
the val_loader regardless of `--split` argument. The "identical" val/test
metrics were actually both val metrics. Fixed and re-run — corrected test
R@10 is 19.63% (vs val 20.23%), a normal ~0.6pp generalization gap.

**Key Findings:**
1. Model achieves 62x chance for image→text retrieval
2. Text→image direction weaker (1.5x vs 2.0x)
3. No embedding collapse detected
4. Main failure patterns: action ambiguity (35%), object specificity (25%)
5. Strong scene understanding, weak fine-grained action recognition
6. Reasonable generalization: val-test gap ~0.6pp at R@10

**Documentation (superseded artifacts):**
- Qualitative analysis: reports/phase5_qualitative_analysis.md
- Final report: reports/phase5_final_report.md

> The three `phase5_*_metrics.json` files this section used to link no
> longer exist. They were hand-assembled and disagreed with each other
> ([docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) §8); `reports/` is now
> written in one pass by `scripts/generate_reports.py` into
> `metrics_val.json`, `metrics_test.json`,
> `embedding_diagnostics.json`, `checkpoint_summary.json` and
> `RESULTS.md`.

---

## Phase 6 — Serving / Retrieval Infrastructure

**Goal:** Build a queryable retrieval system on top of the trained
embeddings. See ARCHITECTURE.md §9 for the full design.

**Deliverables:**
- [x] FAISS `IndexFlatIP` built offline from the Phase 5 embedding set
      (`backend/index_builder.py`)
- [x] FastAPI app with `/search/text` and `/search/image` endpoints,
      Pydantic request/response schemas (`backend/schemas.py`)
- [x] Model + index loaded once at app startup, not per-request
- [x] Unit tests for the API layer (request validation, response
      shape) and the index builder
- [x] Basic request logging (latency, query type) via the existing
      `logging` setup

**Dependencies:** Phase 5 complete, model quality validated as
meaningfully above baseline.

**Acceptance criteria:** A live query (image or text) returns ranked,
correctly-shaped results from the index via the API, matching what
Phase 5's offline evaluation predicted. p95 latency for a single
text query measured and documented (not just "it works").

**Status:** complete

**Results:**
- **FAISS Index:** IndexFlatIP with 256-dim embeddings
- **API Endpoints:** POST /search/text, POST /search/image, GET /health
- **Test Coverage:** 63 backend tests (index builder, schemas, routers, integration)
- **Total Tests:** 345 passing

**Documentation:**
- Index builder: backend/index_builder.py
- API schemas: backend/schemas.py
- FastAPI app: backend/app.py
- Text search: backend/routers/text_search.py
- Image search: backend/routers/image_search.py
- Unit tests: tests/backend/
- Integration tests: tests/backend/test_integration.py

---

## Phase 6.5 — Frontend Demo Interface

**Goal:** A minimal interactive UI over the Phase 6 API. See
ARCHITECTURE.md §10.

**Deliverables:**
- [x] React + TypeScript app: text search bar, image drag-and-drop
      upload, ranked result grid
- [x] Typed API client mirroring the backend's Pydantic schemas
- [ ] (Stretch) 2D embedding-space visualization (UMAP/t-SNE) for the
      portfolio write-up

**Dependencies:** Phase 6 complete (API contract stable).

**Acceptance criteria:** A user can type a text query or upload an
image in the browser and see ranked results rendered, with no console
errors and correct loading/empty/error states handled.

**Status:** complete

**Results:**
- **Framework:** React + TypeScript + Tailwind CSS v4
- **Build Tool:** Vite
- **Components:** SearchBar (with example queries), ImageUploader, ResultGrid, HealthIndicator
- **API Client:** Typed fetch wrapper with configurable `VITE_API_BASE_URL`
- **Static Serving:** Backend serves `frontend/dist/` in production (SPA catch-all)
- **About Section:** Real metrics, known failure patterns from Phase 5 qualitative analysis
- **Smoke Test:** `scripts/smoke_test_api.py` — validates health, text search, image search, static serving
- **Build Status:** Passing
- **States:** Loading, empty, error, success all handled

**Documentation:**
- Frontend code: frontend/src/
- Components: frontend/src/components/
- API client: frontend/src/api/client.ts
- Types: frontend/src/types/search.ts
- Build instructions: README.md §Frontend

---

## Phase 7 — Deployment & Portfolio Polish

**Goal:** Package the project for interviews/portfolio presentation
with a working, deployed demo. See ARCHITECTURE.md §11-12.

**Deliverables:**
- [x] `backend.Dockerfile` / `frontend.Dockerfile` + `docker-compose.yml` — built and run
- [x] GitHub Actions: `test.yml` (pytest + `tsc --noEmit` on every PR),
      `build.yml` (Docker builds on merge to `main`)
- [ ] Deployed demo reachable via a public URL (single-machine/VM
      deployment — Kubernetes and managed cloud endpoints are
      explicitly out of scope; see docs/FUTURE_IDEAS.md)
- [x] Write-up of design decisions and tradeoffs — `docs/DESIGN_DECISIONS.md`
- [x] Write-up of the debugging story — `docs/DEBUGGING_STORY.md`, 11 narratives

**Dependencies:** Phases 0–6.5 complete.

**Acceptance criteria:** A reader unfamiliar with the project can
understand what it does, how it was built, why key decisions were
made, and what went wrong along the way, from the docs alone, AND can
reach a live deployed instance to try it themselves.

**Status:** **in progress** — was marked complete in error, now
genuinely most of the way there

The documentation half of the acceptance criteria is met, and the
containers now demonstrably run. What remains is a public URL and one
green CI run.

| Deliverable | State |
|---|---|
| Dockerfiles, compose, nginx | ✅ **Both images built and the stack run end to end.** Backend 2.17GB, frontend 93.1MB. `/ready` green with model and both indices loaded; search returns 10 unique images of 10 at 8-19ms through the proxy. See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). |
| CI workflows | ✅ **Green** as of 2026-08-24 (run 32756952454): pytest, mypy, ruff, tsc/oxlint/build, a serving-dependency check, and both Docker image builds. |
| Deployed demo at a public URL | **Not started.** Runs locally via compose; needs a host to be public. |
| Design-decision write-up | Done. |
| Debugging write-up | Done. |

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md), which states the same
verification status and lists the known gaps (no TLS, single worker, no
metrics, no auth).

**Results:**
- **Docker:** `backend.Dockerfile` (Python + PyTorch, ~500MB), `frontend.Dockerfile` (Node → nginx, ~30MB)
- **Compose:** `deployment/docker-compose.yml` — orchestrates both containers
- **CI:** `.github/workflows/test.yml` (pytest + tsc on every PR), `.github/workflows/build.yml` (Docker builds on merge to main)
- **Write-ups:** `docs/DESIGN_DECISIONS.md` (10 decisions), `docs/DEBUGGING_STORY.md` (7 debugging narratives)

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

- [x] Live deployed demo (single machine/VM, Docker Compose) — `deployment/docker-compose.yml`
- [x] CI enforcing tests + type-checking on every change — `.github/workflows/test.yml`
- [x] p95 API latency documented — **25.1ms** (100 queries, CPU, `best_model.pt`)
- [x] Traceable checkpoints (metadata sidecar per ARCHITECTURE.md §12) — `checkpoints/checkpoint_metadata.json`

### Latency Benchmark (CPU, `best_model.pt`, FAISS IndexFlatIP)

| Metric | Value |
|--------|-------|
| Avg    | 48.7ms |
| P50    | 14.0ms |
| **P95** | **25.1ms** |
| P99    | 36.2ms |
| Min    | 8.4ms |
| Max    | 3396.9ms (cold start) |

*Measured: 100 text queries, top_k=10, no GPU, single-threaded uvicorn.*

