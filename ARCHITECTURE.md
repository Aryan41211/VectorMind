# ARCHITECTURE.md — VectorMind

This document describes the technical architecture of VectorMind and
the reasoning behind every major design decision. Update it whenever
a structural decision changes — it must always reflect current
reality, not the original plan if that plan changed.

---

## 1. High-Level System Diagram

```
                 ┌────────────────────┐
   Image  ────▶  │   Image Encoder    │
                 │  (CNN, from scratch)│
                 └─────────┬──────────┘
                           │ pooled feature vector
                           ▼
                 ┌────────────────────┐
                 │ Image Projection   │
                 │  Head + L2 Norm    │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │  Shared Embedding  │◀── Contrastive Loss (InfoNCE,
                 │      Space         │     learnable temperature)
                 └─────────▲──────────┘
                           │
                 ┌────────────────────┐
                 │  Text Projection   │
                 │  Head + L2 Norm    │
                 └─────────▲──────────┘
                           │ pooled feature vector
                 ┌────────────────────┐
   Text   ────▶  │   Text Encoder     │
                 │(Transformer, scratch)│
                 └────────────────────┘
```

Both towers map their input into a shared `D`-dimensional embedding
space. Matching image-text pairs are pulled together; non-matching
pairs (within the batch, and via a memory queue — see §6) are pushed
apart.

---

## 2. Image Encoder

**Decision:** Small CNN, ResNet-18-style, trained from scratch (NOT
pretrained on ImageNet).

**Why not a Vision Transformer:** ViTs lack the convolutional
inductive bias (locality, translation equivariance) that lets CNNs
learn efficiently from small datasets. ViT-from-scratch needs far
more data and compute than ~30k images on 6GB VRAM can support. A
from-scratch ViT here would likely underperform a CNN and cost more
to train — the wrong tradeoff for this project's constraints.

**Output:** Global-average-pooled feature vector, dimension set by
the final conv block (e.g. 512), fed into the image projection head.

---

## 3. Text Encoder

**Decision:** Small Transformer encoder built from scratch (4–6
layers, embedding dim 256, learned positional embeddings).

**Why not an LSTM:** A from-scratch Transformer is more demonstrative
of current ML engineering skill for a portfolio, and self-attention
handles variable-length captions well without the sequential
bottleneck of an RNN. At this small scale (4-6 layers) it remains
trainable on 6GB VRAM.

**Tokenizer:** Use an existing subword tokenizer (e.g. a small
pretrained BPE tokenizer) purely for tokenization — no pretrained
*embeddings* or encoder weights are used. Tokenization is a
preprocessing utility, not part of the learned model; the actual
text representations are trained entirely from scratch.

**Output:** Pooled representation (e.g. mean-pooled or [CLS]-style
token) fed into the text projection head.

---

## 4. Projection Heads

**Decision:** A single linear layer per modality mapping the
tower's output to a shared embedding dimension (recommended: 256),
followed by L2 normalization.

L2 normalization is required so that the dot product between
embeddings equals cosine similarity — this is what the contrastive
loss operates on.

---

## 5. Contrastive Loss

**Decision:** Symmetric InfoNCE — cross-entropy in both directions
(image→text and text→image), averaged, with a **learnable**
temperature parameter (initialized as in CLIP, e.g. `log(1/0.07)`),
not a fixed constant.

A learnable temperature lets the model calibrate how "sharp" the
similarity distribution should be over training, rather than us
guessing a fixed value upfront.

### 5.1 The scalar is a logit scale, and it must be clamped

Naming, because this caused a real failure: the parameter is stored as
`log_temperature`, but the loss **multiplies** similarities by
`exp(log_temperature)`. That makes it CLIP's `logit_scale` — the
*reciprocal* of a temperature. Larger values sharpen the distribution;
they do not soften it. The docstring said the opposite until the
2026-08-23 audit.

**Decision (2026-08-23):** clamp the logit scale at **100**, CLIP's own
ceiling, after every optimizer step. Configured at
`configs/training.yaml → temperature.max_logit_scale`.

**Why this is a correctness guard and not a hyperparameter:** the scalar
is unconstrained, and inflating it lowers contrastive loss without
improving the representation. It is the cheapest direction in the loss
landscape, and the optimizer takes it. Phase 4 ran without a ceiling:

| Epoch | 6 | 7 | 9 | 11 | 14 |
|---|---|---|---|---|---|
| Logit scale | 18.6 | 55.2 | 78.6 | 160.5 | 338.6 |
| Val R@10 | 17.12% | **20.23%** | 13.62% | 17.31% | 13.75% |

Image variance fell 83% and text variance 86% over that span. The epoch-7
checkpoint was shipped as healthy; measuring it directly afterwards gave
a matched-vs-unmatched separation of 0.094, against 0.964 for the
Phase 3.5 overfit run. It was not healthy, only earlier on the same
curve.

`log_temperature` is also excluded from weight decay. Decay on it is not
regularization — it is a constant pull toward a logit scale of 1.0,
which is an arbitrary target unrelated to generalization. The clamp is
the bound; weight decay was noise.

See `docs/KNOWN_ISSUES.md` §1.

---

## 6. VRAM-Constrained Batch Strategy (critical)

This is the section that most needs explicit design, because 6GB
VRAM is the binding constraint on the entire project.

**Problem:** Contrastive learning quality scales strongly with the
number of negative samples per positive pair. In-batch negatives
(the standard CLIP approach) require large batch sizes (CLIP itself
used 32,768) to work well — completely infeasible on 6GB.

**Mitigations, in order of priority:**

1. **Mixed precision (`torch.cuda.amp`)** — default, not optional.
   Roughly halves activation memory, allowing a larger batch than
   FP32 would permit.
2. **Gradient accumulation** — simulate a larger *effective* batch
   size by accumulating gradients over several small forward/backward
   passes before an optimizer step. This helps the optimizer's
   statistics but does **not** by itself increase the number of
   negatives seen in a single contrastive comparison — see next point.
3. **Memory queue (MoCo-style)** — maintain a queue of embeddings from
   recent past batches to use as additional negatives, decoupling
   the number of negatives from the physical batch size. This is the
   key mitigation for negative-sample count specifically, since
   gradient accumulation alone doesn't fix it. **It requires a warmup
   period — see §6.1.**
4. **Gradient checkpointing** — fallback if still OOM after the
   above; trades compute for memory by recomputing activations during
   the backward pass instead of storing them.

**Actual batch size and queue size determined empirically in Phase 0.2:**
- **Max safe batch size: 256** (measured on RTX 4050 Laptop GPU, 6.44 GB VRAM)
- **Peak VRAM at batch 256: 4.99 GB** (under 5.2 GB ceiling with 10% safety margin)
- **Search method:** exponential then binary search, AMP enabled, 5.2 GB ceiling
- **Recommended memory-queue size (pending Phase 3.5 validation): 4096** (16× batch for negative diversity)
- **Image tensor shape:** [B, 3, 224, 224]
- **Text tensor shape:** [B, 77] (CLS pooling)
- **Encoder dims:** image 512, text 256, shared embedding 256

The above replaces the placeholder text "Actual batch size and queue size will be determined empirically in Phase 0.2..."

Training uses **batch 128**, not the profiled 256. The profiler measured
a forward/backward pass in isolation; a live epoch also holds the
validation pass and the 4096-entry queue. 128 measured ~4.6 GB peak in
practice. Both values live in `configs/data.yaml`.

### 6.1 The memory queue is not true MoCo, and needs a warmup

**What is missing:** MoCo pairs its queue with a **momentum encoder** —
a slowly-updated EMA copy of the encoder that produces the queued keys.
That is the mechanism that keeps entries written 32 batches ago
comparable to what the current encoder produces. VectorMind has no
momentum encoder; the queue holds raw outputs of the live encoder.

**Why that matters at this scale:** with `queue_size` 4096 against a
batch of 128, queued negatives outnumber in-batch negatives 32 to 1. If
they are stale, the gradient is dominated by noise.

**Measured (2026-08-23):**

| Schedule | Val R@10 after 2 epochs | Separation |
|---|---|---|
| Queue active from step 1 | 0.35% (chance) | 0.000 |
| In-batch negatives first | 2.99% → 5.03% | 0.102 → 0.166 |

**Decision:** the queue fills from step 1 but serves no negatives until
`memory_queue.warmup_epochs` (default 6) has passed, so it activates
already full of *recent* embeddings. Implemented as
`MemoryQueue.active`.

This is the schedule that produced Phase 4's best checkpoint — reached
by accident, because the first six epochs ran with a forgotten
`--no-queue` flag. The improvement was recorded at the time as "the
memory queue helps"; the fuller finding is "the memory queue helps once
the encoder has stabilized." It is deliberate and configured now.

A momentum encoder would remove the need for the warmup and is the
correct long-term fix; it costs a second copy of both towers in VRAM,
which is why it is in `docs/FUTURE_IDEAS.md` rather than here.

---

## 7. Repository Structure

```
src/vectormind/
├── data/          → Phase 1: Dataset/DataLoader, transforms, tokenization
├── models/
│   ├── image_encoder.py     → §2
│   ├── text_encoder.py      → §3
│   ├── projection_head.py   → §4
│   └── vectormind_model.py  → combines towers + heads into one model
├── training/
│   ├── losses.py            → §5 (InfoNCE)
│   ├── memory_queue.py      → §6 (MoCo-style queue)
│   ├── train_loop.py        → training loop, mixed precision, accumulation
│   ├── checkpoint.py        → save/load full training state
│   └── logger.py            → TensorBoard metrics logging
├── evaluation/    → Phase 4: Recall@K, embedding diagnostics
└── utils/         → logging, checkpointing, seeding, config loading
```

---

## 8. Key Design Tradeoffs

| Decision Point            | Chosen                     | Rejected                | Why |
|---------------------------|----------------------------|--------------------------|-----|
| Image encoder             | Small CNN (ResNet-18-style) | ViT-from-scratch        | ViT needs far more data/compute than 30k images + 6GB VRAM supports |
| Text encoder              | Small Transformer (scratch) | LSTM                    | Demonstrates modern architecture understanding; self-attention handles variable-length text without RNN bottlenecks |
| Tokenization               | Pretrained BPE tokenizer only | Train tokenizer from scratch | Tokenization is preprocessing, not a learned component — no shortcut taken on the actual model |
| Negative sampling strategy | In-batch + MoCo-style memory queue | In-batch only | 6GB VRAM caps real batch size too low for in-batch negatives alone to give stable contrastive signal |
| Temperature                | Learnable parameter        | Fixed constant           | Lets the model calibrate similarity sharpness during training, as in CLIP |
| Pretrained weights         | None anywhere in the model | Fine-tune CLIP/OpenCLIP | Explicit project goal: train from scratch for deep understanding + portfolio differentiation |

---

## 9. Serving & Retrieval Architecture (Phase 6)

**Scope boundary:** everything in §1-8 above is locked and governs
Phases 0-5 (training a real model from scratch under the 6GB VRAM
constraint). Nothing in this section or below changes those decisions.
This section specifies what happens *after* a validated checkpoint
exists (ROADMAP.md Phase 6 dependency: "Phase 5 complete, model
quality validated as meaningfully above baseline").

**Vector index:** FAISS, `IndexFlatIP` for exact nearest-neighbor
search at this dataset scale (~30k images). Flat index chosen over
`IndexIVFFlat`/HNSW because at 30k vectors of dimension 256, brute-
force cosine similarity is still sub-millisecond and exact — the
approximate-index tradeoff (recall loss for speed) has no upside
until the corpus is orders of magnitude larger. Revisit if
FUTURE_IDEAS.md's "larger datasets" item is pursued.

### 9.1 Two indices, two index maps, different lengths

The two search directions rank different things, so they need
differently-shaped indices:

| Endpoint | Searches | Rows | Index map |
|---|---|---|---|
| `POST /search/text` | image index | one per **unique image** (3,179) | `image_samples.json` |
| `POST /search/image` | text index | one per **caption** (15,895) | `caption_samples.json` |

This is stated explicitly because getting it wrong was not hypothetical.
The builder originally emitted one image embedding per *(image,
caption)* pair, putting 15,895 rows — five identical vectors per
picture — in the image index. A single top-10 could return the same
photo five times, so effective result diversity was closer to top-2.

Both indices also shared one `sample_metadata.json` keyed by pair
position, so an image-index hit was resolved against a caption-indexed
list and could return a filename belonging to a different image.

`save_indices()` now refuses to write when either map's length disagrees
with its index, and the app refuses to load a mismatched pair, so a
half-rebuilt index directory fails loudly rather than serving results
with metadata from a previous build. See `docs/KNOWN_ISSUES.md` §2.

### 9.2 Serving must preprocess exactly as training did

An uploaded image and a typed query have to travel the same path the
indexed data did, or the query embedding lands somewhere else in the
space than everything it is compared against. Two skews existed:

- The image router re-declared its own `Resize/CenterCrop/Normalize`
  chain with the constants inline, rather than calling
  `get_eval_transforms(configs/data.yaml)`. Any change to the config
  would have desynchronized serving silently.
- The text router padded to the query's own length while training padded
  every caption to a fixed 77 tokens.

Both now read `configs/serving.yaml`, which is pinned to
`configs/data.yaml`. The tokenizer is also baked into the Docker image,
since loading it on first request made the container depend on outbound
network access at query time.

**API layer:** FastAPI, chosen over Flask for native async support,
automatic OpenAPI schema generation, and Pydantic-based request/
response validation — which matters here because both image uploads
and text queries need distinct, strictly-typed request schemas.

```
Client (web UI)
      │  HTTP (multipart/form image upload, or JSON text query)
      ▼
┌─────────────────────────────┐
│      FastAPI application     │
│  ┌─────────────────────────┐ │
│  │ /search/text             │ │  text → tokenize → text encoder →
│  │ /search/image             │ │  image → transform → image encoder →
│  │                           │ │  → shared embedding → FAISS query
│  └─────────────────────────┘ │
│  Model loaded once at startup │  (checkpoint from Phase 4, not
│  (torch.no_grad() inference)  │   reloaded per-request)
└──────────────┬────────────────┘
               │
               ▼
        FAISS index (in-memory,
        loaded from a serialized
        index file built offline
        from Phase 5's embeddings)
```

**Inference-time constraints:** the same RTX 4050 is assumed for local
serving; inference is far cheaper than training (no backward pass, no
memory queue, batch size of 1 per request), so this is not VRAM-bound
the way training is. CPU-only inference (`torch.device("cpu")`) is a
documented fallback for deployment targets without a GPU.

**Why the model is loaded once, not per-request:** loading a
checkpoint and moving it to device has non-trivial latency; doing this
inside the request handler would make every query slow. The model is
loaded at FastAPI startup (`@app.on_event("startup")`) and held in
application state.

---

## 10. Frontend Architecture (Phase 6-7)

**Stack:** React + TypeScript + Tailwind CSS.

**Why React over a server-rendered template:** the demo needs
interactive, stateful UI (live search-as-you-type, image upload
preview, ranked result grids) — a good fit for a component-based
client-side framework. TypeScript is used over plain JS for the same
reason it's used anywhere: catching integration bugs between the
frontend and the FastAPI response schema at compile time rather than
in the browser console.

**Structure:**
```
frontend/
├── src/
│   ├── components/
│   │   ├── SearchBar.tsx        → text query input
│   │   ├── ImageUploader.tsx    → drag-and-drop image query
│   │   ├── ResultGrid.tsx       → ranked retrieval results
│   │   └── EmbeddingExplorer.tsx→ optional: 2D projection (UMAP/t-SNE)
│   │                              of the embedding space for the
│   │                              portfolio demo (Phase 7)
│   ├── api/
│   │   └── client.ts            → typed fetch wrapper around the
│   │                              FastAPI endpoints
│   ├── types/
│   │   └── search.ts            → TypeScript types mirroring the
│   │                              Pydantic response schemas exactly
│   └── App.tsx
├── package.json
└── tailwind.config.ts
```

**Why types are mirrored, not shared:** FastAPI/Pydantic and React/
TypeScript don't share a type system natively. The alternative
(generating TypeScript types from the OpenAPI schema via a codegen
tool) is a legitimate upgrade path once the API stabilizes — noted in
FUTURE_IDEAS.md rather than adopted upfront, since hand-written types
are sufficient for a single-developer project at this stage and avoid
adding a codegen build step before it's earned its keep.

---

## 11. Deployment Architecture (Phase 7)

**Containerization:** Docker, one image for the FastAPI backend
(includes the model checkpoint and FAISS index as build artifacts or
mounted volumes — checkpoint size determines which), a separate static
build for the React frontend served via Nginx or a static host.

**Why two images, not one:** the backend has GPU/CPU + PyTorch
dependencies; the frontend is a static asset bundle after `npm run
build`. Coupling them into one image would force every frontend-only
change to rebuild a multi-gigabyte PyTorch image.

**CI:** GitHub Actions. Two workflows:
- `test.yml` — runs `pytest` (CLAUDE.md §4's testing gate) and frontend
  type-checking (`tsc --noEmit`) on every push/PR. This is the
  automated enforcement of the "never break existing tests" rule.
- `build.yml` — builds both Docker images on merge to `main`, tagged
  with the commit SHA.

**Why GitHub Actions over an alternative CI:** the repo is already on
GitHub; no separate CI account/integration needed, and it's free for
public repos, which matters for a portfolio project.

**What's explicitly out of scope for now:** Kubernetes, managed cloud
GPU inference endpoints, autoscaling. These are named in
FUTURE_IDEAS.md as scaling-roadmap items, not committed to — the
realistic Phase 7 target is a single Docker Compose file running both
containers on one machine (or a single cheap VM), consistent with
ROADMAP.md's "Realistic Success Definition."

---

## 12. Experiment & Monitoring Architecture

**Experiment tracking:** Weights & Biases (per CLAUDE.md §5), logging
loss, temperature, embedding norm/variance, and GPU memory per step
during Phase 3/3.5/4. TensorBoard remains a documented fallback for
fully offline/no-account use.

**Model registry (lightweight):** checkpoints are saved with a
metadata sidecar (`checkpoint_metadata.json`: config hash, epoch,
val Recall@10, git commit SHA) so any checkpoint in `checkpoints/` is
traceable back to the exact code and config that produced it, without
adopting a full model-registry tool prematurely (see FUTURE_IDEAS.md).

**Monitoring in production (Phase 7):** basic request logging
(latency, query type, result count) via FastAPI middleware, written
through the same `logging` setup as training (`utils/logging_config.py`)
for consistency — not a separate observability stack, since query
volume for a portfolio demo doesn't justify one yet.

---

## 13. Updated Repository Structure

Extends §7 above (unchanged) with the Phase 6/7 additions:

```
vectormind/
├── src/vectormind/       → §7 (training/model code — unchanged, locked)
├── backend/
│   ├── app.py                → FastAPI app, startup model loading
│   ├── routers/
│   │   ├── text_search.py
│   │   └── image_search.py
│   ├── schemas.py             → Pydantic request/response models
│   └── index_builder.py       → offline script: embeddings → FAISS index
├── frontend/             → §10
├── deployment/
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
│   └── docker-compose.yml
├── .github/workflows/
│   ├── test.yml
│   └── build.yml
├── tests/                → §7 (mirrors src/vectormind/; backend/ gets
│                            its own tests/backend/ once Phase 6 starts)
└── configs/               → §7 (unchanged)
```

---

## 14. Updated Key Design Tradeoffs (Serving Layer)

| Decision Point | Chosen | Rejected | Why |
|---|---|---|---|
| Vector index | FAISS `IndexFlatIP` | IVF/HNSW approximate index | Exact search is still fast at 30k vectors; approximate indexing has no benefit yet at this scale |
| API framework | FastAPI | Flask | Native async, automatic schema validation, matches typed-request needs for image/text search |
| Frontend framework | React + TypeScript | Server-rendered templates (Jinja2) | Interactive, stateful search UI needs a component model; TS catches schema-drift bugs at compile time |
| Containerization | Two Docker images (backend/frontend) | One combined image | Decouples frontend rebuilds from the multi-GB PyTorch backend image |
| CI | GitHub Actions | Jenkins / CircleCI | Already on GitHub; free for public repos; no separate integration needed |
| Deployment target | Single-machine Docker Compose | Kubernetes / managed cloud endpoints | Matches realistic portfolio-project scope; K8s named as a future scaling item, not a current need |

---

*This document must be updated whenever any decision above changes.
Do not let this file drift out of sync with the actual implementation.*