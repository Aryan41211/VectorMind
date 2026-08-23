# Design Decisions — VectorMind

This document records the key architectural decisions made during
development, why they were made, and what alternatives were considered.
Intended for portfolio readers and interviewers.

---

## 1. From-Scratch Training (No Pretrained Weights)

**Decision:** Train the dual-encoder entirely from scratch on Flickr30k,
without loading pretrained CLIP or OpenCLIP weights.

**Why:** The project's goal is to demonstrate ML engineering
competence — data pipeline, training loop, evaluation, serving — not
to fine-tune someone else's model. Using pretrained weights would
reduce this to a transfer-learning demo, which is useful but doesn't
show the same depth of understanding.

**Tradeoff:** Recall is far lower than pretrained CLIP (~20% R@10 vs
~80%+ for CLIP). This is expected and documented honestly in the demo.

**Alternatives considered:**
- Fine-tune pretrained CLIP: easier, better metrics, but less
  educational value and wouldn't demonstrate the full stack.
- Train from scratch on LAION: larger dataset, better metrics, but
  requires significantly more compute than a laptop GPU.

---

## 2. Dual-Encoder (CLIP-Style) Architecture

**Decision:** Use separate image and text encoders with projection
heads into a shared embedding space, trained with InfoNCE contrastive
loss.

**Why:** The dual-encoder is the standard architecture for
embedding-based retrieval. It decouples encoding (done once per
item) from retrieval (FAISS nearest-neighbor), enabling sub-millisecond
search at query time.

**Alternatives considered:**
- Cross-encoder (concatenate image+text, classify): better accuracy
  but O(n) at query time — unusable for retrieval.
- Late-fusion models: more complex, harder to explain, no clear
  benefit at this scale.

---

## 3. ResNet-50 Image Encoder (Not ViT)

**Decision:** Use a ResNet-50 backbone for the image encoder instead
of a Vision Transformer.

**Why:** ResNet-50 is well-understood, has stable training dynamics,
and fits comfortably in 6GB VRAM. ViTs require more data and careful
regularization to train from scratch — risky at 31k images.

**Tradeoff:** ViTs ultimately achieve better representation quality
at scale, but ResNet-50 is more reliable for a from-scratch training
run on limited data.

---

## 4. Contrastive Loss with Learnable Temperature

**Decision:** Use InfoNCE (softmax cross-entropy over similarity
matrix) with a learnable temperature parameter (log-scale).

**Why:** A fixed temperature requires manual tuning per dataset size
and batch size. Learning it lets the model adapt the sharpness of the
similarity distribution during training. The log-scale ensures the
temperature stays positive without clipping.

---

## 5. Memory Queue for Hard Negatives — reversed 2026-08-24

**Original decision:** maintain a queue of recent embeddings from the
previous N batches as additional negative examples.

**Original reasoning:** with batch size 128, each batch provides only
127 negatives per query. A 4,096-entry queue exposes the model to a much
larger negative pool. Recorded as improving R@10 by 18.2%.

**Current decision: the queue is disabled.** It does not improve
embedding quality here — it destroys it.

A controlled A/B, both arms resuming from the same checkpoint with the
queue as the only variable:

| Epoch 7 from `epoch_006.pt` | Queue active | Queue inactive |
|---|---|---|
| Val R@10 | 10.51% | **19.63%** |
| Separation | 0.062 | **0.322** |
| Logit scale | 67.6 | **18.6** |

**Why the original reasoning was wrong.** It counted negatives without
asking whether they were *comparable*. MoCo pairs its queue with a
momentum encoder — a slow EMA copy that produces the queued keys — and
that is the entire mechanism keeping an embedding written 32 batches ago
meaningful against the current encoder. This implementation borrowed the
queue and not the momentum encoder, so the extra 4,096 negatives are
outputs of a model that no longer exists, outnumbering the valid
in-batch ones 32 to 1.

Faced with thousands of mismatched stale negatives, the cheapest way to
lower the loss is to sharpen the similarity distribution rather than
improve the representation — and an unbounded logit scale is the
cheapest way to sharpen. The scale runs away, the embedding space
collapses into a narrow cone, and recall falls a few epochs later. Phase
4 recorded that final step as an unrelated "temperature overgrowth"
problem.

**Why the 18.2% figure held up for weeks.** It was one epoch measured
immediately after activation, before the collapse reached the metric.
It was never a controlled comparison — `--no-queue` substituted a
size-1 stub queue that `load_checkpoint` rejected against a real
checkpoint, so the two arms could not share a starting state. And no
embedding-health metric existed, so Recall@K was the only signal, which
is exactly the signal that lags a collapse.

**The transferable lesson:** when borrowing a technique, check which
part of it is load-bearing. The queue is the visible half of MoCo; the
momentum encoder is the half that makes it work.

**Retained implementation detail:** embeddings are detached before
enqueueing, so no gradient flows through stale entries. That was always
correct — it just was not sufficient.

**If revisited:** implement a momentum encoder and re-run the A/B. See
[FUTURE_IDEAS.md](FUTURE_IDEAS.md), [KNOWN_ISSUES.md](KNOWN_ISSUES.md)
§11, and [EXPERIMENTS.md](EXPERIMENTS.md) 006.

---

## 6. FAISS Flat Index (Not IVFFlat/HNSW)

**Decision:** Use FAISS `IndexFlatIP` (brute-force inner product
search) instead of approximate indices.

**Why:** At 30k vectors of dimension 256, brute-force search is
sub-millisecond and exact. The approximate-index tradeoff (recall
loss for speed) has no upside until the corpus is orders of magnitude
larger.

---

## 7. Two Docker Images (Not One)

**Decision:** Separate `backend.Dockerfile` (Python + PyTorch) and
`frontend.Dockerfile` (Node → nginx) instead of a single image.

**Why:** The backend has GPU/CPU + PyTorch dependencies (~2GB+). The
frontend is a static asset bundle after `npm run build` (~200KB).
Coupling them would force every frontend-only change to rebuild the
multi-gigabyte PyTorch image.

**Tradeoff:** Two images to manage, but Docker Compose handles
orchestration cleanly.

---

## 8. Model Checkpoint as Volume (Not Baked Into Image)

**Decision:** Mount `checkpoints/` and `backend/indices/` as Docker
volumes instead of COPYing them into the image.

**Why:** The checkpoint is ~292MB. Baking it into the image means
every code change triggers a 292MB layer rebuild. Mounting keeps the
image small (~500MB with PyTorch) and lets you swap checkpoints
without rebuilding.

---

## 9. React + TypeScript (Not Vanilla JS)

**Decision:** Use React with TypeScript for the demo frontend.

**Why:** TypeScript catches integration bugs between the frontend
and the FastAPI response schema at compile time. React's component
model fits the interactive, stateful UI (search-as-you-type, image
upload preview, result grids).

**Alternatives considered:**
- Vanilla JS: simpler, but no type safety — integration bugs surface
  in the browser console, not at build time.
- Vue/Svelte: equally valid, but React has the largest ecosystem
  and is more recognizable to interviewers.

---

## 10. Vite (Not CRA or Webpack)

**Decision:** Use Vite as the frontend build tool.

**Why:** Vite's dev server is significantly faster than CRA/Webpack
(HMR in <100ms vs seconds). The production build uses Rollup, which
produces smaller bundles than Webpack. For a portfolio project where
developer experience matters, Vite is the clear choice.

---

*Last updated: Phase 7 (ROADMAP.md)*
