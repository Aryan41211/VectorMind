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
   gradient accumulation alone doesn't fix it.
4. **Gradient checkpointing** — fallback if still OOM after the
   above; trades compute for memory by recomputing activations during
   the backward pass instead of storing them.

**Actual batch size and queue size will be determined empirically in
Phase 0.2** by profiling real memory usage on the RTX 4050 — this
document will be updated with the final numbers once measured.

**Validation:** Whether this strategy actually works (i.e. the queue
gives real negative diversity and the model can learn at all under
this batch/queue configuration) is confirmed in ROADMAP.md Phase 3.5
(tiny-subset overfit sanity check) before any full training run. If
Phase 3.5 fails to converge, revisit the numbers in this section
first — it is the most likely root cause.

---

## 7. Repository Structure

```
src/vectormind/
├── data/          → Phase 1: Dataset/DataLoader, transforms, tokenization
├── models/
│   ├── image_encoder.py     → §2
│   ├── text_encoder.py      → §3
│   ├── projection_heads.py  → §4
│   └── vectormind_model.py  → combines towers + heads into one model
├── training/
│   ├── losses.py            → §5 (InfoNCE)
│   ├── memory_queue.py      → §6 (MoCo-style queue)
│   └── train_loop.py        → training loop, mixed precision, accumulation
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

*This document must be updated whenever any decision above changes.
Do not let this file drift out of sync with the actual implementation.*