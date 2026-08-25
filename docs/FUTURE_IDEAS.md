# FUTURE_IDEAS.md — VectorMind

A research backlog. Nothing here is committed to, scheduled, or a
dependency of Phases 0-7 in ROADMAP.md. Items are here specifically so
they're captured *without* pulling focus from the current plan — see
PROJECT_RULES.md #27: serving/deployment scope extends but never
contradicts the core training constraints, and none of the below
should be pursued in a way that does.

---

## Multilingual Retrieval

Extend the text encoder to handle multiple languages. Would require a
multilingual tokenizer (still preprocessing-only, per
ARCHITECTURE.md §3's principle) and either a multilingual caption
dataset or a translation-augmented version of Flickr30k. Open question:
whether a single shared text encoder handles multiple languages well
at this model scale, or whether quality degrades — worth a small
experiment before committing further.

## Audio Retrieval

A third modality/tower (audio → shared embedding space), following the
same dual-encoder decoupling principle (ARCHITECTURE.md §2) so it can
be added without touching the existing image/text towers. Would need
its own dataset (e.g. audio captioning data) and its own VRAM profiling
pass — the 6GB constraint applies to any additional tower too.

## Video Retrieval

Harder than audio: requires either frame-sampling + the existing image
encoder (cheap, but loses temporal information) or a video-specific
architecture (expensive, likely infeasible from scratch on 6GB VRAM).
If pursued, frame-sampling is the realistic starting point given this
project's hardware constraint.

## Medical Imaging

Domain-specific retrieval (e.g. radiology image ↔ report). Would
require a domain-appropriate dataset (with real licensing/privacy
considerations, unlike Flickr30k's public general-purpose images) and
likely a different image encoder inductive bias than natural photos.
Not a small extension — effectively a separate project sharing this
one's architecture pattern.

## Satellite Imagery

Similar to medical imaging: different image statistics (multi-band,
large-scale spatial structure) than Flickr30k's natural photos; would
likely need a different image encoder design, not just a retrained
version of the current one.

## Hybrid Retrieval

Combining dense (embedding-based) retrieval with sparse
(keyword/BM25-style) retrieval, common in production search systems.
Would sit in the `backend/` serving layer (ARCHITECTURE.md §9),
re-ranking or fusing FAISS results with a sparse index — doesn't
require any change to the trained model itself.

## Knowledge Distillation

Train a smaller/faster model to mimic the trained VectorMind model's
embeddings, for cheaper inference. Interesting mainly once a validated
Phase 4/5 checkpoint exists — distilling an unvalidated model isn't
worth doing.

## Quantization

Post-training int8 quantization for cheaper CPU inference
(ARCHITECTURE.md §9's CPU fallback). Straightforward with
`torch.quantization`; worth doing once inference latency is actually
measured (ROADMAP.md Phase 6 acceptance criteria) and shown to
benefit from it — not preemptively.

## LoRA-Style Fine-Tuning Experiments

Once a from-scratch base checkpoint exists, experimenting with
LoRA-style low-rank adaptation for downstream fine-tuning tasks (e.g.
adapting to a new small dataset without full retraining) would be an
interesting extension of the "trained from scratch" story: the base
model is still from scratch, but adaptation techniques are explored on
top of it.

## Momentum Encoder for the Memory Queue

**Status:** the highest-value open research item in this project.

The memory queue was disabled in Phase 4b because it collapsed the
embedding space — R@10 10.51% with it against 19.63% without, from an
identical checkpoint (KNOWN_ISSUES.md §11, EXPERIMENTS.md 006). The
cause is that this implementation borrowed MoCo's queue without MoCo's
momentum encoder.

A momentum encoder is a second copy of both towers updated as an
exponential moving average of the live weights:

```
theta_momentum = m * theta_momentum + (1 - m) * theta_live      # m ~ 0.999
```

Queued keys are produced by that slow copy, so an embedding written 32
batches ago is still comparable to what the current encoder produces.
That comparability is what makes thousands of extra negatives useful
rather than noise.

**Cost:** a second copy of both towers in VRAM. At 23.9M parameters that
is ~96MB in fp32 — affordable within 6GB, since the momentum encoder
needs no gradients or optimizer state. The forward pass for queued keys
is extra compute, though it can reuse the batch already loaded.

**Why it matters here:** it is the difference between two claims. "A
memory queue does not work at this scale" is not supported by the
evidence. "A memory queue does not work *without the mechanism that
makes it work elsewhere*" is — and only an implementation with a
momentum encoder can distinguish them. Until that experiment is run,
this project has tested its own implementation rather than MoCo.

**Acceptance criterion:** re-run the Experiment 006 A/B a third time,
three arms — no queue, queue, queue with momentum encoder — all from a
shared checkpoint, reporting separation alongside Recall@K.

---

## Uniformity From Scratch, Not Resumed

**Status:** open, and the cheapest remaining representation experiment.

The shipped model reaches a HEALTHY embedding space with a Wang & Isola
uniformity term at weight 0.2 (ARCHITECTURE.md §5.2, EXPERIMENTS.md
009), but it got there by **resuming** an InfoNCE-only checkpoint into
the new objective. That forces the model to unlearn a space it has
already committed to, and the evidence that it was still doing so is
that the w=0.2 run's best epoch is 14 where the baseline's is 12 — it
was still recovering when the run stopped.

Training from scratch with the term enabled removes the unlearning
entirely. The plausible outcome is that the 0.13pp image→text cost
disappears, and possibly reverses; the honest outcome is that nobody
knows, because it has not been run.

**Why it is not on the critical path:** the current model already passes
every health threshold and is better than the InfoNCE-only baseline on
text→image retrieval, which is the direction the demo exercises. This is
an improvement left on the table, not a defect.

**Cost:** one full training run, ~15 epochs on the 6GB laptop GPU.

Two smaller questions belong with it: whether w=0.2 is a peak or a
plateau (w=0.15 and w=0.3 would say), and whether w=0.1's regression is
real or an artifact of its shorter run.

## Distributed Training

Not relevant at this project's scale (single 6GB GPU, ~30k images) —
listed for completeness only. Would only become relevant if
FUTURE_IDEAS.md's "larger datasets" item were pursued at a scale
requiring multiple GPUs.

## Larger Datasets

Scaling beyond Flickr30k (e.g. Flickr30k + additional public
caption datasets, or a larger public dataset entirely) would directly
stress-test the current negative-sampling strategy
(ARCHITECTURE.md §6) and likely require revisiting the FAISS index
choice (TECH_STACK.md's FAISS section already flags `IndexIVFFlat`/HNSW
as the likely next step if corpus size grows by orders of magnitude).

## Approximate Nearest-Neighbor Improvements

Benchmark `IndexIVFFlat` or HNSW against the current `IndexFlatIP`
once/if corpus size actually motivates it (see ARCHITECTURE.md §9 —
explicitly not needed at 30k vectors).

## Online Indexing

Currently the FAISS index is built offline once (`backend/index_builder.py`)
from a fixed embedding set. Supporting incremental additions to the
index without a full rebuild would matter if VectorMind ever needed to
index new images without downtime — not a current requirement.

## Model Registry

The lightweight checkpoint-metadata-sidecar approach
(ARCHITECTURE.md §12) is sufficient for a single-developer project
with a handful of checkpoints. A real model registry (MLflow Model
Registry or similar) would matter if checkpoint volume or team size
grew enough that manual sidecar-file tracking became unwieldy.

## Feature Store

Not relevant unless this evolves into a multi-model or
multi-downstream-consumer system, which it currently is not. Listed
for completeness.

## Experiment Dashboard

W&B's own dashboard (TECH_STACK.md) already covers this for training
runs. A custom dashboard would only be worth building if W&B's UI
became a genuine limitation, which hasn't happened.

## Cloud Deployment / Scaling Roadmap

Explicitly out of scope for Phase 7 (ARCHITECTURE.md §11): the
realistic target is single-machine/VM Docker Compose. A future scaling
path, if actual usage ever justified it: managed container hosting →
managed GPU inference endpoint → Kubernetes, roughly in that order of
increasing complexity — each step justified by a real, measured need
(traffic, latency, cost), not adopted speculatively.
