# PROJECT_CONTEXT.md — VectorMind

The definitive explanation of what VectorMind is, why it exists, and
what it is trying to prove. Read this before ARCHITECTURE.md if you
want the "why" before the "how."

---

## 1. Why This Project Exists

Cross-modal retrieval (CLIP and its successors) is one of the more
consequential ideas in modern ML — it underlies image search, content
moderation, multimodal RAG, and a large fraction of "chat with your
data" products that touch images. Most public demonstrations of this
idea, though, are thin wrappers around an already-pretrained CLIP
checkpoint: load the weights, embed things, done. That's a legitimate
engineering task, but it doesn't demonstrate understanding of *why*
the architecture is shaped the way it is, or what breaks when you
can't just inherit a 400M-image-pair pretrained model.

VectorMind exists to close that gap: build the same category of
system, from nothing, under a real hardware constraint, and document
every decision honestly — including the ones that didn't work on the
first attempt.

## 2. Problem Statement

Given a small, fully from-scratch model (no pretrained weights
anywhere in the pipeline), a modest public dataset (Flickr30k, ~30k
images), and consumer hardware (a single 6GB laptop GPU), can a
dual-encoder contrastive model learn a shared embedding space good
enough for non-trivial cross-modal retrieval — and can the engineering
process that gets there be made legible enough that another engineer
could reproduce it from the documentation alone?

## 3. Motivation

Three things, in order of importance:

1. **Depth over breadth.** Fine-tuning a pretrained model teaches you
   the fine-tuning API. Building the dual-encoder, the contrastive
   loss, and the negative-sampling strategy from scratch teaches you
   *why* CLIP-style systems are shaped the way they are — what
   actually happens if you don't have 32,768-size batches, and what
   you do instead.
2. **Constraint-driven engineering.** A 6GB VRAM ceiling is a forcing
   function. It rules out lazy defaults (huge batch sizes, ViT-from-
   scratch, fp32-everywhere) and forces every choice in
   ARCHITECTURE.md to be justified against the actual hardware, not
   against what a paper with a different budget did.
3. **Portfolio legibility.** A project that only shows a final metric
   is less convincing than one that shows the debugging story: what
   was risky, what was checked before spending compute on it (Phase
   3.5's sanity gate), and what the failure modes would have looked
   like if it hadn't been checked.

## 4. Vision

Short-term vision: a correctly-engineered, from-scratch dual-encoder
model with documented, above-random-chance retrieval performance on
Flickr30k, served through a real API and a minimal web UI, deployed
somewhere reachable.

Long-term vision (see FUTURE_IDEAS.md for the full backlog): the same
architecture and engineering discipline extended to more modalities
(audio, video), larger datasets, and more efficient training/inference
(distillation, quantization) — pursued only if they add real learning
value, not to inflate scope for its own sake.

## 5. Product Philosophy

- **Honesty over impressiveness.** ROADMAP.md's "Realistic Success
  Definition" is a philosophy, not just a section header: report the
  real Recall@10 number next to the random-chance baseline, not a
  cherry-picked qualitative example.
- **The process is part of the deliverable.** Phase 3.5's tiny-subset
  overfit check exists because embedding collapse is a silent failure
  mode — the fact that it was checked *before* spending a full training
  run is as much the point as the final model.
- **Constraints are design inputs, not obstacles to route around.**
  The 6GB VRAM limit shapes the negative-sampling strategy
  (ARCHITECTURE.md §6) rather than being something to complain about.

## 6. Core Concepts

**Semantic search:** retrieving items by meaning rather than exact
keyword/tag match — e.g. the query "a dog running on a beach" should
retrieve photographically distinct images that all depict that scene.

**Shared embedding space:** a single vector space where both an image
and its matching caption map to nearby points, and non-matching pairs
map to distant points. This is what makes cross-modal comparison
possible: image and text embeddings become directly comparable via
cosine similarity once L2-normalized (ARCHITECTURE.md §4).

**Contrastive learning:** training by comparison rather than absolute
labels — pulling matching pairs together and pushing non-matching
pairs apart, rather than predicting a fixed class label. InfoNCE
(ARCHITECTURE.md §5) is the specific contrastive objective used here.

**Representation learning:** the broader idea that a model can learn
useful, general-purpose feature vectors (embeddings) as a byproduct of
solving a proxy task (here, matching images to captions), rather than
those features being hand-engineered.

## 7. Expected Capabilities

At the end of Phase 5 (evaluation), VectorMind should be able to:
- Given a text query, rank a held-out set of images by relevance
- Given an image query, rank a held-out set of captions by relevance
- Report Recall@1/5/10 for both directions, clearly above random
  chance, with documented qualitative failure cases

At the end of Phase 7, additionally:
- Serve those capabilities through a live API and a browser UI
- Be deployed somewhere a third party can actually try it

## 8. Future Evolution

See FUTURE_IDEAS.md for the concrete backlog. In philosophy terms:
future evolution should extend the *same* discipline (constraint-
aware, from-scratch where it teaches something, honestly evaluated) to
new modalities or scales — not abandon that discipline the moment more
compute becomes available.

## 9. Success Criteria

Restating ROADMAP.md's "Realistic Success Definition" as explicit
criteria:

1. Val/test Recall@10 (both directions) is reported alongside the
   random-chance baseline, and is clearly, reproducibly above it.
2. The Phase 3.5 sanity check passed and is recorded as passed before
   any full training run happened.
3. A reader unfamiliar with the project can understand what it does,
   why each major decision was made, and what went wrong along the
   way, from the documentation alone (ROADMAP.md Phase 7 acceptance
   criteria).
4. No pretrained CLIP/OpenCLIP weights were loaded anywhere in the
   model at any point (ARCHITECTURE.md §8) — this is a hard
   correctness criterion, not a preference.
