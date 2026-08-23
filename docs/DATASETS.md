# DATASETS.md — VectorMind

---

## Current Dataset: Flickr30k

**Source:** originally distributed by the University of Illinois
(Denotation Graph project); commonly mirrored on Kaggle and Hugging
Face Datasets (e.g. `nlphuji/flickr30k`).

**Contents:** ~31,783 images, each with 5 human-written English
captions (~158,915 captions total).

**Why chosen:** explicit hard constraint (CLAUDE.md §1) — public,
well-known, small enough to be tractable on a single 6GB GPU, large
enough to demonstrate non-trivial contrastive learning, and widely
used as an academic benchmark so results are comparable to a known
reference point.

### Licensing — verified, with a flagged discrepancy

The **original/official** distribution states the dataset is provided
**for non-commercial research and/or educational purposes only**, and
that the images themselves remain subject to **Flickr's own Terms of
Use** (the dataset creators do not own the image copyrights) — this is
stated consistently across the official UIUC Denotation Graph page and
the Flickr30K Entities distribution.

**Discrepancy found:** some third-party mirror/marketing sites (e.g.
certain Kaggle-adjacent dataset-catalog pages) describe a Flickr30k
variant as "CC0" (public domain, unrestricted commercial use). This
**contradicts** the original source's non-commercial research/
education restriction. I am not treating the CC0 claim as reliable —
it appears on a third-party aggregator page, not the original
distributors' page, and CC0 is inconsistent with images that remain
subject to Flickr's own ToU.

**Practical implication for this project:** VectorMind is a
non-commercial research/portfolio project, so this is not a live legal
risk today. But **treat the original UIUC/Flickr30K-Entities terms as
authoritative**, not the CC0 claims, and re-verify directly from
whichever specific source you actually download from before any
future commercial use, redistribution, or public dataset re-hosting.
Do not assume a mirror's license tag is correct without checking it
against the original terms.

**Access:** the canonical route requires filling out a form (per the
official Flickr30K Entities distribution instructions) rather than an
unrestricted direct download — record exactly which mirror/access
route was actually used once Phase 1 begins, since that determines
which specific terms apply.

### Preprocessing (Phase 1 scope)

- Image transforms: resize/crop to the configured `image_size`
  (`configs/data.yaml` uses 224), normalization with ImageNet-standard
  mean/std. Train: Resize(256) → RandomCrop(224) → RandomHorizontalFlip →
  ToImage → ToDtype(float32, scale) → Normalize. Eval: Resize(256) →
  CenterCrop(224) → ToImage → ToDtype(float32, scale) → Normalize.
  Uses torchvision v2 transforms API.
- Tokenization: `bert-base-uncased` tokenizer from HuggingFace
  (pretrained BPE, used only for tokenization — no pretrained
  embeddings). Max length 77 tokens, right-padded with pad_token_id.
  Chosen for wide availability, reasonable vocab size (30,522), and
  compatibility with the CLIP-style text encoder design.
- Pairing: each image paired with each of its 5 captions (5 pairs per
  image). This is the standard contrastive learning approach — the
  DataLoader shuffles these naturally.
- Split: train/val/test (0.8/0.1/0.1) by **image** (not caption) —
  all 5 captions for a given image stay in the same split (zero
  leakage). Deterministic with `random_seed: 42`.
- Dataset source: HuggingFace Datasets (`nlphuji/flickr30k`) —
  most accessible mirror, no form required. Cached to
  `data/raw/flickr30k/`.

### Augmentations

**Not yet decided.** Standard options for this scale (random crop,
horizontal flip, color jitter) are reasonable candidates but no
specific augmentation policy has been chosen. TECH_STACK.md notes
`albumentations` as a candidate library if augmentation needs grow
beyond what `torchvision.transforms` covers — not currently justified.

### Evaluation Split

Held-out test split from the same Flickr30k data (ROADMAP.md Phase 5)
— Recall@1/5/10 computed on this split for both image→text and
text→image directions, compared against a random-chance baseline.

---

## Future Datasets

Tied to FUTURE_IDEAS.md's research backlog — **none of these are
committed to**:

| Idea | Dataset need | Status |
|---|---|---|
| Multilingual retrieval | Multilingual captions (translation-augmented Flickr30k, or a native multilingual caption dataset) | Not started; open question on translation quality vs. native data |
| Audio retrieval | An audio-captioning dataset | Not started |
| Video retrieval | A video-captioning dataset (or frame-sampled Flickr-style data as a cheaper starting point) | Not started |
| Medical imaging | A licensed medical image-report dataset — real privacy/licensing considerations, unlike Flickr30k | Not started |
| Satellite imagery | A remote-sensing image-caption dataset | Not started |
| Larger-scale training | A larger public image-caption dataset (e.g. a filtered subset of a larger public corpus) | Not started; would require revisiting the negative-sampling and FAISS-index strategy per ARCHITECTURE.md |

## Dataset Comparison (current vs. commonly cited alternatives)

| Dataset | Size | Why not chosen for this project |
|---|---|---|
| **Flickr30k** (chosen) | ~30k images, 5 captions each | — |
| MS-COCO Captions | ~120k images, 5 captions each | Larger than needed for a 6GB-VRAM from-scratch demonstration; Flickr30k is the more commonly cited "small-scale" benchmark for this exact use case |
| Conceptual Captions (CC3M/CC12M) | 3M-12M image-caption pairs | Far beyond what's tractable to train from scratch on a single 6GB GPU in a reasonable timeframe; designed for large-scale pretraining, not this project's scale |
| Visual Genome | ~108k images, dense region annotations | Built for region-level grounding tasks, not simple whole-image/whole-caption contrastive pairs — a mismatch for this project's task framing |

## Scaling Strategy

If dataset scale ever increases (FUTURE_IDEAS.md "Larger Datasets"):
1. Re-run Phase 0.2-style VRAM profiling — a larger/more diverse
   dataset doesn't by itself change per-batch VRAM usage, but any
   accompanying model-size increase would.
2. Revisit the FAISS index choice (`IndexFlatIP` → `IndexIVFFlat`/HNSW)
   per ARCHITECTURE.md §9's explicit scope note.
3. Revisit whether single-GPU training remains feasible in a
   reasonable timeframe, or whether FUTURE_IDEAS.md's "Distributed
   Training" item needs to move from backlog to active consideration.

Each step above is a real decision to be made *when* the need is
real, not preemptively — consistent with PROJECT_RULES.md's YAGNI
framing (CODING_STANDARDS.md §12).
