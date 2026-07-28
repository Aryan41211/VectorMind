# VectorMind

A research-grade multimodal semantic search platform that learns a
shared embedding space for images and text using contrastive learning
— trained **entirely from scratch**, no pretrained CLIP/OpenCLIP
weights, on a single 6GB laptop GPU.

> This is a portfolio/research project, not a reproduction of the CLIP
> paper. It is not trying to match published CLIP numbers — it's
> trying to demonstrate a correct, debuggable, well-documented ML
> engineering pipeline end to end. See [ROADMAP.md](./ROADMAP.md)'s
> "Realistic Success Definition" for the honest framing.

---

## Vision

Most public CLIP-style demos fine-tune or wrap an already-pretrained
model. VectorMind instead builds both towers — the image encoder and
the text encoder — from scratch, under a hard hardware constraint
(RTX 4050, 6GB VRAM), and treats every architectural decision as
something that has to be *justified against what's actually trainable
at this scale*, not against what a 400M-pair-scale system would do.
See [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md) for the full motivation.

## Features

- **Dual-encoder architecture** — independently swappable image tower
  (small CNN), text tower (small Transformer), and projection heads
- **Contrastive training from scratch** — symmetric InfoNCE loss with
  a learnable temperature, MoCo-style memory queue to decouple negative
  sample count from the physical batch size on 6GB VRAM
- **Cross-modal retrieval** — image → text and text → image search
- **Full serving stack** (Phase 6+) — FAISS vector index, FastAPI
  backend, React/TypeScript frontend, Dockerized deployment with CI

## Architecture Overview

```
Image ──▶ Image Encoder ──▶ Projection Head ──▶┐
                                                  ├──▶ Shared Embedding Space ──▶ Contrastive Loss
Text  ──▶ Text Encoder  ──▶ Projection Head ──▶┘         (InfoNCE, learnable temperature)
```

Full diagrams and design rationale for every box above (and the
serving/frontend/deployment layers built on top of a trained model)
live in [ARCHITECTURE.md](./ARCHITECTURE.md).

## Technology Stack

| Layer | Technology |
|---|---|
| Model / training | PyTorch, mixed precision, from-scratch CNN + Transformer |
| Data | Flickr30k (public, ~30k images / 5 captions each) |
| Experiment tracking | Weights & Biases / TensorBoard |
| Vector index | FAISS |
| Backend API | FastAPI |
| Frontend | React, TypeScript, Tailwind CSS |
| Deployment | Docker, GitHub Actions |

Full rationale and alternatives considered for each: [TECH_STACK.md](./TECH_STACK.md).

## Demo

_Coming in Phase 6.5/7 — screenshots and a live link will go here once
the frontend demo and deployment are built. See
[ROADMAP.md](./ROADMAP.md) for current phase status._

## Installation

```bash
git clone <this-repo-url>
cd vectormind
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

GPU setup: install the PyTorch build matching your installed CUDA
version (check `nvidia-smi`, then use the correct index URL from
https://pytorch.org/get-started/locally/ — do not assume a `cu1xx` tag
without checking).

### Environment Verification

Run the verification script to confirm all dependencies are correctly
installed and CUDA is available:

```bash
python scripts/verify_env.py
```

Expected output:
```
============================================================
VectorMind Environment Verification
============================================================

Results:
------------------------------------------------------------
  [PASS] Python 3.12.10 (need 3.12.x)
  [PASS] PyTorch 2.13.0+cu126 | CUDA: True | Device: NVIDIA GeForce RTX 4050 Laptop GPU | VRAM: 6.0 GB
  [PASS] AMP (torch.autocast) works on CUDA
  [PASS] torchvision 0.28.0+cu126
  [PASS] transformers 5.14.1
  [PASS] tokenizers 0.22.2
  [PASS] faiss 1.14.3
  [PASS] cv2 5.0.0
  [PASS] PIL 12.3.0
  [PASS] fastapi 0.140.7
  [PASS] uvicorn 0.51.0
  [PASS] yaml 6.0.3
  [PASS] wandb 0.28.1
  [PASS] pytest 8.4.2
------------------------------------------------------------
ALL CHECKS PASSED
```

## Quick Start

```bash
# Run the test suite
pytest tests/ -v

# Profile the max VRAM-safe batch size on your GPU (Phase 0.2)
python scripts/profile_vram.py --config configs/profiling.yaml
```

### VRAM Profiling

The profiling script (`scripts/profile_vram.py`) empirically determines
the maximum batch size that fits in your GPU's VRAM under mixed
precision. It uses encoder dimensions representative of the planned
Phase 2 architecture (ARCHITECTURE.md §2-4) and runs a
contrastive-loss-shaped forward/backward pass.

Configuration is in `configs/profiling.yaml`:
- `use_amp: true` — mixed precision (required per CLAUDE.md §9)
- `min_batch_size` / `max_batch_size` — search bounds
- `warmup_iters` / `measure_iters` — iterations for stable measurement
- `vram_safety_margin_fraction: 0.10` — 10% headroom for memory queue,
  dataloader pinned memory, and fragmentation

Output:
- Console log with each batch size attempt
- `logs/vram_profile_results.json` — structured results
- `logs/profile_vram.log` — detailed log

**Current measured result (RTX 4050 Laptop, 6.44 GB VRAM):**
- Max safe batch size: **256**
- Peak VRAM at batch 256: **4.99 GB**
- Search method: exponential then binary search, 5.2 GB ceiling, AMP enabled
- Tensor shapes: Image `[B, 3, 224, 224]`, Text `[B, 77]`
- Encoder dims: image 512, text 256, shared embedding 256

See [ARCHITECTURE.md §6](./ARCHITECTURE.md#6-vram-constrained-batch-strategy-critical)
for full details and design rationale.

Data pipeline, training, and serving entrypoints will be documented
here as each phase lands — see [ROADMAP.md](./ROADMAP.md) for what's
implemented today versus planned.

## Repository Structure

```
vectormind/
├── src/vectormind/    → model, data, training, evaluation code
├── backend/            → FastAPI serving layer (Phase 6)
├── frontend/           → React demo UI (Phase 6.5)
├── deployment/         → Docker, docker-compose (Phase 7)
├── configs/            → all hyperparameters (never hardcoded)
├── scripts/            → one-off tooling (e.g. VRAM profiling)
└── tests/              → mirrors src/vectormind/
```

Full ownership/responsibility breakdown: [FOLDER_STRUCTURE.md](./FOLDER_STRUCTURE.md).

## Development Roadmap

See [ROADMAP.md](./ROADMAP.md) for the complete, living phase-by-phase
plan (Phase 0 through Phase 7), including dependencies, acceptance
criteria, and current status per phase.

## Screenshots

_Placeholder — will be added once the Phase 6.5 frontend exists._

## Contributing

This is currently a solo portfolio project. If you're another engineer
picking this up: read
[CLAUDE.md](./CLAUDE.md) first (the permanent engineering rules),
then [DEVELOPMENT_GUIDE.md](./DEVELOPMENT_GUIDE.md) for the expected
workflow, then [ARCHITECTURE.md](./ARCHITECTURE.md) and
[ROADMAP.md](./ROADMAP.md) for current state. Conventional Commits
required (`feat:`, `fix:`, `docs:`, etc. — see CLAUDE.md §7).

## License

_Not yet chosen — add before any public release beyond portfolio use._