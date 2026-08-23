# PROJECT_STATUS.md — VectorMind

**This is a live document.** Update it at every milestone, not just at
phase boundaries. Where this file and ROADMAP.md disagree, ROADMAP.md's
per-phase acceptance criteria win — fix this file, don't reinterpret
those.

---

## Project Name
VectorMind

## Current Phase
Phase 7 — Deployment & Portfolio Polish (per ROADMAP.md)

## Current Stage
Phase 7 artifacts are **written but unexecuted**. `deployment/`
(Dockerfiles, compose, nginx) and `.github/workflows/` exist as files;
no image has been built and no CI run has occurred. Phases 0 through 6.5
are complete and verified.

## Overall Completion
Roughly **85%** across ROADMAP.md's 8 phases (0 through 7). Phases 0–6.5
are done; Phase 7 is authored but not validated, and its "deployed demo
reachable via a public URL" deliverable is not met.

## Current Status
- **Best Checkpoint:** Epoch 7, Step 7944 (`checkpoints/train/best_model.pt`)
- **Test Recall@1 (I2T):** 4.62% (147× chance)
- **Test Recall@5 (I2T):** 13.43% (85× chance)
- **Test Recall@10 (I2T):** 19.63% (62× chance)
- **Test Recall@10 (T2I):** 15.09% (48× chance)
- **Val→Test Gap (R@10):** −0.60pp (reasonable generalization)
- **Embedding health:** **anisotropic — matched/unmatched separation 0.094** (Phase 3.5 reference: 0.964). Earlier reports labelled this HEALTHY; see `docs/KNOWN_ISSUES.md` §1.
- **Total tests:** 345 passing (345/345 as of 2026-08-23, after the missing `tensorboard` dependency was added)

## Completed Work
- [x] **Phase 0 — Project Setup:** Repo structure, VRAM profiling, core docs
- [x] **Phase 1 — Data Pipeline:** Flickr30k loading, transforms, tokenizer, dataset, splitter
- [x] **Phase 2 — Model Architecture:** Image encoder, text encoder, projection heads, dual-encoder
- [x] **Phase 3 — Training Infrastructure:** Loss, memory queue, train loop, checkpointing, logging
- [x] **Phase 3.5 — Sanity Check:** Overfit tiny subset (100% Recall@1, separation 0.964)
- [x] **Phase 4 — Baseline Training:** 8 epochs, Epoch 7 best, memory queue improved R@10 by 18.2%
- [x] **Phase 5 — Evaluation:** Test set metrics, embedding diagnostics, qualitative analysis
- [x] **Phase 6 — Serving:** FAISS index builder, FastAPI app, text/image search endpoints
- [x] **Phase 6.5 — Frontend:** React + TypeScript + Tailwind, production static serving, smoke test

## In Progress
- [ ] **Phase 7 — Deployment:** Dockerfiles, compose, and CI workflows are written but have never been run. `requirements.txt` was missing eight imported packages until 2026-08-23, so neither could have succeeded. Both need an actual green run before this phase is closed.
- [ ] Public deployed demo — not started.

## Key Achievements
1. Model trained from scratch, no pretrained CLIP weights anywhere in the pipeline
2. Test Recall@10 = 19.63% (62× chance), with an honest val→test gap of 0.6pp
3. Phase 3.5 gate enforced before any full training run, and it caught nothing precisely because the pipeline was correct
4. Memory-queue ablation is a real, measured result: +18.2% relative R@10
5. 345 tests across data, model, training, evaluation, and serving layers
6. Full serving stack: FAISS + FastAPI + typed React client
7. Two real bugs found and documented, not hidden: gradient-norm logging after `zero_grad()`, and a `_, loader, _` destructuring bug that made "test" metrics silently be val metrics

## Known Problems
Tracked in `docs/KNOWN_ISSUES.md` — 10 open entries. The four that
matter most:
1. Embedding space is severely anisotropic; reports call it healthy.
2. The image FAISS index holds 5 duplicate vectors per image (15,895 vectors, 3,180 unique).
3. `backend/` imports `src.vectormind.*` while `src/` imports `vectormind.*` — the backend Docker image cannot start.
4. ~90 of 132 commits are empty commits with fabricated messages (CLAUDE.md §7 violation).

## Documentation
- `reports/phase5_test_metrics.json`, `reports/phase5_val_metrics.json`
- `reports/phase5_embedding_diagnostics.json` — **contains numbers that do not reproduce; see KNOWN_ISSUES §8**
- `reports/phase5_qualitative_analysis.md`, `reports/phase5_final_report.md`
- `reports/phase4_final_report.md`, `reports/baseline_analysis.md`
- `docs/DESIGN_DECISIONS.md`, `docs/DEBUGGING_STORY.md`, `docs/KNOWN_ISSUES.md`
- `TRAINING_LOG.md`, `EXPERIMENTS.md`

## Environment Status
- Python 3.12.10
- PyTorch 2.13.0+cu126
- Node.js 22.17.1, npm 10.9.2
- RTX 4050 laptop GPU, 6GB VRAM
- Pinned working set: `requirements.lock.txt` (70 packages, UTF-8)

## Repository Status
Phases 0–6.5 are committed and pushed to `origin/main`. **Phase 7 work
is uncommitted** — `deployment/`, `.github/`, `docs/DESIGN_DECISIONS.md`,
and `docs/DEBUGGING_STORY.md` are untracked, alongside uncommitted edits
across 19 files.

## Last Updated
2026-08-23
