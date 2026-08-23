# PROJECT_STATUS.md — VectorMind

**This is a live document.** Update it at every milestone, not just at
phase boundaries. Where this file and ROADMAP.md disagree, ROADMAP.md's
per-phase acceptance criteria win — fix this file, don't reinterpret
those.

---

## Project Name
VectorMind

## Current Phase
Phase 4b complete (retrain), Phase 7 (deployment) in progress.

## Current Stage
The Phase 4 checkpoint has been retired and replaced. Its embedding
space had collapsed — separation 0.094 — while the Phase 5 reports
called it HEALTHY. The cause was the memory queue, not the temperature
([KNOWN_ISSUES.md](KNOWN_ISSUES.md) §11). The current checkpoint is
better on every retrieval metric *and* on a space 3.5× better separated.

## Overall Completion
Roughly **90%**. Phases 0–6.5 complete; Phase 4b delivered a better
checkpoint with regenerated reports; Phase 7's deployment artifacts are
written but have never been built or run.

## Current Results

**Checkpoint:** `checkpoints/train/best_model.pt` — epoch 10, step 9930,
learned logit scale 22.10. Every figure below is regenerable with
`python scripts/generate_reports.py`.

### Retrieval (test split: 3,179 images / 15,895 captions)

| Direction | K | Measured | Chance | vs chance |
|---|---|---|---|---|
| image → text | 1 | 6.04% | 0.031% | **192×** |
| image → text | 5 | 16.04% | 0.157% | **102×** |
| image → text | 10 | 23.91% | 0.314% | **76×** |
| text → image | 1 | 5.06% | 0.031% | **161×** |
| text → image | 5 | 14.63% | 0.157% | **93×** |
| text → image | 10 | 21.53% | 0.315% | **68×** |

Chance is the exact complement of drawing K non-relevant items, not the
`k/n` shortcut — see [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §1b for why
every earlier multiple in this project was ~30× too low.

### Embedding health

| Metric | Phase 4 (retired) | Current | Threshold |
|---|---|---|---|
| Separation | 0.094 | **0.330** | > 0.25 ✅ |
| Mean image–image cosine | 0.810 | **0.383** | < 0.5 ✅ |
| ‖mean embedding‖ | 0.900 | **0.619** | < 0.5 ❌ |
| Logit scale | 55.2 → 500+ | **22.1** | clamped at 100 |

**Grade: ANISOTROPIC.** Not healthy. Separation clears its floor so
retrieval is meaningful, but the space still carries a shared
directional component. This is reported rather than rounded up — the
previous checkpoint's report claimed HEALTHY on worse numbers, and that
is the failure this whole audit was about.

### Val → test

Val R@10 23.06%, test R@10 23.91% — the test split scores **higher**, a
+0.84pp gap. Comfortably within noise at this split size, and no sign of
overfitting.

## Tests
- **444** Python tests
- **56** frontend tests (was 0)
- mypy, ruff, tsc and oxlint all clean across the repository

## Completed Work
- [x] **Phase 0 — Setup:** repo structure, VRAM profiling, core docs
- [x] **Phase 1 — Data:** Flickr30k loading, transforms, tokenizer, splitter
- [x] **Phase 2 — Model:** image encoder, text encoder, projection heads
- [x] **Phase 3 — Training:** loss, memory queue, train loop, checkpointing
- [x] **Phase 3.5 — Sanity check:** overfit 100 images (100% R@1, separation 0.964)
- [x] **Phase 4 — Baseline training:** superseded, checkpoint retired
- [x] **Phase 4b — Clamped retrain:** current checkpoint, queue disabled
- [x] **Phase 5 — Evaluation:** re-run against the new checkpoint
- [x] **Phase 6 — Serving:** FAISS index, FastAPI, both search endpoints
- [x] **Phase 6.5 — Frontend:** React + TypeScript, themed, tested, accessible

## In Progress
- [ ] **Phase 7 — Deployment.** Dockerfiles, compose, nginx and CI are written and statically checked; **neither image has been built** and CI has never been observed green.
- [ ] Public deployed demo — not started, needs a host.

## Key Achievements
1. Trained from scratch, no pretrained vision-language weights anywhere
2. Test R@10 **23.91%, 76× chance**, with a val→test gap of +0.84pp
3. Found and fixed the collapse: the memory queue was causing it, not mitigating it — a controlled A/B reversed the project's own published conclusion
4. Built the metric that catches it: separation, not variance, and it now runs every epoch
5. 500 tests across data, model, training, evaluation, serving and UI
6. Every reported number regenerable from one script against one checkpoint

## Known Problems
Tracked in [KNOWN_ISSUES.md](KNOWN_ISSUES.md). Open items:
1. The space is still ANISOTROPIC — ‖mean embedding‖ 0.619 against a 0.5 threshold.
2. Training stopped at epoch 10 of 20; val R@10 was still improving.
3. Neither Docker image has been built; CI never observed green.
4. No public deployment.

## Environment
- Python 3.12.10, PyTorch 2.13.0+cu126
- Node.js 22.17.1, npm 10.9.2
- RTX 4050 laptop GPU (6GB VRAM), 16GB system RAM
- Pinned working set: `requirements.lock.txt`

## Repository Status
All work committed and pushed to `origin/main`. History was rewritten on
2026-08-24 to remove 42 empty commits; backup at `backup-pre-rewrite`.

## Last Updated
2026-08-24
