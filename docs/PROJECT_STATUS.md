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
better on every retrieval metric *and* on a space 3.7× better separated.

The serving index now covers the **full 31,783-image corpus** rather
than the 3,179-image test split, so a query has the whole dataset to
match against. Reported metrics below are still measured on the test
split alone.

## Overall Completion
Roughly **95%**. Phases 0–6.5 complete; Phase 4b delivered a better
checkpoint with regenerated reports; Phase 7's containers are built and
verified running. What remains is one green CI run and a public URL.

## Current Results

**Checkpoint:** `checkpoints/train/best_model.pt` — epoch 12, step 11916,
learned logit scale 24.06. **Converged**: epochs 13–14 improved training
loss but not validation, so this is the peak rather than a stopping
point. Every figure below is regenerable with
`python scripts/generate_reports.py`.

### Retrieval (test split: 3,179 images / 15,895 captions)

| Direction | K | Measured | Chance | vs chance |
|---|---|---|---|---|
| image → text | 1 | 7.64% | 0.031% | **243×** |
| image → text | 5 | 20.79% | 0.157% | **132×** |
| image → text | 10 | 28.91% | 0.314% | **92×** |
| text → image | 1 | 5.98% | 0.031% | **190×** |
| text → image | 5 | 18.24% | 0.157% | **116×** |
| text → image | 10 | 26.22% | 0.315% | **83×** |

Chance is the exact complement of drawing K non-relevant items, not the
`k/n` shortcut — see [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §1b for why
every earlier multiple in this project was ~30× too low.

### Embedding health

Test split, from `reports/metrics_test.json`.

| Metric | Phase 4 (retired) | InfoNCE only | **Current** | Threshold |
|---|---|---|---|---|
| Separation | 0.094 | 0.347 | **0.482** | > 0.25 ✅ |
| Mean image–image cosine | 0.810 | 0.345 | **0.027** | < 0.5 ✅ |
| Mean text–text cosine | 0.881 | 0.386 | **0.013** | < 0.5 ✅ |
| ‖mean image embedding‖ | 0.900 | 0.588 | **0.165** | < 0.5 ✅ |
| ‖mean text embedding‖ | 0.938 | 0.621 | **0.115** | < 0.5 ✅ |
| Unmatched similarity | 0.843 | 0.257 | **0.011** | ≈ 0 |
| Logit scale | 55.2 → 500+ | 24.1 | **17.1** | clamped at 100 |

**Grade: HEALTHY.** All three thresholds pass, with room. The two norm
rows are graded as their maximum, so the text norm is the one that
decides — it was the metric that kept the previous checkpoint at
ANISOTROPIC.

This is the first HEALTHY grade the project has produced on real data;
Phase 3.5's 0.964 separation was an overfit of 100 images. It is
computed by `scripts/generate_reports.py` rather than written by hand,
which is the specific failure §1 records: the Phase 4 report claimed
HEALTHY on separation 0.094.

### Val → test

Val R@10 29.17%, test R@10 28.91% — a gap of 0.26pp. Essentially
identical, so the checkpoint generalizes; the convergence seen in the
late epochs is the training split being fitted, not the test split
being missed.

## Tests
- **526** Python tests
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
- [x] **Phase 7 — Containers.** Both images built and the stack run end to end: `/ready` green, 31,783 images indexed, search returning distinct results through nginx.
- [x] **Phase 7 — TLS and auth overlays run** (2026-08-25). Caddy → nginx → backend serves HTTPS with HSTS and an HTTP→HTTPS redirect; basic auth gates `/` and `/search` while leaving the probes open. Running them for the first time found four defects, all fixed (KNOWN_ISSUES §14).
- [x] **CI green** — run 32756952454, all five jobs. A sixth job now validates `deployment/` against real responses rather than syntax alone.
- [ ] Public deployed demo — the stack is built, run and gated; what is missing is a host with a public name and a certificate.

## Key Achievements
1. Trained from scratch, no pretrained vision-language weights anywhere
2. Test R@10 **28.91%, 92× chance** (text→image **26.22%, 83×**), with a val→test gap of 0.26pp
3. Found and fixed the collapse: the memory queue was causing it, not mitigating it — a controlled A/B reversed the project's own published conclusion
4. Built the metric that catches it: separation, not variance, and it now runs every epoch
5. 526 Python tests and 56 frontend tests across data, model, training, evaluation, serving and UI
6. Every reported number regenerable from one script against one checkpoint

## Known Problems
Tracked in [KNOWN_ISSUES.md](KNOWN_ISSUES.md). Open items:
1. ~~The space is still ANISOTROPIC~~ — **resolved** 2026-08-25. A uniformity term at weight 0.2, chosen by a three-point sweep, took ‖mean image embedding‖ 0.577 → 0.154 and separation 0.356 → 0.490. The shipped model grades **HEALTHY** on all three thresholds, and it did not cost retrieval: -0.13pp image→text, **+1.30pp text→image**. KNOWN_ISSUES §12, EXPERIMENTS 009.
2. ~~Training stopped early~~ — **resolved.** Epochs 13–14 dropped training loss 14% without moving validation R@10, so epoch 12 is the converged checkpoint (EXPERIMENTS.md 007). "Train longer" is exhausted as a lever.
3. ~~CI never observed green~~ — **resolved** 2026-08-24, all five jobs passing.
4. No public deployment. ACME issuance against a real domain is the one part of the TLS path that has never been exercised — the internal-CA run proves the proxy topology, not the certificate path.

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
