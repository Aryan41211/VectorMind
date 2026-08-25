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
| image → text | 1 | 7.71% | 0.031% | **245×** |
| image → text | 5 | 20.60% | 0.157% | **131×** |
| image → text | 10 | 28.91% | 0.314% | **92×** |
| text → image | 1 | 5.71% | 0.031% | **181×** |
| text → image | 5 | 17.33% | 0.157% | **110×** |
| text → image | 10 | 25.20% | 0.315% | **80×** |

Chance is the exact complement of drawing K non-relevant items, not the
`k/n` shortcut — see [KNOWN_ISSUES.md](KNOWN_ISSUES.md) §1b for why
every earlier multiple in this project was ~30× too low.

### Embedding health

| Metric | Phase 4 (retired) | Current | Threshold |
|---|---|---|---|
| Separation | 0.094 | **0.347** | > 0.25 ✅ |
| Mean image–image cosine | 0.810 | **0.322** | < 0.5 ✅ |
| ‖mean embedding‖ | 0.900 | **0.621** | < 0.5 ❌ |
| Logit scale | 55.2 → 500+ | **24.1** | clamped at 100 |

**Grade: ANISOTROPIC.** Not healthy. Separation clears its floor so
retrieval is meaningful, but the space still carries a shared
directional component. This is reported rather than rounded up — the
previous checkpoint's report claimed HEALTHY on worse numbers, and that
is the failure this whole audit was about.

### Val → test

Val R@10 29.30%, test R@10 28.91% — a gap of 0.39pp. Essentially
identical, so the checkpoint generalizes; the convergence seen in
epochs 13–14 is the training split being fitted, not the test split
being missed.

## Tests
- **451** Python tests
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
2. Test R@10 **28.91%, 92× chance**, with a val→test gap of 0.39pp
3. Found and fixed the collapse: the memory queue was causing it, not mitigating it — a controlled A/B reversed the project's own published conclusion
4. Built the metric that catches it: separation, not variance, and it now runs every epoch
5. 507 Python tests and 56 frontend tests across data, model, training, evaluation, serving and UI
6. Every reported number regenerable from one script against one checkpoint

## Known Problems
Tracked in [KNOWN_ISSUES.md](KNOWN_ISSUES.md). Open items:
1. The space is still ANISOTROPIC — ‖mean embedding‖ 0.621 against a 0.5 threshold. This is now the *only* health threshold it fails, and the main open modelling problem.
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
