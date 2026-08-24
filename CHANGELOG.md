# Changelog — VectorMind

All notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
does not publish versioned releases, so entries are grouped by date.

---

## [Unreleased] — 2026-08-23/24 — Audit, retrain, and production hardening

A full-repository audit that turned into a retrain. The headline is a
reversed conclusion: the memory queue that `ARCHITECTURE.md` called
"the key mitigation" is what was collapsing the embedding space.

### The finding

`reports/phase5_embedding_diagnostics.json` claimed the embedding space
was `"HEALTHY"` with a matched-vs-unmatched separation of 0.33. Measured
directly from the shipped vectors it was **0.094** — every embedding sat
inside a narrow cone, and the reported figure did not reproduce.

A controlled A/B from a single checkpoint, one variable, found the cause:

| Epoch 7 from `epoch_006.pt` | Queue active | Queue inactive |
|---|---|---|
| Val R@10 | 10.51% | **19.63%** |
| Separation | 0.062 | **0.322** |
| Logit scale | 67.6 | **18.6** |

Phase 4 had logged the resulting decline as a separate "temperature
overgrowth" problem. It was not separate: without a momentum encoder,
4096 stale negatives against a batch of 128 make it cheaper to minimise
loss by inflating the logit scale than by improving the representation.
The scale runs away, the space collapses, recall follows several epochs
later.

The original +18.2% figure survived because it was measured one epoch
after activation, was never a controlled comparison (`--no-queue`
substituted a stub queue that could not be resumed from), and there was
no embedding-health metric to contradict a recall number that lags
collapse.

See `ARCHITECTURE.md` §6.1, `docs/KNOWN_ISSUES.md` §11, and
`docs/EXPERIMENTS.md` 004–006.

### Relevance and accuracy (2026-08-24, second pass)

Two separate problems, addressed separately.

**The demo searched a tenth of the dataset.** The index was built from
the test split — 3,179 of 31,783 images — so for most queries nothing
relevant existed to retrieve, and the model appeared far worse than it
is. `--split all` indexes the whole corpus. Reported metrics are
unaffected: Recall@K is measured on the held-out test split by
`scripts/generate_reports.py`, which never reads the index.

Consequence: the index is now 234MB, with `text_index.faiss` alone above
GitHub's 100MB file limit, so it is no longer committed. README and
DEPLOYMENT document the build step (~10 min on a GPU).

**Training had been stopped early.** It was killed at epoch 10 of 20
with validation still climbing and loss still falling. Resuming to epoch
11 improved every metric again:

| | Epoch 10 | Epoch 11 |
|---|---|---|
| Test R@1 | 6.04% | **6.29%** (200× chance) |
| Test R@5 | 16.04% | **17.77%** (113× chance) |
| Test R@10 | 23.91% | **25.64%** (82× chance) |
| T2I R@10 | 21.53% | **23.07%** (73× chance) |
| Separation | 0.330 | **0.344** |

Against the originally shipped checkpoint that is 19.63% → 25.64% R@10,
on a space 3.7× better separated. The grade remains **ANISOTROPIC**:
‖mean embedding‖ 0.620 against a 0.5 threshold.

Training remains incomplete — epoch 11 of 20, still improving when
stopped.

### Added

**Correctness and diagnostics**
- `src/vectormind/evaluation/embedding_health.py` — separation, anisotropy and ‖mean embedding‖, the metrics that actually detect collapse. Per-dimension variance does not: embeddings are L2-normalized, so a cone still shows nonzero variance. Tested against a synthetic cone reproducing the Phase 4 geometry.
- `VectorMindModel.clamp_log_temperature()` — CLIP's ceiling of 100, applied after every optimizer step.
- `src/vectormind/training/oom.py` — releases the allocator cache and retries a step after a transient allocation failure. Covers host failures too: a run died at epoch 7 with a cuDNN **host** allocation error while the GPU was fine.
- `src/vectormind/evaluation/evaluator.py` — one shared split-evaluation loop, replacing six copies.
- `scripts/generate_reports.py` — regenerates every report from one checkpoint in one run, with correct random-chance baselines.

**Serving and deployment**
- `backend/middleware.py` — request correlation ids, sliding-window rate limiting, body-size guard, security headers.
- `/ready` — readiness distinct from liveness, naming which artifact is missing.
- `configs/serving.yaml` — every path, origin, limit and tokenizer setting.
- `.dockerignore` — builds were uploading 2.8GB of checkpoints and 1.3GB of images as context.
- nginx now proxies the API, making the deployed app same-origin.

**Frontend**
- Design token system, three-state theming applied before first paint, loading skeletons, a real accessible dialog, request timeouts with typed errors, and an error boundary.
- 56 tests, from zero.

**Documentation**
- `LICENSE` (MIT), `CHANGELOG.md`, `docs/KNOWN_ISSUES.md`, `docs/DEPLOYMENT.md`, `docs/README.md`.

### Fixed

**Model and training**
- The logit scale was unbounded and ran to 500+; `log_temperature` was also subject to weight decay, which is a pull toward an arbitrary target rather than regularization.
- The memory queue is disabled by default (see above).
- `--no-queue` substituted a size-1 stub that `load_checkpoint` rejected against a real checkpoint, so the baseline arm of any experiment could only start from scratch.
- A resumed run reset its best-so-far to 0.0 and overwrote `best_model.pt` with the first epoch it completed — it replaced a 17.46% checkpoint with a 10.51% one.
- Every training step ran the text encoder twice, once for the loss and again to fill the queue.

**Serving**
- The image index held one vector per *caption*: 15,895 rows for 3,180 images, so a text search could return the same photo five times in one top-10.
- Both indices shared one index map, so an image-index hit was resolved against a caption-indexed list.
- `backend/` imported `src.vectormind.*` while `src/` imported `vectormind.*`; the Docker image could not start.
- CORS paired `allow_origins: ["*"]` with `allow_credentials: true`, which browsers reject outright.
- The image router re-declared its own transform pipeline instead of reusing `get_eval_transforms`; the text router padded queries differently than training did.

**Reproducibility**
- `requirements.txt` omitted `fastapi`, `uvicorn`, `faiss-cpu`, `numpy`, `tensorboard`, `transformers`, `opencv-python` and `matplotlib` — every one imported by committed code. CI and the Docker image could not have worked.
- `requirements.lock.txt` was UTF-16 and unparseable by pip.
- `pyproject.toml` had no `[project]` block, so `pip install -e .` was impossible and mypy aborted before checking a line.

**Correctness of reported numbers**
- Every "× random baseline" figure was ~30× too low. The 1% / 10% baselines are correct for the Phase 3.5 **100-image** subset and were carried over to the 3,179-image test split. Real chance for R@10 is 0.314%, so the result is **62× chance, not 2.0×**.
- `symmetric_infonce`'s docstring described the temperature effect backwards.
- The frontend advertised a ResNet-50 encoder; the config specifies ResNet-18-style.

**Documentation**
- `ROADMAP.md` had its title, preamble and Constraints section duplicated, spliced mid-sentence.
- `EXPERIMENTS.md` stated "no training has occurred" while `TRAINING_LOG.md` documented 8 epochs.
- `PROJECT_STATUS.md` claimed Phase 7 complete and "all phases committed" while that work was entirely uncommitted.

### Changed

- **Git history rewritten.** 42 of 201 commits were empty, carrying fabricated `chore: preserve milestone…` messages — a direct violation of `CLAUDE.md` §7. Dropped; 159 real commits remain and the tree is byte-identical. Backup at `backup-pre-rewrite`.
- Seventeen root Markdown files consolidated to five plus `docs/`.
- `num_workers`/`prefetch_factor` 4→2, cutting pinned dataloader buffers from 1.15GB to 0.29GB. These are bounded by system RAM, not VRAM — which failed twice.
- mypy, ruff, tsc and oxlint now pass across the whole repository. None had run clean before.

### Removed

- `docs/PHASE_5_PROMPT.md` — 596 lines of AI execution-prompt scaffolding.
- Three self-graded "verification summary" checklists duplicating ROADMAP status.
- 32MB of `.npy` arrays already contained in the committed FAISS indices.
- ~570 lines of duplicated evaluation code across six scripts.

### Still open

Tracked in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md). The shipped
`backend/indices/` and `reports/` still come from the superseded model
and have not yet been regenerated; neither Docker image has been built;
CI has not been observed green.

---

## Prior history

Phases 0 through 6.5 were developed between 2026-07-26 and 2026-08-07.
See `ROADMAP.md` for per-phase deliverables, `docs/TRAINING_LOG.md` for
training runs, and `docs/PROJECT_MEMORY.md` for the decision record.

**A note on the commit log.** Of the 201 commits that existed before the
2026-08-24 rewrite, 42 were empty commits carrying messages of the form
`chore: preserve milestone — …`. They changed no files and were not part
of the engineering work. They have been removed rather than left to
misrepresent the history, and this note exists so the removal is itself
on the record.
