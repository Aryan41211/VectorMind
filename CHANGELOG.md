# Changelog — VectorMind

All notable changes to this project. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project
does not publish versioned releases, so entries are grouped by date.

---

## [Unreleased] — 2026-08-25 — The deployment path, executed

Everything under `deployment/` had been written, reviewed and, for the
base stack, run. The TLS overlay had not. Running it found three defects
that would each have stopped a public deployment, and one that was
quietly costing a second of cold-start latency.

### Fixed

- **The Caddyfile did not parse.** `transport` was written as a
  site-level directive; it is a subdirective of `reverse_proxy`. Caddy
  exits with `unrecognized directive: transport`, so the documented
  go-public command could never have started the TLS terminator and the
  domain could never have been issued a certificate.
- **The TLS overlay did not rebind the app to loopback.** Compose merges
  sequences by appending, not by replacing, so the overlay's
  `127.0.0.1:8080` mapping was *added* to the base file's `0.0.0.0:8080`
  rather than substituted for it. The app's nginx stayed published on
  every interface behind the TLS terminator — the opposite of the
  overlay's purpose — and the duplicate entries also collided at bind
  time. Fixed with `ports: !override`.
- **No security headers on anything the browser loads.** nginx replaces
  the inherited `add_header` set in any location that declares one of
  its own, so `/`, `/assets/`, `/images/` and `/nginx-health` all
  returned none of the four. Only the proxied API paths kept them — and
  an API path is where the "all four present, exactly once" verification
  had been run. The headers now live in
  `deployment/security-headers.inc` and are included per location.
- **Two duplicate-header bugs, same reading.** `/assets/` and `/images/`
  set `expires` alongside an `add_header Cache-Control`, emitting two
  `Cache-Control` headers; `/nginx-health` used `add_header
  Content-Type`, which appends, so it returned both
  `application/octet-stream` and `text/plain`.
- **The tokenizer reached the HF Hub on the first query.** The cache was
  baked into the image correctly, but `transformers` still made an
  outbound request before using it. `HF_HUB_OFFLINE=1` after the
  download step, plus an offline load as a build gate. Cold first query
  through the proxy: **2.07s → 1.06s**.

### Added

- `.github/workflows/test.yml` gains a **deployment** job. It validates
  the Caddyfile, every documented compose combination, and the
  loopback-only property of the TLS overlay — then builds the frontend
  image and asserts on the **actual response headers** of a running
  container. The header assertions fail against the pre-fix image and
  pass against this one. `nginx -t` at build time was previously the
  only gate over `deployment/`, and it checks one file's syntax.
- `deployment/security-headers.inc`, so the header set has one
  definition rather than one per location.

### Documentation

- `docs/DEPLOYMENT.md`: verification table re-measured and corrected —
  the index row said 3,179 (the test split) where the shipped index
  serves 31,783; the security-header row said all four were present.
  Added what the TLS and auth overlays were actually observed to do, and
  narrowed "Going public" from *not executed* to *executed locally
  against Caddy's internal CA; never against a real domain or ACME*.
- `ROADMAP.md`: the Production Goals list ticked "Live deployed demo"
  while Phase 7 recorded the same deliverable as not started. The two
  now agree, and the unticked one is correct.
- `ARCHITECTURE.md` §9.1 gave index row counts for the test split while
  describing the shipped index; both sets of numbers are now stated with
  which split each belongs to.
- `docs/KNOWN_ISSUES.md` §14 records all four defects with measurements.

### Not fixed

There is still no public deployment. Everything up to the host is now
built, run and gated; what is missing is a machine with a public name
and a certificate, which is a decision about money rather than an
engineering task.

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

**Training had been stopped early, and then reached convergence.** The
run was interrupted six times across two days by memory pressure on a
laptop whose GPU also drives the display. Resumed each time, it reached
epoch 12 and stopped improving:

| | Originally shipped | Converged (epoch 12) |
|---|---|---|
| Test R@1 | 4.62% | **7.71%** (245× chance) |
| Test R@5 | 13.43% | **20.60%** (131× chance) |
| Test R@10 | 19.63% | **28.91%** (92× chance) |
| T2I R@10 | 15.09% | **25.20%** (80× chance) |
| Separation | 0.094 | **0.347** |
| Logit scale | 55 → 500+ | **24.1** |

R@10 up **47% relative**, on a space 3.7× better separated.

**It is converged, not interrupted.** Epochs 13–14 dropped training loss
14% without improving validation R@10 once — the model fitting its
training split rather than learning. Early stopping at patience 5 would
have selected the same weights.

The run survived six interruptions only because the resume path restores
the best-so-far score from the checkpoint. Without that fix, any one of
them would have overwritten the best weights with the next epoch's.

The grade remains **ANISOTROPIC**: ‖mean embedding‖ 0.621 against a 0.5
threshold, now tracked as `docs/KNOWN_ISSUES.md` §12 and the project's
main open modelling problem.

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
