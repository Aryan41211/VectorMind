# Known Issues & Open Problems — VectorMind

An honest, current list of what is broken, misleading, or unresolved in
this repository. Kept deliberately blunt: for a portfolio project the
list of things you *know* are wrong is more valuable than the list of
things that pass.

Each entry states the evidence, not just the claim. Entries are removed
only when fixed, never when they become inconvenient.

**Last audited:** 2026-08-23

---

## 1. The embedding space is severely anisotropic (the "no collapse" claim is wrong)

**Severity:** high — this is the project's central technical finding and
it is currently documented backwards.

`reports/phase5_embedding_diagnostics.json` reports
`"overall_status": "HEALTHY"`, `"collapse_risk": "LOW"`, and a
matched/unmatched similarity separation of `0.33` (matched `0.45`,
unmatched `0.12`).

Measured directly from the shipped test-split embeddings
(2,000-sample subset of the 15,895 vectors used to build
`backend/indices/`):

| Quantity | Measured | A healthy contrastive space |
|---|---|---|
| Mean image–image cosine (off-diagonal) | **0.810** | ≈ 0 |
| Mean text–text cosine (off-diagonal) | **0.881** | ≈ 0 |
| Matched image–text cosine | 0.937 | high |
| **Unmatched** image–text cosine | **0.843** | ≈ 0 |
| **Separation (matched − unmatched)** | **0.094** | large |
| ‖mean image embedding‖ | **0.900** | ≈ 0 (1.0 = total collapse) |
| ‖mean text embedding‖ | **0.938** | ≈ 0 |

Every embedding sits inside a narrow cone. The `0.33` separation figure
in the diagnostics JSON does not reproduce and appears to have been
written rather than computed — the real separation is **0.094**, about
3.5× smaller.

The Phase 3.5 overfit run is the useful contrast: separation `0.964`,
per-dim variance `0.0039`. Phase 4 has per-dim variance `0.00075`, ~5×
lower. The pipeline can produce a healthy space; the full run did not.

**Why it happened (probable):** the learnable logit scale
(`log_temperature`) is unbounded and grew 14.3 → 55.2 by the best epoch
and past 500 by epoch 15. A large logit scale plus a near-uniform
similarity distribution is the classic signature of a contrastive model
that minimises loss by shrinking the usable volume of the sphere rather
than by separating classes.

**Why the model still beats chance:** 2.0× random @ R@10 is achievable
inside a narrow cone — rank ordering survives even when absolute
similarities are compressed. The retrieval numbers are real; the
"healthy embedding space" interpretation of them is not.

**Fix direction:**
1. Clamp the logit scale to CLIP's ceiling of `ln(100)` (`logit_scale.clamp_(max=4.6052)` after each step). CLIP itself does this — its absence here is the likeliest single cause.
2. Re-run Phase 4 and report separation and ‖mean embedding‖ alongside Recall@K, not variance alone.
3. Replace the hand-authored numbers in `reports/phase5_embedding_diagnostics.json` with script output.

**Do not** re-label this as healthy without re-running the measurement.

---

## 1b. Every "× random baseline" figure is wrong — and understates the result

**Severity:** high — it is the project's headline claim, quoted in
`ROADMAP.md`, `PROJECT_STATUS.md`, `TRAINING_LOG.md`, the Phase 5
reports, and the frontend's About panel.

The claim is *"Test Recall@10 = 19.63% (2.0× random baseline)"*, which
implies a chance rate near 10%.

Chance for image→text Recall@10 on the test split is the probability
that a random 10 of the 15,895 captions contains one of the 5 belonging
to the query image:

```
1 - C(15890, 10) / C(15895, 10)  =  0.31%
```

| Metric | Measured | Claimed chance | Real chance | Claimed | **Real** |
|---|---|---|---|---|---|
| R@1 (I→T) | 4.62% | 1% | 0.031% | 4.6× | **147×** |
| R@5 (I→T) | 13.43% | 5% | 0.157% | 2.7× | **85×** |
| R@10 (I→T) | 19.63% | 10% | 0.314% | 2.0× | **62×** |
| R@10 (T→I) | 15.09% | 10% | 0.315% | 1.5× | **48×** |

**Where the wrong numbers came from:** the Phase 3.5 sanity check
overfits a **100-image** subset, where R@1 chance genuinely is 1% and
R@10 chance genuinely is 10%. `ROADMAP.md` records those correctly at
lines 169–170. Those same two figures were then reused for the full
3,179-image / 15,895-caption test split, where they are roughly 30×
too large.

This is the one error in this repository that runs in the project's
favour: a from-scratch dual encoder on 31k images at **62× chance** is a
substantially better result than "2× chance", and it was undersold for
weeks.

**Fix:** `scripts/generate_reports.py` computes the baseline from the
actual candidate-pool size and relevant-item count, and writes it beside
every recall figure. Note it uses the exact complement rather than the
`k/n` shortcut, which overstates chance whenever an image has more than
one valid caption.

---

## 2. The image FAISS index contains 5 duplicate vectors per image

**Severity:** high — user-visible in the demo.

`backend/indices/image_index.faiss` holds 15,895 vectors, but only
**3,180 are unique**. Flickr30k has 5 captions per image, and the index
builder emits one image embedding per *caption* rather than per *image*.

Consequence: `POST /search/text` searches the image index, so a text
query can return the same image up to 5 times inside top-10. The
effective result diversity is closer to top-2 than top-10.

It also makes `index_metadata.json`'s `"total_images": 15895` wrong —
that is the caption count. The real image count is 3,179.

**Fix direction:** deduplicate image embeddings before
`index.add()` (one row per image), keep a separate caption→image id map
for the text index, and either dedupe at query time or over-fetch and
collapse by filename.

---

## 3. `requirements.txt` did not install a working environment (fixed 2026-08-23)

Previously missing `fastapi`, `uvicorn`, `faiss-cpu`, `numpy`,
`tensorboard`, `transformers`, `opencv-python`, and `matplotlib`, all of
which the code imports. Consequences at the time:

- `.github/workflows/test.yml` could not have passed — `pip install -r requirements.txt` then `pytest` fails at `import fastapi`.
- `deployment/backend.Dockerfile` built an image whose entrypoint fails at `import fastapi`.
- 6 of 345 tests failed locally on `import tensorboard`.

`requirements.lock.txt` was additionally written as **UTF-16**, so
`pip install -r requirements.lock.txt` failed to parse.

Both files have been rewritten. Status: **fixed, but CI has never
actually been observed green** — see issue 5.

---

## 4. Backend imports break outside the repo root

**Severity:** high — the Docker image does not run.

`backend/app.py` and `backend/index_builder.py` import via
`from src.vectormind... import ...`, while everything inside
`src/vectormind/` imports via `from vectormind... import ...`.

This only resolves because the working directory is the repo root *and*
the package is `pip install -e .` into the venv. `backend.Dockerfile`
does `COPY src/ src/` without installing the package, so `src.vectormind`
resolves as a namespace package but its internal `vectormind.*` imports
have nothing on `sys.path` to satisfy them.

**Fix direction:** use `from vectormind... import ...` everywhere and add
`RUN pip install -e .` (plus a real `[project]` block in
`pyproject.toml`, which currently has only tool config) to the
Dockerfile.

---

## 5. Nothing in `deployment/` or `.github/` has ever been executed

`ROADMAP.md` marks Phase 7 **complete**. In fact:

- No Docker image has been built (issues 3 and 4 guarantee the backend image fails).
- No CI run exists — `.github/` is untracked, so GitHub has never seen the workflows.
- The Phase 7 deliverable "Deployed demo reachable via a public URL" is unchecked in the deliverables list while the phase header says complete.

**Fix direction:** either build and run the containers and record the
result, or set Phase 7 back to *in progress* and mark the deliverables
honestly.

---

## 6. Train/serve tokenization skew

Training (`src/vectormind/data/tokenizer.py`) uses
`AutoTokenizer.from_pretrained(cfg.tokenizer_name)` with
`padding="max_length", max_length=77`.

Serving (`backend/routers/text_search.py`) uses `BertTokenizer` with
`padding=True` (dynamic length) and a hardcoded `max_length=77`.

Two problems: the sequence-length distribution at inference does not
match training, and both the model name and the length are hardcoded in
the router instead of read from `configs/data.yaml` — a direct violation
of CLAUDE.md §6.

The router also calls `from_pretrained("bert-base-uncased")` at first
request, which requires network access. In a container with no HF cache
baked in, the first query fails.

---

## 7. CORS is configured invalidly

`backend/app.py` sets `allow_origins=["*"]` together with
`allow_credentials=True`. Browsers reject that combination outright —
the wildcard origin is not permitted when credentials are allowed. It is
also broader than a demo needs. Restrict to the frontend origin from
config.

---

## 8. `reports/` numbers are mutually inconsistent

Three files describe the same evaluation and disagree:

- `phase5_embedding_diagnostics.json` — matched 0.45 / unmatched 0.12, separation 0.33
- `phase5_qualitative_analysis.md` — every listed retrieval score is 0.97–0.99, for correct *and* incorrect results
- direct measurement — matched 0.937 / unmatched 0.843, separation 0.094

The qualitative report is closest to reality. Whichever numbers are
kept, they must come from one script with one seed.

`checkpoint_summary.json` also records
`"temperature_discrepancy": "Reported 53.51, actual 55.24"` — an
acknowledged, unresolved inconsistency left in a shipped artifact.

---

## 9. Four training scripts duplicate the same evaluation code

`scripts/train.py` (761 lines), `resume_training.py` (519),
`benchmark_epoch.py` (692), and `hyperparameter_experiment.py` (506)
each define their own `compute_recall_at_k` and `evaluate`. That is
~2,500 lines with a shared core that already exists in
`src/vectormind/evaluation/`.

CLAUDE.md §3 caps files at ~400 lines and forbids duplicate logic; all
four violate both.

---

## 10. Repository hygiene

- **Git history:** ~90 of 132 commits are *empty* commits with fabricated `chore: preserve milestone…` messages. CLAUDE.md §7 explicitly forbids commit padding. See the audit note in `CHANGELOG.md`.
- **Binary payload:** `backend/indices/` still ships ~35 MB of `.faiss` + `sample_metadata.json` in git. Kept deliberately so the demo runs from a clean clone; revisit if the repo grows.
- **Doc sprawl:** 16 Markdown files at the repository root. Several contradict each other (see issue 5) and `EXPERIMENTS.md` claimed no training had occurred while `TRAINING_LOG.md` documented 8 epochs.
- **No frontend tests.** `frontend/src/` has 5 components and 0 test files, while the Python side has 345 tests.
- **`mypy` does not run clean** — `src/vectormind/utils/config.py` resolves under two module names, which aborts the check before it reaches the code.

---

## Fixed in the 2026-08-23 audit

- `ROADMAP.md` had its entire header and Constraints section duplicated, spliced mid-sentence.
- `requirements.txt` rewritten with all eight missing dependencies.
- `requirements.lock.txt` regenerated as UTF-8.
- `EXPERIMENTS.md` claimed "no training has occurred" — backfilled with the three real runs.
- The frontend "About This Demo" panel advertised a **ResNet-50** image encoder; `configs/model.yaml` specifies `resnet18_style`.
- `symmetric_infonce`'s docstring described the temperature effect backwards ("higher = softer"; it is higher = sharper) and contained a stray CJK character.
- `.gitignore` had a UTF-8 BOM and no rules for tool caches.
- Removed `docs/PHASE_5_PROMPT.md` (596 lines of AI prompt scaffolding), three phase "verification summary" checklists, and 32 MB of `.npy` arrays redundant with the FAISS indices.
