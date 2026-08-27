# Known Issues & Open Problems — VectorMind

An honest, current list of what is broken, misleading, or unresolved in
this repository. Kept deliberately blunt: for a portfolio project the
list of things you *know* are wrong is more valuable than the list of
things that pass.

Each entry states the evidence, not just the claim. Entries are removed
only when fixed, never when they become inconvenient.

**Last audited:** 2026-08-25 (deployment path)

---

## 1. The embedding space is severely anisotropic ✅ FIXED (2026-08-24)

**Status:** root cause found and removed. Retained in full because the
diagnosis is the most useful thing in this document.

**Fix:** the cause was the memory queue, not the temperature (§11). With
the queue disabled and the logit scale clamped, and trained to
convergence at epoch 12:

| | Before | After |
|---|---|---|
| Separation | 0.094 | **0.347** |
| Mean image–image cosine | 0.810 | **0.322** |
| ‖mean embedding‖ | 0.900 | **0.621** |
| Test R@10 | 19.63% | **28.91%** |

Better retrieval *and* a 3.7× better separated space. See
ARCHITECTURE.md §6.1 and EXPERIMENTS.md 006.

**Now fully closed.** This fix left the grade at ANISOTROPIC rather than
HEALTHY — ‖mean embedding‖ still exceeded its 0.5 threshold, so the space
retained a shared directional component. A Wang & Isola uniformity term
at weight 0.2 removed the remainder on 2026-08-25 without costing
retrieval, and the shipped model now grades **HEALTHY** on all three
thresholds. See §12.

**Severity when open:** high — this was the project's central technical
finding and it was documented backwards.

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

## 2. The image FAISS index contains 5 duplicate vectors per image ✅ FIXED (2026-08-23)

**Fix:** `deduplicate_image_embeddings()` collapses to one row per
image, the two indices get separate index maps, and `save_indices()`
refuses to write when either map length disagrees with its index. The
frontend has a regression test asserting result keys stay unique.
**The shipped index still needs rebuilding** — see the checklist at the
end of this file.

**Severity when open:** high — user-visible in the demo.

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

## 4. Backend imports break outside the repo root ✅ FIXED (2026-08-23)

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

## 5. Nothing in `deployment/` or `.github/` has ever been executed ✅ MOSTLY FIXED (2026-08-24)

**Fix:** both images are now built and the stack has been run end to
end — `/ready` green, search returning 10 unique images of 10 at 8-19ms
through the nginx proxy ([DEPLOYMENT.md](DEPLOYMENT.md)). Building
surfaced three real defects: nginx could not resolve its upstream at
config-test time, compose services had no image names, and security
headers arrived twice on proxied paths.

**Still open:** no public deployment — deliberately, see §13.

**CI is green** as of 2026-08-24 (run 32756952454). Getting there took
two more fixes, both the same class of works-on-my-machine gap this
audit started with: `types-PyYAML` was installed locally but never
declared, so mypy passed here and failed there; and three tests asserted
`GET /` returns 200, which is only true when the gitignored
`frontend/dist` exists.

**Severity when open:** high. As originally written:

- No Docker image has been built (issues 3 and 4 guarantee the backend image fails).
- No CI run exists — `.github/` is untracked, so GitHub has never seen the workflows.
- The Phase 7 deliverable "Deployed demo reachable via a public URL" is unchecked in the deliverables list while the phase header says complete.

**Fix direction:** either build and run the containers and record the
result, or set Phase 7 back to *in progress* and mark the deliverables
honestly.

---

## 6. Train/serve tokenization skew ✅ FIXED (2026-08-23)

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

## 7. CORS is configured invalidly ✅ FIXED (2026-08-23)

`backend/app.py` sets `allow_origins=["*"]` together with
`allow_credentials=True`. Browsers reject that combination outright —
the wildcard origin is not permitted when credentials are allowed. It is
also broader than a demo needs. Restrict to the frontend origin from
config.

---

## 8. `reports/` numbers are mutually inconsistent ✅ FIXED (2026-08-24/25)

**Fixed by `scripts/generate_reports.py`**, which writes every metric
file in one pass from one checkpoint. The three files below no longer
exist; `metrics_val.json`, `metrics_test.json`,
`embedding_diagnostics.json`, `checkpoint_summary.json` and
`RESULTS.md` replace them, and `checkpoint_summary.json` no longer
carries a `temperature_discrepancy` field because there is nothing left
to disagree with.

Two residual mismatches were closed on 2026-08-25:

- `RESULTS.md` reported the checkpoint as **epoch 11** where every other
  document said epoch 12. Both were describing the same weights: the
  checkpoint stores the training loop's 0-based index. The report now
  prints the human epoch number.
- The health table listed only `||mean image embedding||` (0.588) while
  the verdict beside it quoted 0.621 — the *text* norm, which is the
  larger of the two and therefore the graded one. Both norms are now in
  the table, with the thresholds they are graded against.

The original defect, for the record:

Three files described the same evaluation and disagreed:

- `phase5_embedding_diagnostics.json` — matched 0.45 / unmatched 0.12, separation 0.33
- `phase5_qualitative_analysis.md` — every listed retrieval score is 0.97–0.99, for correct *and* incorrect results
- direct measurement — matched 0.937 / unmatched 0.843, separation 0.094

The qualitative report is closest to reality. Whichever numbers are
kept, they must come from one script with one seed.

`checkpoint_summary.json` also records
`"temperature_discrepancy": "Reported 53.51, actual 55.24"` — an
acknowledged, unresolved inconsistency left in a shipped artifact.

---

## 9. Four training scripts duplicate the same evaluation code ✅ FIXED (2026-08-23/28)

First fixed 2026-08-23: all four scripts now call the shared
`compute_recall_at_k`/`evaluate` in `src/vectormind/evaluation/`.

Fixed 2026-08-28: the duplicated *loops* are gone too.
`scripts/train.py` (761 lines), `resume_training.py` (519),
`benchmark_epoch.py` (692), and `hyperparameter_experiment.py` (506)
each carried their own epoch loop (AMP, gradient accumulation, memory
queue, scheduler stepping, checkpoint cadence, early stopping) — ~2,500
lines of logic that had already drifted apart. The loop now lives once
in `src/vectormind/training/trainer.py`; `scripts/train.py` is a thin
CLI over it, and the three legacy scripts were deleted. The old scripts'
lone uncontrolled difference — `train/lr` logging the static initial LR
to TensorBoard while the real LR decayed — is fixed the same way: the
shared loop reads the LR from the scheduler every epoch.

---

## 10. Repository hygiene ✅ MOSTLY FIXED (2026-08-23/24)

- **Git history:** ~90 of 132 commits are *empty* commits with fabricated `chore: preserve milestone…` messages. CLAUDE.md §7 explicitly forbids commit padding. See the audit note in `CHANGELOG.md`.
- **Binary payload:** ~~`backend/indices/` still ships ~35 MB of `.faiss` + `sample_metadata.json` in git~~ — **no longer true, and this line was stale.** The index outgrew that arrangement: at 234 MB, with `text_index.faiss` alone above GitHub's 100 MB file limit, it cannot be committed. `backend/indices/` and `checkpoints/` are both gitignored and are obtained by running the build commands in [DEPLOYMENT.md](DEPLOYMENT.md#prerequisites). The cost is that a clean clone does **not** run the demo without a training run and an index build first, which DEPLOYMENT.md states and this file previously contradicted.
- **Doc sprawl:** 16 Markdown files at the repository root. Several contradict each other (see issue 5) and `EXPERIMENTS.md` claimed no training had occurred while `TRAINING_LOG.md` documented 8 epochs.
- **No frontend tests.** ~~`frontend/src/` has 5 components and 0 test files~~ — fixed: 56 frontend tests across 5 files, against 526 Python tests.
- **`mypy` does not run clean** — ~~`src/vectormind/utils/config.py` resolves under two module names~~ — fixed; `mypy src/vectormind backend` reports no issues across 39 source files, and it runs in CI.

---

## 11. The memory queue causes the collapse it was added to prevent ✅ FIXED (2026-08-24)

**Severity when open:** high — it is the root cause of §1, and the
project's architecture document named it "the key mitigation".

`ARCHITECTURE.md` §6 added a MoCo-style memory queue to decouple the
number of negatives from the 6GB-limited batch size. `TRAINING_LOG.md`
credited it with +18.2% Recall@10. Both were wrong.

**Controlled A/B, same starting checkpoint, one variable:**

| Epoch 7 from `epoch_006.pt` | Queue active | Queue inactive |
|---|---|---|
| Train loss | 3.86 | **2.51** |
| Val R@1 | 1.73% | **3.81%** |
| Val R@10 | 10.51% | **19.63%** |
| Separation | 0.062 | **0.322** |
| Mean image–image cosine | 0.872 | **0.409** |
| Logit scale | 67.6 | **18.6** |

One epoch of queue negatives undid six epochs of improvement.

**Why the original conclusion survived so long.** Three compounding
problems:

1. **Measured one epoch after activation**, before the collapse reached recall. Epochs 9-15 then fell to 13.62%, and that decline was logged as a *separate* "temperature overgrowth" failure rather than as this one arriving.
2. **Never a controlled A/B.** `--no-queue` substituted a size-1 stub queue, which `load_checkpoint` rejects against a 4096-entry checkpoint, so the baseline arm could only run from scratch. The "comparison" was epoch 6 of one run against epoch 7 of another.
3. **No embedding-health metric existed.** Recall@K was the only signal, and it is precisely the signal that lags a collapse.

**Mechanism.** MoCo pairs its queue with a momentum encoder, whose slow
EMA update keeps queued keys comparable to what the live encoder
produces. VectorMind has none. At `queue_size` 4096 against batch 128,
stale negatives outnumber in-batch ones 32:1, and minimising loss
against thousands of mismatched stale negatives is easier by sharpening
the similarity distribution than by improving the representation. An
unbounded logit scale is the cheapest way to sharpen. So the scale runs
away, the space collapses, and recall follows several epochs later —
which is exactly the sequence Phase 4 recorded without connecting.

**Warmup does not rescue it.** Holding the queue inactive for six epochs
so it activates already full of recent embeddings only delayed the
damage; the collapse arrived in full at the first epoch after
activation. Staleness is structural here, not a startup transient.

**Fix:** train with in-batch negatives only. `warmup_epochs` is retained
for anyone re-running the experiment. A momentum encoder is the correct
way to make a queue work at this scale and is in
[FUTURE_IDEAS.md](FUTURE_IDEAS.md) — until it exists, the honest claim
is "a queue does not work here *without the mechanism that makes it work
elsewhere*", which is a statement about this implementation rather than
about MoCo.

---

## Remaining before the repository is consistent

Fixes landed in code that the shipped artifacts do not yet reflect.
Listed so the gap is visible rather than assumed closed.

- [x] **Rebuild `backend/indices/`** — done, then rebuilt again from
  `--split all`: the shipped index holds **31,783** image vectors and
  158,915 caption vectors, verified live in the container. The earlier
  3,179 was the test split, a tenth of the corpus.
- [x] **Regenerate `reports/`** — done, all from one script run.
- [x] **Propagate the final numbers** — done across README, PROJECT_STATUS and the About panel.
- [x] **Build and run both Docker images** (§5) — done 2026-08-24.
- [x] **Observe CI green once** (§3, §5) — done 2026-08-24, run 32756952454. All five jobs: pytest, mypy, ruff, tsc/oxlint/build, the serving-dependency check, and both Docker image builds.

---

## 12. The embedding space is still anisotropic ✅ FIXED (2026-08-25)

**Severity when open:** medium — retrieval worked, but this was the one
health threshold the shipped model failed, and the last open modelling
problem in the project.

**Fix:** a Wang & Isola uniformity term at weight **0.2**, found by a
three-point sweep (EXPERIMENTS.md 008 and 009). It is the shipped
default in `configs/training.yaml`, and
`checkpoints/train/best_model.pt` is the model trained with it.

Measured on the **test split**, from `reports/metrics_test.json` —
regenerated by `scripts/generate_reports.py` against the shipped
checkpoint, not written by hand:

| Metric | Was (w=0.0) | **Now (w=0.2)** | Threshold | |
|---|---|---|---|---|
| ‖mean image embedding‖ | 0.588 | **0.165** | < 0.5 | ✅ |
| **‖mean text embedding‖** | **0.621** | **0.115** | **< 0.5** | ✅ |
| Separation | 0.347 | **0.482** | > 0.25 | ✅ |
| Mean image–image cosine | 0.345 | **0.027** | < 0.5 | ✅ |
| Mean text–text cosine | 0.386 | **0.013** | < 0.5 | ✅ |
| Unmatched similarity | 0.257 | **0.011** | ≈ 0 | ✅ |
| **Grade** | ANISOTROPIC | **HEALTHY** | | |

The text norm is the row that mattered: the two norms are graded as
their maximum, and 0.621 was the number keeping the model at
ANISOTROPIC (§8 records the earlier confusion between the two).

**It cost nothing.** On the test split:

| Recall@10 | w=0.0 | **w=0.2** | |
|---|---|---|---|
| image → text | 28.91% | **28.91%** | identical to four significant figures |
| text → image | 25.20% | **26.22%** | **+1.03pp** |

Text→image also improved at K=1 (5.71% → 5.98%) and K=5 (17.33% →
18.24%). Image→text R@1 gave up 0.06pp. That is the whole cost.

Text→image is the direction the demo's search box exercises — a visitor
types a sentence and gets photographs — so the model got better at the
thing the product does while the headline metric held.

On val (EXPERIMENTS.md 009, from checkpoint metadata) the same change
reads −0.13pp image→text and +1.30pp text→image. The two splits agree on
direction and disagree only in the third decimal, which is what one
would expect from a 0.26pp val→test gap.

This is the first HEALTHY grade the project has produced on real data.
Phase 3.5's 0.964 separation was an overfit of 100 images.

**Why this took two attempts, and why the first one was right to
refuse.** Weight 0.5 was measured first (008): it also produced HEALTHY,
but cost 2.0pp of R@10, and it was a first guess rather than a swept
value. Shipping it on that evidence would have repeated §11 exactly —
adopting a loss change from one run. The sweep that justified shipping is
009, and it contains a counter-example: **w=0.1 is worse than both its
neighbours** (R@10 26.75%), so the curve is not monotonic and "a smaller
weight costs less" would have been the wrong lesson to draw from 0.5
alone.

The 0.621 figure quoted in earlier revisions of this section was the
*text* mean norm measured on the test split; the table above reads the
image norm from the checkpoint's own val-split metadata, so that every
row comes from one source. Both graded the same model as failing.

**What is still open:** the term was applied by *resuming* into it rather
than training from scratch with it enabled, which forces the model to
unlearn a space it has already committed to. w=0.2's best epoch is 14
against the baseline's 12, which suggests it was still recovering when
the run stopped. That is an improvement left on the table, not a defect —
see [FUTURE_IDEAS.md](FUTURE_IDEAS.md).

---

## 13. Three defects found on 2026-08-25 ✅ ALL FIXED

Found while running the §12 weight sweep. Grouped because they share a
cause: a claim written in a document, never checked against the code or
the file it described.

### 13a. The memory queue defaulted **on**

ARCHITECTURE.md §6 has stated since the A/B that
`memory_queue.enabled` "is retained and defaults to off". There was no
such key. `scripts/train.py` built the queue unless `--no-queue` was
passed, so the real default was **on** — the setting measured at 87%
worse R@10 and 81% worse separation (§11).

Nobody noticed because every run since the A/B passed the flag by hand.
Anyone following the documented command would have reproduced the exact
failure this project spent two days diagnosing.

**Fixed:** `memory_queue.enabled: false` exists in
`configs/training.yaml`, `train.py` reads it, and `--queue` /
`--no-queue` override it in either direction.

### 13b. The checkpoint registry described a retired run

`checkpoints/checkpoint_metadata.json` is what ARCHITECTURE.md §12 offers
as traceability and what ROADMAP.md's production goals tick off as
delivered. It was hand-maintained, so it drifted: after the Phase 4b
retrain it still described `best_model.pt` as *"epoch 7, val R@10
0.2023, BEST — queue enabled (4096 entries)"* while that path held a
queue-disabled epoch-12 checkpoint. One entry's metrics were annotated
`"extrapolated from trend"` — a guess, in a registry.

The commit SHA §12 promises was never recorded at all, so the registry
could not have carried it even in principle.

**Fixed:** `scripts/index_checkpoints.py` regenerates the file from each
checkpoint's own embedded metadata and leaves unknown fields null;
`save_checkpoint` now records commit, branch and dirty state.

### 13c. The failure-rank histogram counted the opposite of its name

`compute_failure_analysis` returned `failure_rank_distribution`, a
length-*k* list whose entry 0 was incremented once per **failure** and
whose other *k-1* entries were never touched. Beside the label "rank
distribution", `[638, 0, 0, ...]` reads as *every miss was a near miss*.
It was a failure count in a list.

Its docstring also advertised three failure modes — semantic, visual,
ambiguous — that the function has never computed.

**Fixed:** replaced with `hit_rank_distribution`, the rank at which each
successful query first found a correct caption, summing to the number of
successes; docstring rewritten to describe what the code does.

---

## 14. Four defects on the public-deployment path ✅ ALL FIXED (2026-08-25)

Found by running the TLS overlay for the first time. Grouped because
they share the cause §13 shares: a file that nothing ever executed.
`nginx -t` during the frontend image build was the only gate over
`deployment/`, and it checks one file's syntax.

**Severity when open:** high — three of the four sat directly on the
only remaining Phase 7 deliverable, and one was a live security gap in
the stack as shipped.

### 14a. The Caddyfile did not parse

`deployment/Caddyfile` put `transport http { read_timeout 120s }` at
site level. `transport` is a subdirective of `reverse_proxy`. Caddy
does not warn and fall back — it exits:

```
Error: adapting config using caddyfile: /etc/caddy/Caddyfile:44:
unrecognized directive: transport
```

So `docker compose -f docker-compose.yml -f docker-compose.tls.yml up`,
the command DEPLOYMENT.md gives as Step 3 of going public, could never
have started Caddy, and the domain could never have been issued a
certificate. Anyone following the runbook would have hit this as their
first act on a fresh VM, with DNS already pointed and the failure
reading like a TLS problem.

**Fixed:** nested inside `reverse_proxy`. `caddy validate` now runs in
CI, which is the check that was missing rather than the fix.

### 14b. The TLS overlay left the app bound on every interface

`docker-compose.tls.yml` rebinds the frontend to `127.0.0.1` so that
Caddy is the only way in. Its own comment explained that this works
"because compose merges port lists by replacement for the same container
port".

It does not. Compose merges sequences by **appending**, so the effective
config carried both mappings:

```yaml
ports:
  - {target: 80, published: "8080"}                        # base file, 0.0.0.0
  - {target: 80, published: "8080", host_ip: 127.0.0.1}    # overlay
```

The app's nginx therefore stayed published on all interfaces behind the
TLS terminator — the exact condition the overlay exists to prevent, and
plain HTTP on a host whose firewall rules the runbook only opens for 80
and 443. The two entries also collide at bind time, so the service
either fails to start with `address already in use` or comes up with no
published port at all, depending on ordering.

**Fixed:** `ports: !override`. CI asserts the merged config has exactly
one frontend mapping and that it is on loopback.

### 14c. No security headers on anything the browser actually loads

The four security headers were set once at server level in
`nginx.conf`. nginx replaces the inherited `add_header` set in any
location that declares an `add_header` of its own — so every location
that set a `Cache-Control` or a `Content-Type` silently dropped all
four. Measured against the shipped image:

| Path | X-Frame-Options | X-Content-Type-Options | Referrer-Policy | Permissions-Policy |
|---|---|---|---|---|
| `/` (the SPA document) | ❌ | ❌ | ❌ | ❌ |
| `/assets/*.js` | ❌ | ❌ | ❌ | ❌ |
| `/images/*` | ❌ | ❌ | ❌ | ❌ |
| `/nginx-health` | ❌ | ❌ | ❌ | ❌ |
| `/search/*`, `/health`, `/ready` | ✅ | ✅ | ✅ | ✅ |

Only the proxied API paths — which declare no `add_header` — kept them.
The demo's own HTML page was framable.

DEPLOYMENT.md recorded "Security headers ✅ all four present, exactly
once", because the check had been run against an API path. This is the
same shape of error as §11 and §13: a real measurement, generalised to
a case it did not cover.

Two duplicate-header bugs came out of the same reading:

- `location /assets/` set `expires 1y` **and** `add_header Cache-Control "public, immutable"`, emitting two `Cache-Control` headers. Which one a cache honours is unspecified. Same pattern in `location /images/`.
- `location = /nginx-health` used `add_header Content-Type text/plain`, which appends rather than replaces, so the probe returned both `application/octet-stream` and `text/plain`.

**Fixed:** the headers moved to `deployment/security-headers.inc`,
included in every location; `expires` replaced by a single explicit
`Cache-Control`; `add_header Content-Type` replaced by `default_type`.
CI now asserts, against a running container, that each header appears
exactly once on `/`, `/assets/`, `/nginx-health` and an SPA fallback
route — a check that fails on the pre-fix image and passes on this one.

### 14d. The tokenizer reached the Hub on the first query

`backend.Dockerfile` pre-downloads the tokenizer specifically so that
"the container needs outbound network access at query time, and fails
without it" would stop being true. The cache is baked in correctly and
the tokenizer does load from it — but `transformers` still made an
outbound request first, logging an unauthenticated-Hub warning on every
cold start. Not a failure, but latency and a dependency the image had
already paid to remove.

**Fixed:** `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, set after
the download step, with an offline tokenizer load added as a build gate
so the claim is enforced rather than asserted. Cold first query through
the proxy: **2.07s → 1.06s**.

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
