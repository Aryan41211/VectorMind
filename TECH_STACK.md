# TECH_STACK.md — VectorMind

Every technology used in this project, why it was chosen, what was
rejected, and how it integrates. Current-phase (0-5, training) tools
are marked **[current]**; serving/deployment tools (Phase 6-7) are
marked **[phase 6+]** and are not yet in use — see ROADMAP.md for when
each becomes relevant.

---

## Python **[current]**

**Purpose:** primary language for the entire training pipeline and
backend.

**Why selected:** the ML ecosystem (PyTorch, tokenizers, FAISS
bindings) is Python-first; there is no serious alternative for this
project's model-training half.

**Alternatives considered:** none seriously — this is not a decision
point where alternatives were close.

**Trade-offs:** slower raw execution than compiled languages, mitigated
by PyTorch's C++/CUDA backend doing the actual tensor math; Python is
only the orchestration layer.

**Integration:** everything in `src/vectormind/`, `scripts/`,
`backend/`.

---

## PyTorch **[current]**

**Purpose:** the deep learning framework — model definition, autograd,
mixed precision, checkpointing.

**Why selected:** explicit hard constraint in CLAUDE.md §1. More
Pythonic and debuggable than TensorFlow/JAX for a from-scratch,
single-developer research project; eager-mode execution makes it
easier to inspect intermediate tensors while debugging the contrastive
loss and memory queue.

**Alternatives considered:** TensorFlow (rejected: less common in
current research code, steeper debugging curve for eager-mode
inspection); JAX (rejected: excellent for research at scale, but its
functional style and `jit` boundaries add friction for a project this
size with a single developer who needs to move fast and debug
interactively).

**Trade-offs:** PyTorch's dynamic graph has runtime overhead vs.
JAX/XLA's compiled graphs — irrelevant at this project's scale (single
GPU, ~30k images).

**Integration:** `src/vectormind/models/`, `training/`.

---

## Torchvision **[current]**

**Purpose:** image transforms (resize, normalize, augmentation) and
standard building blocks for the CNN image encoder.

**Why selected:** ships with PyTorch, well-tested, avoids
reimplementing standard transforms.

**Alternatives considered:** `albumentations` (more augmentation
options, but adds a dependency not needed at this project's
augmentation complexity level — revisit if augmentation strategy
becomes a bottleneck in Phase 1).

**Trade-offs:** none significant at this scale.

**Integration:** `src/vectormind/data/` (Phase 1).

---

## Tokenizers (Hugging Face `tokenizers`) **[current]**

**Purpose:** subword (BPE) tokenization of captions — preprocessing
only, per ARCHITECTURE.md §3. No pretrained embeddings or encoder
weights are used from this library.

**Why selected:** fast (Rust-backed), well-documented BPE
implementation; using it purely as a tokenizer (not a model) keeps the
"trained from scratch" claim in ARCHITECTURE.md §8 accurate — the
tokenizer defines vocabulary, it doesn't contain learned language
understanding transferred into VectorMind's text encoder.

**Alternatives considered:** training a custom tokenizer from scratch
on Flickr30k captions (rejected: ARCHITECTURE.md §3 explicitly frames
tokenization as preprocessing, not a learned component worth spending
project time on — the "from scratch" claim is about the model, not
every string-processing utility around it).

**Trade-offs:** ties vocabulary to whatever corpus the chosen
pretrained tokenizer was built on; captions with very unusual
vocabulary may tokenize into more subword pieces than ideal. Acceptable
for Flickr30k's everyday-scene captions.

**Integration:** `src/vectormind/data/` (Phase 1).

---

## FAISS **[phase 6+]**

**Purpose:** vector similarity index for retrieval at serving time.
See ARCHITECTURE.md §9.

**Why selected:** the standard, battle-tested library for exact and
approximate nearest-neighbor search over dense vectors; `IndexFlatIP`
gives exact cosine similarity search (after L2 normalization) with
no approximation error, which matters for correctly validating that
serving-time retrieval matches Phase 5's offline evaluation numbers.

**Alternatives considered:** a hand-rolled brute-force `torch.matmul`
search (rejected: FAISS is more optimized and battle-tested, and this
is exactly the kind of solved problem not worth reimplementing);
Annoy/HNSWlib (rejected for now: their approximate-search speed
advantage isn't needed at 30k vectors — see ARCHITECTURE.md §9's
scope note; revisit if FUTURE_IDEAS.md's "larger datasets" is pursued).

**Trade-offs:** none significant at this scale; would need
reassessment (likely to `IndexIVFFlat` or HNSW) if the corpus grows by
orders of magnitude.

**Integration:** `backend/index_builder.py`, loaded at FastAPI startup.

---

## FastAPI **[phase 6+]**

**Purpose:** the backend API layer serving search queries.

**Why selected:** native async support, automatic OpenAPI schema
generation, and Pydantic-based validation — a good match for typed,
distinct request schemas (text query vs. image upload). See
ARCHITECTURE.md §9.

**Alternatives considered:** Flask (rejected: no native async, no
built-in request validation — would need Flask-RESTX or manual
validation to match what FastAPI gives by default); Django REST
Framework (rejected: far more machinery — auth, ORM, admin panel —
than this project needs; VectorMind has no relational data model).

**Trade-offs:** smaller ecosystem of third-party extensions than
Flask/Django, not a practical concern at this project's scope.

**Integration:** `backend/app.py`, `backend/routers/`, `backend/schemas.py`.

---

## React **[phase 6+]**

**Purpose:** the frontend demo UI (search bar, image upload, result
grid). See ARCHITECTURE.md §10.

**Why selected:** component-based model fits the UI's interactive,
stateful needs (live search, upload previews, ranked grids) better
than server-rendered templates.

**Alternatives considered:** Vue (rejected: comparable fit, but React
has the larger ecosystem and more prior familiarity assumed for a
single-developer project moving fast); server-rendered Jinja2 templates
via FastAPI (rejected: works for simple forms, but a live-updating
search-as-you-type UI is meaningfully more awkward without client-side
state).

**Trade-offs:** requires a separate build step and a second
Docker image (ARCHITECTURE.md §11), vs. a template's zero-build
simplicity — accepted because the UI's interactivity needs outweigh
that cost.

**Integration:** `frontend/`.

---

## Tailwind CSS **[phase 6+]**

**Purpose:** styling for the React frontend.

**Why selected:** utility-first classes let a primarily-backend/ML
developer build a reasonably polished UI without deep custom CSS
architecture.

**Alternatives considered:** hand-written CSS/SCSS (rejected: slower
for a non-frontend-specialist to produce a clean result); a component
library like Material UI (rejected: heavier, more opinionated visual
identity than desired for a portfolio demo that should look
intentionally designed, not like a generic admin dashboard).

**Trade-offs:** utility-class-heavy markup is less readable than
semantic CSS class names at a glance — acceptable for a UI this size.

**Integration:** `frontend/tailwind.config.ts`, component files.

---

## TypeScript **[phase 6+]**

**Purpose:** type safety for the frontend, mirroring backend Pydantic
schemas.

**Why selected:** catches frontend/backend schema-drift bugs at
compile time. See ARCHITECTURE.md §10 for why types are hand-mirrored
rather than codegen'd from the OpenAPI schema initially.

**Alternatives considered:** plain JavaScript (rejected: no compile-
time safety net for a two-sided API contract that will evolve).

**Trade-offs:** added build-step complexity vs. plain JS — accepted,
standard for any React project of this size.

**Integration:** `frontend/src/`.

---

## Node.js **[phase 6+]**

**Purpose:** runs the frontend build tooling (Vite/React build,
TypeScript compiler, Tailwind's JIT compiler).

**Why selected:** required by the React/TypeScript/Tailwind toolchain
— not an independent choice.

**Alternatives considered:** none — this is a dependency of the
frontend stack, not a standalone decision.

**Trade-offs:** none beyond what React itself already costs.

**Integration:** `frontend/package.json`.

---

## MLflow / TensorBoard **[current, Phase 3+]**

**Purpose:** experiment tracking — loss, temperature, embedding
norm/variance, GPU memory (CLAUDE.md §5).

**Why selected:** Weights & Biases is the default per CLAUDE.md §5
(cloud-hosted, good comparison-across-runs UI); TensorBoard is the
documented fully-offline fallback for anyone without a W&B account.
"MLflow" appears here because it's a common alternative worth naming;
it is not currently used — W&B/TensorBoard cover this project's needs
without MLflow's additional model-registry machinery, which overlaps
with the lightweight checkpoint-metadata approach in
ARCHITECTURE.md §12.

**Alternatives considered:** MLflow (rejected for now: its model
registry and deployment-tracking features are more than this project
needs at Phase 3-4; revisit only if FUTURE_IDEAS.md's "model registry"
item is pursued).

**Trade-offs:** W&B requires an account/network access for full
features; TensorBoard fallback covers the no-network case.

**Integration:** `src/vectormind/training/train_loop.py` (Phase 3).

---

## pytest **[current]**

**Purpose:** all unit and integration testing (CLAUDE.md §4).

**Why selected:** the standard Python testing framework; fixture
system and `tmp_path` handling made the existing `utils/config.py` and
`utils/logging_config.py` tests straightforward to write.

**Alternatives considered:** `unittest` (rejected: more boilerplate,
less ergonomic fixtures, no real advantage here).

**Trade-offs:** none significant.

**Integration:** `tests/`, configured via `pyproject.toml`
(`[tool.pytest.ini_options]`).

---

## Black **[current]**

**Purpose:** opinionated code formatting.

**Why selected:** removes formatting bikeshedding entirely — no
decisions to make or review-comment about.

**Alternatives considered:** manual formatting with a style guide
(rejected: inconsistent in practice without automated enforcement).

**Trade-offs:** occasionally formats in a way a developer wouldn't
choose by hand — accepted as the cost of zero-debate consistency.

**Integration:** run via pre-commit or CI (`test.yml`, Phase 7);
not yet wired in — add when the CI workflow is built.

---

## Ruff **[current]**

**Purpose:** linting (unused imports, obvious bugs, style issues) —
fast, Rust-based.

**Why selected:** meaningfully faster than `flake8`/`pylint` on this
codebase's size, and covers most of what both plus `isort` do in one
tool.

**Alternatives considered:** `flake8` + `isort` + `pylint` separately
(rejected: three tools, slower, more config surface for equivalent
coverage).

**Trade-offs:** newer tool, smaller plugin ecosystem than `pylint` —
not a practical concern for this project's rule set.

**Integration:** same as Black — wire into CI in Phase 7.

---

## mypy **[current]**

**Purpose:** static type checking, enforcing CLAUDE.md §3's "type
hints required on all function signatures" rule.

**Why selected:** the standard Python type checker; already configured
in `pyproject.toml` (`disallow_untyped_defs = true`).

**Alternatives considered:** Pyright (comparable; mypy chosen for
being the longer-established default with no specific advantage from
switching).

**Trade-offs:** can be verbose with generics/complex typing — not a
significant issue at this project's code complexity.

**Integration:** `pyproject.toml` `[tool.mypy]`; run via CI in Phase 7.

---

## CUDA **[current]**

**Purpose:** GPU acceleration for training and (optionally) inference.

**Why selected:** required by the hard hardware constraint (RTX 4050)
in CLAUDE.md §1 — not an independent choice.

**Alternatives considered:** none — this follows from the hardware,
not a framework preference.

**Trade-offs:** ties the exact PyTorch install command to the specific
CUDA driver version installed — verify with `nvidia-smi` before
installing (see README.md Installation section).

**Integration:** `torch.cuda`, `torch.cuda.amp` throughout training code.

---

## GitHub Actions **[phase 7]**

**Purpose:** CI — test/lint/type-check on every PR, Docker builds on
merge. See ARCHITECTURE.md §11.

**Why selected:** already hosting the repo on GitHub; free for public
repos, no separate CI account needed.

**Alternatives considered:** Jenkins (rejected: requires
self-hosting/maintaining a CI server — unnecessary operational
overhead for a solo project); CircleCI (rejected: no meaningful
advantage over GitHub Actions here, adds a third-party account).

**Trade-offs:** tied to GitHub as the hosting platform — acceptable,
no plan to migrate hosting.

**Integration:** `.github/workflows/test.yml`, `build.yml`.

---

## Docker **[phase 7]**

**Purpose:** containerizing the backend and frontend for consistent,
reproducible deployment. See ARCHITECTURE.md §11.

**Why selected:** standard, avoids "works on my machine" deployment
issues, and makes the eventual deployment target (single VM,
`docker-compose up`) simple.

**Alternatives considered:** bare-metal/VM deployment without
containers (rejected: reproducibility risk — dependency drift between
dev and deploy environments); Kubernetes (rejected as premature — see
ARCHITECTURE.md §11's explicit scope note; named in FUTURE_IDEAS.md as
a future scaling item, not a current need).

**Trade-offs:** added build/image-management overhead vs. bare
deployment — accepted for reproducibility.

**Integration:** `deployment/backend.Dockerfile`,
`deployment/frontend.Dockerfile`, `deployment/docker-compose.yml`.

---

## Optional Future Technologies

Named here for completeness; not committed to. See FUTURE_IDEAS.md for
the reasoning behind each:

- Kubernetes / managed cloud GPU inference (if serving scale ever
  demands it)
- A model registry tool (MLflow Model Registry or similar), if the
  lightweight checkpoint-metadata approach (ARCHITECTURE.md §12)
  stops being sufficient
- A feature store (only relevant if this evolves into a
  multi-model/multi-dataset system, which it currently is not)
- OpenAPI-to-TypeScript codegen (upgrade path once the API schema
  stabilizes, replacing the hand-mirrored types in `frontend/src/types/`)
