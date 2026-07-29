# FOLDER_STRUCTURE.md — VectorMind

Every directory in the repository: its purpose, what depends on it,
what it must never depend on, and what's expected to be added to it in
future phases.

---

## `src/vectormind/` — the locked training/model codebase

### `src/vectormind/data/`
**Purpose:** Flickr30k loading, transforms, tokenization, paired
`Dataset`/`DataLoader`, train/val/test splitting.
**Populated in:** Phase 1
**Depends on:** `utils/` (config, logging).
**Must never depend on:** `models/`, `training/` — data pipeline code
has no reason to import model classes.
**Future additions:** augmentation strategies, additional dataset
adapters if FUTURE_IDEAS.md's multi-dataset items are pursued.

### `src/vectormind/models/`
**Purpose:** image encoder, text encoder, projection heads — the
architecture itself (ARCHITECTURE.md §2-4)
**Populated in:** Phase 2
**Depends on:** `utils/`.
**Must never depend on:** `training/`, `data/` — a model must be
constructible and runnable without importing the training loop
(ARCHITECTURE.md §2's decoupling principle) or a specific dataset.
**Future additions:** alternative encoder implementations, if ever
swapped in behind the same interfaces (ARCHITECTURE.md §2).

### `src/vectormind/training/`
**Purpose:** InfoNCE loss, MoCo-style memory queue, training loop,
mixed precision, gradient accumulation, checkpointing
**Populated in:** Phase 3.
**Depends on:** `models/`, `data/`, `utils/`.
**Must never depend on:** `evaluation/`, `backend/` (Phase 6+) —
training code has no reason to know about serving.
**Future additions:** hyperparameter sweep tooling, if pursued.

### `src/vectormind/evaluation/`
**Purpose:** Recall@K, embedding-space diagnostics (collapse/
uniformity checks).
**Populated in:** Phase 4/5.
**Depends on:** `models/`, `data/`, `utils/`.
**Must never depend on:** `training/` directly for its core metrics
logic (it may load a saved checkpoint, but shouldn't need the training
loop's internals) — decouples "did it learn" from "how it learned."
**Future additions:** qualitative-failure-case tooling (e.g. an
artifact generator for the Phase 5 manual review).

### `src/vectormind/utils/`
**Purpose:** config loading, logging setup, checkpointing, seeding —
shared, dependency-free utilities.
**Populated in:** ongoing, starting Phase 0.
**Depends on:** nothing else in `src/vectormind/` — this is the
lowest layer everything else can depend on.
**Must never depend on:** any other `src/vectormind/` subpackage —
if it did, it would no longer be a safe shared dependency.

---

## `backend/` (Phase 6)

**Purpose:** FastAPI serving layer — endpoints, schemas, the offline
FAISS index builder. See ARCHITECTURE.md §9.
**Depends on:** `src/vectormind/models/` (to load the trained
architecture), `src/vectormind/utils/`.
**Must never depend on:** `src/vectormind/training/` — serving has no
reason to import the training loop, loss, or memory queue.
**Ownership:** owned by whoever is doing Phase 6 work; changes here
should not require touching `src/vectormind/`

## `frontend/` (Phase 6.5)

**Purpose:** React/TypeScript demo UI. See ARCHITECTURE.md §10.
**Depends on:** `backend/`'s API contract only (via HTTP), never
imports Python code directly.
**Ownership:** independently deployable/buildable from the Python side
entirely — this is why it's a separate Docker image (ARCHITECTURE.md
§11).

## `deployment/` (Phase 7)

**Purpose:** Dockerfiles and `docker-compose.yml` tying `backend/` and
`frontend/` together for deployment. See ARCHITECTURE.md §11.
**Depends on:** `backend/`, `frontend/` (as build contexts).
**Must never contain:** application logic — this folder is
orchestration/packaging only.

## `.github/workflows/` (Phase 7)

**Purpose:** CI definitions (`test.yml`, `build.yml`).
**Depends on:** the whole repo, read-only (checks out and runs
against it) — workflows should not need repo-specific logic beyond
invoking `pytest`, `mypy`, `tsc`, and `docker build`

## `configs/`

**Purpose:** every hyperparameter, path, and setting, as YAML — the
single source of truth per CLAUDE.md §6. No code lives here.
**Depends on:** nothing.
**Everything depends on it:** `src/vectormind/*`, `scripts/`,
eventually `backend/` for serving-time settings.

## `scripts/`

**Purpose:** one-off tooling that isn't part of the importable package
— e.g. `profile_vram.py` (Phase 0.2). Not imported by `src/vectormind/`
or `tests/`; scripts are entry points, not libraries.
**Depends on:** `src/vectormind/utils/` (and whichever other
subpackages a given script needs, e.g. `models/` once Phase 2 exists).

## `tests/`

**Purpose:** mirrors `src/vectormind/` structure exactly (CLAUDE.md
§4) — `tests/utils/test_config.py` tests `src/vectormind/utils/config.py`,
and so on. Once `backend/` exists, `tests/backend/` follows the same
mirroring convention.
**Depends on:** whatever it's testing, plus `pytest` fixtures.

## `data/raw/`, `data/processed/` (gitignored)

**Purpose:** the actual Flickr30k files and processed/cached versions.
Never committed (CLAUDE.md §7) — regenerable from the download step
in Phase 1, not treated as source-controlled project state.

## `checkpoints/` (gitignored)

**Purpose:** saved model weights + metadata sidecars
(ARCHITECTURE.md §12). Never committed — too large, and
regenerable/reproducible from a training run given the config and
commit SHA recorded in the metadata sidecar.

## `logs/` (gitignored)

**Purpose:** local log files (`scripts/profile_vram.py`'s
`profiling.log_path`, training logs if file-logging is enabled).
Never committed — ephemeral, machine-specific.

---

## Dependency Direction Summary

```
utils/  ◀── everything (lowest layer, no internal deps)
data/   ◀── training/, evaluation/, backend/(loads via models, not data directly)
models/ ◀── training/, evaluation/, backend/
training/  (nothing else in src/vectormind/ depends on this — it's a leaf)
evaluation/ (also a leaf)
backend/  ◀── frontend/ (via HTTP only, not a Python import)
```

The rule this diagram encodes: dependencies flow inward toward
`utils/` and `models/`/`data/`, never sideways between `training/`,
`evaluation/`, and `backend/` — each of those is a leaf consumer, not a
shared dependency for anything else.
