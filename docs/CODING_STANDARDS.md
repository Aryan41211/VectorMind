# CODING_STANDARDS.md — VectorMind

Detailed coding standards extending CLAUDE.md §3. CLAUDE.md states the
rules; this document explains them with examples and covers the
frontend/API conventions CLAUDE.md doesn't (since it predates the
serving layer).

---

## 1. Naming Conventions

**Python:**
- Modules/packages: `snake_case` (`memory_queue.py`, not `MemoryQueue.py`)
- Classes: `PascalCase` (`ImageEncoder`, `InfoNCELoss`)
- Functions/variables: `snake_case` (`load_config`, `batch_size`)
- Constants: `UPPER_SNAKE_CASE`, defined at module level, never inline
  magic numbers (CLAUDE.md §3)
- Private module-internal helpers: prefix with `_` (`_try_batch_size`)

**TypeScript/React (Phase 6+):**
- Components: `PascalCase.tsx` (`SearchBar.tsx`)
- Hooks: `camelCase`, prefixed `use` (`useSearchResults.ts`)
- Types/interfaces: `PascalCase` (`SearchResponse`, not `ISearchResponse`
  — no Hungarian-notation `I` prefix)
- Files exporting a single component: filename matches the component name

## 2. Folder Conventions

- One responsibility per module/folder (CLAUDE.md §2/§3): `models/`
  never contains training-loop code; `training/` never contains model
  architecture definitions.
- Test files mirror source structure exactly: `src/vectormind/utils/config.py`
  → `tests/utils/test_config.py`.
- No file over ~400 lines (CLAUDE.md §3) — if a module approaches this,
  split by responsibility, not by arbitrary line-count chunks (e.g.
  split `train_loop.py` into `train_loop.py` + `train_step.py` if the
  single-step logic and the epoch-orchestration logic can be separated
  meaningfully, not just cut in half).

## 3. Import Rules

- Standard library imports, then third-party, then local
  (`vectormind.*`) — each group separated by a blank line. Ruff's `I001`
  enforces this, and CI runs it.
- **Always `vectormind.*`, never `src.vectormind.*`.** The `src.` form
  resolves only when the process happens to start in the repository
  root, and it silently breaks anything that does not — `backend/` used
  it and the Docker image could not start, because modules under `src/`
  import each other as `vectormind.*` and nothing put that on the path.
  `pip install -e .` is what makes the short form work everywhere.
- No wildcard imports (`from module import *`) anywhere.
- No circular imports between `src/vectormind/` subpackages — this is
  why ARCHITECTURE.md §2 requires components to interact through
  defined interfaces rather than importing each other's concrete
  classes directly where avoidable.

## 4. Documentation Style

- Every public class/function: docstring covering purpose, inputs,
  outputs, assumptions, limitations (CLAUDE.md §3). See
  `src/vectormind/utils/config.py::load_config` for the reference
  format used throughout this repo.
- Complex functions get a comment explaining *why*, not *what*
  (CLAUDE.md §8) — e.g. the safety-margin logic in
  `scripts/profile_vram.py::find_max_batch_size` explains *why* a
  margin is reserved (memory queue + dataloader + fragmentation), not
  just restating the arithmetic.
- Module-level docstrings state which ROADMAP.md phase populates that
  module (see the existing `src/vectormind/*/__init__.py` files for
  the pattern).

## 5. Type Hints

- Required on all function signatures, including return types
  (CLAUDE.md §3), enforced via `mypy` (`disallow_untyped_defs = true`
  in `pyproject.toml`).
- Prefer `from __future__ import annotations` at the top of new
  modules (already used in `utils/config.py`, `utils/logging_config.py`)
  so modern union syntax (`str | None`) works regardless of the exact
  Python 3.x minor version.
- TypeScript: `strict` mode enabled once `frontend/tsconfig.json`
  exists (Phase 6.5) — no implicit `any`.

## 6. Logging

- `logging` module only, never bare `print()` in library code
  (CLAUDE.md §5) — enforced by convention and code review, not
  currently by an automated lint rule (candidate: a Ruff rule
  disallowing `print` outside `scripts/`).
- Every module gets its own logger: `logger = logging.getLogger(__name__)`
  at module level (see `utils/config.py`, `utils/logging_config.py` for
  the pattern), never a shared global logger imported everywhere.
- Required log events per CLAUDE.md §5: data load complete, epoch
  start/end, checkpoint saved, evaluation complete, retrieval
  completed, errors, warnings.

## 7. Configuration

- No hardcoded hyperparameters, paths, or settings anywhere in `src/`
  or `scripts/` (CLAUDE.md §6) — everything through `configs/*.yaml`
  via `utils/config.load_config()`.
- Config schemas are checked with `require_keys()` at the point of use,
  not assumed — see `scripts/profile_vram.py::main` for the pattern
  (`require_keys(config, [...])` immediately after loading).

## 8. Testing

- Every new module in `src/vectormind/` requires a test file before
  the task is considered complete (CLAUDE.md §4) — no exceptions for
  "small" utility modules; `utils/config.py` and `utils/logging_config.py`
  both have full test coverage despite being small.
- Test naming: `test_<behavior_under_test>`, not `test_<function_name>_1`,
  `test_<function_name>_2` — the name should describe what's being
  verified (see `tests/utils/test_config.py` for examples like
  `test_load_config_missing_file_raises`).
- Use `tmp_path` (pytest's built-in fixture) for any test needing a
  real filesystem path — never write to a hardcoded path or the repo
  root during tests.
- Never break existing tests; if a change may affect a previously
  completed module, explicitly run and note that regression check
  (CLAUDE.md §4) rather than assuming it's fine.

## 9. Error Handling

- Raise specific, informative exceptions with actionable messages —
  see `utils/config.py::load_config`'s `FileNotFoundError` message,
  which tells the caller what to check, not just that the file is
  missing.
- Never silently swallow exceptions (`except Exception: pass`).
  `scripts/profile_vram.py`'s OOM handling is the one deliberate
  exception: it catches `torch.cuda.OutOfMemoryError` specifically
  (not a bare `except`), logs it, and continues the search — this is
  documented, expected control flow, not error-swallowing.
- API layer (Phase 6): FastAPI's exception handlers return structured
  error responses (status code + machine-readable error code + human
  message), never a raw stack trace to the client.

## 10. Performance

- Mixed precision default, not optional, given the 6GB VRAM ceiling
  (CLAUDE.md §9).
- Identify obvious bottlenecks when they appear (data loading stalls,
  unnecessary CPU↔GPU transfers); do not optimize prematurely beyond
  that (CLAUDE.md §9) — e.g. don't hand-write a custom CUDA kernel
  before profiling shows the standard PyTorch ops are actually the
  bottleneck.

## 11. Code Review Checklist

Before considering any task complete:
- [ ] Type hints on all new/changed function signatures
- [ ] Docstrings on all new public classes/functions
- [ ] No magic numbers — config values or named constants only
- [ ] No hardcoded paths
- [ ] Corresponding test file exists and passes
- [ ] Existing tests still pass (regression check noted explicitly)
- [ ] Logging used, not `print()`
- [ ] No file exceeds ~400 lines
- [ ] Docs (ARCHITECTURE.md/ROADMAP.md) updated if a design decision changed
- [ ] Conventional Commit message, one logical unit of work

## 12. Design Patterns / Principles

- **SOLID, applied where it genuinely simplifies** (CLAUDE.md §3) — not
  dogmatically. Example of appropriate use: the dual-encoder's
  swappable-tower design (ARCHITECTURE.md §2) is a direct application
  of the Dependency Inversion Principle, and it's justified because
  encoder swapping is an actual, anticipated need (comparing CNN vs.
  ViT was a real design decision point). Example of *inappropriate*
  over-application: do not create an abstract `BaseLossFunction`
  interface for a single loss function with no planned alternatives —
  that's ceremony, not simplification.
- **DRY** — the memory queue (used by both the training loop and
  potentially evaluation code) lives in one place
  (`training/memory_queue.py`), not duplicated.
- **KISS** — the VRAM profiling script's exponential-search-then-
  binary-search approach (`scripts/profile_vram.py`) is the simplest
  strategy that's still efficient; a naive linear scan from 1 upward
  would work but waste far more GPU time.
- **YAGNI** — no Kubernetes manifests, no model registry, no feature
  store exist yet (see FUTURE_IDEAS.md) because nothing in Phases 0-7
  currently needs them.

## 13. Python Style

- Formatted by Black, linted by Ruff, type-checked by mypy (see
  TECH_STACK.md for why each was chosen). Once CI exists (Phase 7),
  these run automatically on every PR — until then, run manually
  before committing: `black src/ tests/ scripts/ && ruff check src/ tests/ scripts/ && mypy src/`.

## 14. React Style (Phase 6.5+)

- Functional components with hooks only — no class components.
- Props typed explicitly via a named `interface` or `type`, not inline
  object types, for anything with more than one prop.
- No business logic (API calls, data transformation) inside component
  render bodies — goes in `frontend/src/api/client.ts` or a custom hook.

## 15. API Conventions (Phase 6+)

- All request/response bodies defined as Pydantic models in
  `backend/schemas.py` — no raw `dict` request bodies.
- Endpoint naming: `/search/text`, `/search/image` — resource-and-verb
  style, not RPC-style (`/doTextSearch`).
- Errors return a consistent shape: `{"error_code": str, "message": str}`,
  with an appropriate HTTP status code — never a 200 with an error
  buried in the body.
