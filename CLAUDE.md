# CLAUDE.md — VectorMind Development Rules

This file is the permanent, version-controlled source of truth for how
this project is engineered. It persists across sessions — any AI
assistant or contributor working on this repo follows these rules.

---

## 1. Project Identity

VectorMind is a research-grade multimodal semantic search platform.
It learns a shared embedding space for images and text using
contrastive learning (CLIP-style dual-encoder architecture), trained
**entirely from scratch** — no pretrained CLIP/OpenCLIP weights.
It enables cross-modal retrieval (image→text and text→image search)
and demonstrates production-quality ML engineering, not just a
research script.

**Hard constraints that shape every decision below:**
- Training hardware: RTX 4050 laptop GPU, 6GB VRAM
- Framework: PyTorch
- Dataset: Flickr30k (public, ~30k images / 5 captions each)
- Trained from scratch — every design choice must be justified against
  what's actually trainable at this scale, not against what SOTA
  systems do at 400M-pair scale

---

## 2. Architecture Principles

- The dual-encoder design stays fully decoupled: the image tower,
  text tower, and projection heads must each be independently
  swappable without touching the other components.
- Every component interacts through defined interfaces/base classes
  — no component may assume a specific encoder implementation.
- All hyperparameters live in `configs/`, never hardcoded in source.
- Prefer the design that is easiest to extend and explain over the
  design that is fastest to write.

---

## 3. Coding Standards

- Type hints required on all function signatures.
- Docstrings required on every public class/function: purpose,
  inputs, outputs, assumptions, limitations.
- No file over ~400 lines — split by responsibility if exceeded.
- No magic numbers — named constants or config values only.
- No hardcoded paths — use config-driven paths.
- Avoid deeply nested functions, duplicate logic, unnecessary globals.
- Follow SOLID principles wherever it genuinely simplifies the design
  (not applied dogmatically where it adds ceremony).

---

## 4. Testing Requirements

- Every new module in `src/vectormind/` requires a corresponding test
  file in `tests/` before the task is considered complete.
- Unit tests first; add integration tests for the data pipeline and
  training loop checkpoints.
- Never break existing tests. If a change may affect a previously
  completed module, run and note the regression check explicitly.
- No full-scale training run (Phase 4 in ROADMAP.md) may begin until
  the tiny-subset overfit sanity check (Phase 3.5) has passed and is
  recorded as passed in ROADMAP.md. This is a hard gate, not a
  suggestion — it exists specifically to catch silent failures
  (embedding collapse, broken loss, data bugs) before compute is
  spent on a full run.

---

## 5. Logging Standards

- Use Python's `logging` module — no bare `print()` in library code.
- Required log events: data load complete, epoch start/end, checkpoint
  saved, evaluation complete, retrieval completed, errors, warnings.
- Training metrics (loss, temperature, embedding norms, gradient
  norms) go to Weights & Biases (or TensorBoard), not just console
  output.

---

## 6. Configuration Management

- No hardcoded: learning rate, batch size, optimizer, scheduler,
  model dimensions, dataset paths, checkpoint paths, API settings.
- All of the above live in `configs/*.yaml`.

---

## 7. Git Workflow & Commit Conventions

- Conventional Commits only: `feat:`, `fix:`, `docs:`, `refactor:`,
  `test:`, `chore:`.
- One logical unit of work per commit. Never bundle unrelated changes.
- Every completed task or subtask ends in a commit pushed to
  `origin main`, with a message that accurately and specifically
  describes what changed — commit messages are part of the project's
  documentation, not a formality.
- Never commit `data/raw/`, `data/processed/`, `checkpoints/`, `logs/`,
  or any generated artifacts.
- Do not create commits purely to increase commit count. Commit
  history should read as an honest engineering log — this is itself
  a portfolio asset.

---

## 8. Documentation Standards

- `ROADMAP.md` updated at every milestone completion.
- `ARCHITECTURE.md` updated whenever a structural or design decision
  changes — it must always reflect the current, real state of the
  system, not the original plan if that plan changed.
- `README.md` kept current with setup and usage instructions.
- Complex functions get a short comment on *why* they exist, not just
  what they do.

---

## 9. Performance Considerations

- Mixed precision (`torch.cuda.amp`) is default, not optional, given
  the 6GB VRAM ceiling.
- Gradient accumulation used to simulate larger effective batch sizes.
- Identify obvious bottlenecks (data loading stalls, unnecessary CPU↔
  GPU transfers) when they appear; do not optimize prematurely beyond
  that.

---

## 10. AI Assistant Behavior for This Repo

- Acts as senior engineer + mentor: explains design reasoning, flags
  risks (especially VRAM/compute constraints) before implementing,
  and never silently makes an architectural decision without stating
  the tradeoff.
- Always checks `ROADMAP.md` and `ARCHITECTURE.md` before proposing
  new work, to preserve consistency with prior decisions.
- Pushes back on requests that would harm long-term code quality or
  misrepresent the project (e.g. commit-padding), rather than
  complying silently.

---

## 11. Related Documents

This file covers engineering rules. Related documents, each
independently readable but consistent with everything above:

- `PROJECT_CONTEXT.md` — why this project exists, product vision
- `ARCHITECTURE.md` — full technical architecture (training §1-8,
  serving/frontend/deployment §9-14)
- `ROADMAP.md` — phases, milestones, dependencies, current status
- `TECH_STACK.md` — every technology used, why, and alternatives considered
- `CODING_STANDARDS.md` — detailed style rules extending §3 above
- `DEVELOPMENT_GUIDE.md` — the full feature lifecycle, planning through deployment
- `PROJECT_RULES.md` — permanent, non-negotiable engineering rules
- `FOLDER_STRUCTURE.md` — purpose and ownership of every directory
- `FUTURE_IDEAS.md` — research backlog, explicitly not on the critical path

If any of the above ever contradicts this file or ARCHITECTURE.md's
locked training decisions (§1-8), this file and ARCHITECTURE.md §1-8
win — flag the contradiction rather than silently picking one.