# PROJECT_STATUS.md — VectorMind

**This is a live document.** Update it at every milestone, not just at
phase boundaries. Anything not directly confirmed in this conversation
is marked **UNKNOWN** below rather than assumed — see the note at the
bottom on why.

---

## Project Name
VectorMind

## Current Phase
Phase 0 — Project Setup & Architecture Decisions (per ROADMAP.md)

## Current Stage
Phase 0.2 (VRAM profiling) — reported by the user as run, but the
specific measured output has not been provided, so it cannot be
recorded here or in ARCHITECTURE.md §6 yet (see "Known Issues" below).

## Overall Completion %
Rough estimate based on ROADMAP.md's 8 phases (0 through 7): **~6%**
(Phase 0 nearly complete; Phases 1-7 not started). Treat this as a
coarse phase-count estimate, not a precise metric — no formal
story-point or task-count system exists for this project.

## Current Status
Documentation foundation is complete and comprehensive (11 markdown
files covering context, architecture, roadmap, standards, and
process). Actual code beyond the Phase 0 skeleton and VRAM profiling
script has not been written. No model, data pipeline, or training
loop code exists yet.

## Completed Work
- [x] Repository structure created (`src/vectormind/{data,models,training,evaluation,utils}`,
      `tests/`, `configs/`, `scripts/`)
- [x] `.gitignore`, `requirements.txt`, `pyproject.toml` (pytest + mypy config)
- [x] `utils/config.py` (YAML config loader) + tests (6 tests, passing)
- [x] `utils/logging_config.py` (logging setup) + tests (4 tests, passing)
- [x] `scripts/profile_vram.py` (Phase 0.2 VRAM profiling script) —
      syntax-checked, **not runtime-verified** (no GPU/torch in the
      sandbox this was built in; must be run on the actual RTX 4050)
- [x] `configs/profiling.yaml`
- [x] Core docs written/expanded: `README.md`, `CLAUDE.md`,
      `ARCHITECTURE.md`, `ROADMAP.md`, `PROJECT_CONTEXT.md`,
      `TECH_STACK.md`, `CODING_STANDARDS.md`, `DEVELOPMENT_GUIDE.md`,
      `PROJECT_RULES.md`, `FOLDER_STRUCTURE.md`, `FUTURE_IDEAS.md`
- [x] Found and fixed a real bug in the original `ROADMAP.md`: it
      contained two duplicate copies of Phases 0-6, and the duplicate
      was missing the Phase 3.5 sanity-check gate

## Current Work
Waiting on real Phase 0.2 measurements from the user to finalize
ARCHITECTURE.md §6 and formally close out Phase 0.

## Pending Work
- Record actual Phase 0.2 batch-size/VRAM numbers (blocked on user input)
- Phase 1: Flickr30k download/verification, tokenizer selection,
  `Dataset`/`DataLoader` implementation, leak-free train/val/test split
- Phases 2-7: not started (see ROADMAP.md for full detail)

## Known Issues
1. **Phase 0.2 numbers not recorded.** The user reported (in this
   conversation) having run CUDA verification, the batch-size
   profiling script, and a GitHub push — but did not provide the
   actual output (recommended batch size, peak VRAM, GPU name from
   `torch.cuda.get_device_name(0)`, or `vram_profile_results.json`).
   **ARCHITECTURE.md §6 still contains the original placeholder text**,
   not a measured number. This is the single most important open item.
2. **`profile_vram.py` has never been executed against a real GPU**
   in any environment observed in this conversation — it was written,
   syntax-checked, and AST-parsed in a sandbox with no CUDA/torch
   available. Its logic has not been validated end-to-end.

## Current Blockers
Blocked on: the user running `scripts/profile_vram.py` on their actual
RTX 4050 (if not already done) and reporting back the
`recommended_batch_size`, peak memory, and GPU name.

## Next Immediate Task
Get and record the real Phase 0.2 numbers; update ARCHITECTURE.md §6
and mark Phase 0 status as complete in ROADMAP.md.

## Next Major Milestone
Phase 1 — Data Pipeline (Flickr30k `Dataset`/`DataLoader`).

## Last Updated
This document reflects the state of the project as of this
conversation. There is no independent timestamp system in place yet —
**add one** (e.g. a header with the date of last edit, updated by
whoever edits this file) once this moves into active multi-session
development, per DEVELOPMENT_GUIDE.md §5.

## Environment Status
**UNKNOWN in detail.** The user has stated (in this conversation) that
a venv was set up, PyTorch with CUDA was installed, and
`torch.cuda.is_available()` was verified — but did not share the exact
Python version, PyTorch version, CUDA version, or the actual
`torch.cuda.get_device_name(0)` output. Do not assume these match the
placeholder guidance in `requirements.txt`/`README.md` until confirmed.

## Repository Status
**Partially unknown.** The user stated the core docs (`README.md`,
`CLAUDE.md`, `ARCHITECTURE.md`, `ROADMAP.md`) were pushed to GitHub
after the first round of work. Whether the second round (the 7
supplementary docs: `PROJECT_CONTEXT.md`, `TECH_STACK.md`,
`CODING_STANDARDS.md`, `DEVELOPMENT_GUIDE.md`, `PROJECT_RULES.md`,
`FOLDER_STRUCTURE.md`, `FUTURE_IDEAS.md`) or the code skeleton
(`src/vectormind/`, `scripts/`, `configs/`, `tests/`) have been pushed
is **not confirmed** — those were delivered as a downloadable zip, and
no push confirmation (e.g. a commit SHA) has been reported back.

## Hardware Status
RTX 4050 laptop GPU, 6GB VRAM — this is the fixed project constraint
(CLAUDE.md §1). CUDA availability reported as verified by the user;
exact driver/CUDA toolkit version **UNKNOWN**.

## Training Status
Not started. No model architecture code exists yet (Phase 2), so no
training is possible yet regardless of Phase 0/1 status.

## Deployment Status
Not started. Backend/frontend/deployment scope (Phase 6/7) exists only
as architecture documentation (ARCHITECTURE.md §9-14) — no
`backend/`, `frontend/`, or `deployment/` code exists.

## Documentation Status
Complete for the current phase: all 11 files from the two prior
documentation rounds exist, are internally consistent (cross-checked
during this pass), and are free of the duplication bug found in the
original `ROADMAP.md`. This document and its 3 companions
(`PROJECT_MEMORY.md`, `DATASETS.md`, `EXPERIMENTS.md`) are the fourth
round, filling gaps the prior two rounds didn't cover.

---

**Note on "UNKNOWN" markers above:** this file is meant to be the
single source of truth for a new engineer (or a new AI conversation)
joining with zero other context. Filling in a plausible-sounding guess
for hardware/environment/git status would be actively misleading for
that purpose. Replace each UNKNOWN with the real value as soon as it's
confirmed — don't leave this file stale once the real numbers exist.
