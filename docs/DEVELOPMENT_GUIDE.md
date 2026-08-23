# DEVELOPMENT_GUIDE.md — VectorMind

How development actually happens on this repo, step by step, for any
piece of work — a new module, a bug fix, or a doc update.

---

## 1. Planning

Before writing any code:
- Read ROADMAP.md — which phase does this work belong to? Is its
  dependency satisfied (e.g. don't start Phase 2 model code before
  Phase 0's architecture decisions and Phase 0.2's batch size are
  locked)?
- Read ARCHITECTURE.md for any section relevant to the work — don't
  re-derive a design decision that's already made and justified there.
- If the work isn't covered by an existing phase/deliverable, decide
  explicitly whether it's in scope (check PROJECT_CONTEXT.md's vision)
  or should go in FUTURE_IDEAS.md instead.

## 2. Design

- For anything touching model architecture, training strategy, or the
  serving stack: state the design and its tradeoffs *before*
  implementing (CLAUDE.md §10 — the AI-assistant behavior rule, which
  applies equally to a human contributor). Update ARCHITECTURE.md as
  part of this step, not after the fact.
- For anything with more than one reasonable approach, write the
  tradeoff down (see ARCHITECTURE.md §8/§14's "Key Design Tradeoffs"
  tables as the format to follow).

## 3. Implementation

- Follow CODING_STANDARDS.md throughout (type hints, docstrings, no
  magic numbers, config-driven, proper logging).
- Minimize unnecessary changes — don't reformat or restructure
  unrelated code while implementing something else (CLAUDE.md §6).
- Keep one logical unit of work per branch/commit in mind from the
  start, not as an afterthought at commit time.

## 4. Testing

- Write the corresponding test file in `tests/` alongside the new
  module — this is a hard gate (CLAUDE.md §4), not a follow-up task.
- Run the full test suite, not just the new tests, and explicitly note
  the regression check result (CLAUDE.md §4): "ran `pytest tests/ -v`,
  all N tests pass" — not just "should be fine."
- For anything touching the training loop or a checkpoint format: the
  Phase 3.5 tiny-subset sanity check is a hard gate before any Phase 4
  full run (CLAUDE.md §4, ROADMAP.md Phase 3.5) — never skip this to
  save time.

## 5. Documentation

- Update ARCHITECTURE.md if a structural/design decision changed
  (CLAUDE.md §8) — it must reflect current reality, not the original
  plan if that plan changed.
- Update ROADMAP.md's checklist and status at every milestone
  (CLAUDE.md §8).
- Update README.md if setup/usage instructions changed.

## 6. Review

Self-review against CODING_STANDARDS.md §11's checklist before
considering the work done. For a second reviewer (human or AI
assistant): check consistency with ARCHITECTURE.md/ROADMAP.md first
(CLAUDE.md §10) before reviewing code style.

## 7. Commit

- Conventional Commits only (CLAUDE.md §7): `feat:`, `fix:`, `docs:`,
  `refactor:`, `test:`, `chore:`.
- One logical unit of work per commit — never bundle a model change
  with an unrelated doc fix.
- Commit message describes what changed and why, specifically — not
  "update model.py."
- Never commit `data/raw/`, `data/processed/`, `checkpoints/`, `logs/`
  (already in `.gitignore`).

## 8. Push

- Push to `origin main` immediately after every completed task/subtask
  (CLAUDE.md §7) — don't batch several days of work into one push.

## 9. Validation

- For training-related changes: confirm the change doesn't silently
  break the Phase 3.5 sanity check's assumptions (e.g. changing the
  loss implementation requires re-running Phase 3.5 before trusting
  any subsequent Phase 4 run).
- For serving-layer changes (Phase 6+): confirm the API's response
  still matches Phase 5's offline evaluation numbers for the same
  query (ROADMAP.md Phase 6 acceptance criteria) — a serving bug that
  silently changes ranking is worse than a crash, because it's easy to
  miss.

## 10. Deployment (Phase 7)

- CI (`test.yml`) must pass before merge.
- `build.yml` builds Docker images on merge to `main`.
- Deploy via `docker-compose up` on the target machine/VM
  (ARCHITECTURE.md §11) — no manual, undocumented deployment steps.

## 11. Post-Deployment

- Check basic request logging (ARCHITECTURE.md §12) after any backend
  deploy to confirm the service is actually receiving and correctly
  handling requests, not just that the container started.

## 12. Issue Fixing

- Reproduce first — for a training bug, this often means the Phase
  3.5 tiny-subset setup, which is fast to iterate on, rather than
  debugging against a full run.
- Fix, then add a regression test that would have caught the bug, not
  just a fix with no test (this is what CLAUDE.md §4's testing
  requirements are for).
- Note the fix in the relevant doc if it reveals a wrong assumption —
  e.g. if a "known risk" in ROADMAP.md's risk table turns out to be
  the actual cause of a real bug, update that table's "where it's
  addressed" column to reflect what was actually done.

## 13. Regression Testing

- Run `pytest tests/ -v` after any change, not just tests for the
  directly-modified module (CLAUDE.md §4: "never break existing
  tests").
- For frontend changes (Phase 6.5+): `tsc --noEmit` to catch type
  errors before they reach the browser.

## 14. Feature Lifecycle Summary

```
Plan (check ROADMAP/ARCHITECTURE)
   │
   ▼
Design (state tradeoffs, update ARCHITECTURE.md)
   │
   ▼
Implement (CODING_STANDARDS.md)
   │
   ▼
Test (write + run tests, note regression check)
   │
   ▼
Document (update ARCHITECTURE.md/ROADMAP.md/README.md as needed)
   │
   ▼
Review (self-review checklist)
   │
   ▼
Commit (Conventional Commits, one logical unit)
   │
   ▼
Push (immediately, to origin main)
   │
   ▼
[Phase 7+] CI validates → Deploy → Post-deployment check
```
