# PROJECT_RULES.md — VectorMind

Permanent engineering rules. These are non-negotiable defaults — a
deviation requires an explicit, documented reason (in the relevant doc
and the commit message), not a silent exception.

---

1. **Never load pretrained CLIP/OpenCLIP weights anywhere in the
   model.** This is the project's defining constraint (CLAUDE.md §1,
   ARCHITECTURE.md §8). The pretrained BPE tokenizer is the one
   explicitly documented exception, and it is preprocessing, not a
   learned model component (ARCHITECTURE.md §3).

2. **Never hardcode paths.** Everything through `configs/*.yaml` via
   `utils/config.load_config()` (CLAUDE.md §6).

3. **Never hardcode hyperparameters.** Learning rate, batch size,
   optimizer, scheduler, model dimensions — all config-driven
   (CLAUDE.md §6).

4. **Everything configurable stays configurable** — don't reintroduce
   a hardcoded value "temporarily" during debugging and forget to
   revert it before committing.

5. **One responsibility per module.** `models/` never contains
   training-loop logic; `data/` never contains model architecture
   (ARCHITECTURE.md §2, CODING_STANDARDS.md §2).

6. **No duplicate logic.** If the same computation appears in two
   places, it belongs in one shared location (CODING_STANDARDS.md §12,
   DRY).

7. **Never rewrite a stable, tested module without a documented
   reason.** "I think this could be cleaner" is not sufficient reason
   to touch working, tested code with no bug and no new requirement —
   log the actual reason (new requirement, discovered bug, measured
   performance issue) in the commit message.

8. **Always benchmark before optimizing.** No premature optimization
   beyond fixing obvious, observed bottlenecks (CLAUDE.md §9).

9. **Always test.** Every new module in `src/vectormind/` gets a
   corresponding test file before the task is considered complete
   (CLAUDE.md §4) — no exceptions for "small" utilities.

10. **Never break existing tests.** If a change may affect a
    previously completed module, explicitly run and note the
    regression check (CLAUDE.md §4).

11. **The Phase 3.5 sanity check is a hard gate, not a suggestion.**
    No full-scale training run (Phase 4) begins until the tiny-subset
    overfit check has passed and is recorded as passed in ROADMAP.md
    (CLAUDE.md §4).

12. **Always document.** Docstrings on every public class/function
    (CLAUDE.md §3); comments on complex functions explaining *why*
    (CLAUDE.md §8).

13. **Always review** against CODING_STANDARDS.md §11's checklist
    before considering work complete.

14. **Always maintain architecture consistency.** ARCHITECTURE.md must
    always reflect current reality, not the original plan if that plan
    changed (CLAUDE.md §8) — update it in the same piece of work that
    changes the design, not later.

15. **Always update ROADMAP.md at every milestone completion**
    (CLAUDE.md §8) — checklist items, status field, and any risk-table
    entries that turned out to be relevant.

16. **Always explain trade-offs** for any decision with more than one
    reasonable option — in ARCHITECTURE.md's or TECH_STACK.md's
    tradeoff-table format, not buried in a commit message only.

17. **Conventional Commits only.** `feat:`, `fix:`, `docs:`,
    `refactor:`, `test:`, `chore:` (CLAUDE.md §7). No vague messages
    ("update", "fix stuff", "final").

18. **One logical unit of work per commit.** Never bundle unrelated
    changes (CLAUDE.md §7).

19. **Never commit generated artifacts** — `data/raw/`,
    `data/processed/`, `checkpoints/`, `logs/` (CLAUDE.md §7,
    enforced via `.gitignore`).

20. **Never pad commit history.** Do not create commits purely to
    increase commit count — commit history is a portfolio asset and
    must read as an honest engineering log (CLAUDE.md §7).

21. **Push immediately after every completed task/subtask** — don't
    let local, unpushed work accumulate (CLAUDE.md §7).

22. **No file over ~400 lines.** Split by responsibility if a module
    exceeds this (CLAUDE.md §3, CODING_STANDARDS.md §2).

23. **Type hints required on all function signatures**, enforced via
    mypy (CLAUDE.md §3).

24. **`logging` module only — never bare `print()` in library code**
    (CLAUDE.md §5).

25. **Mixed precision (`torch.cuda.amp`) is default, not optional**,
    given the 6GB VRAM ceiling (CLAUDE.md §9).

26. **Any decision in ARCHITECTURE.md §1-8 (the locked training
    architecture) requires updating ARCHITECTURE.md's own change note
    if it's ever revisited** — these sections are the project's most
    load-bearing decisions and must never silently drift from the
    implementation.

27. **New serving/deployment scope (FAISS, FastAPI, React, Docker,
    CI) extends but never contradicts Phases 0-5's training
    constraints.** If a serving-layer decision ever seems to require
    changing a training-layer decision, that's a signal to stop and
    reconcile explicitly (in ARCHITECTURE.md), not to quietly let two
    docs disagree.

28. **Report real numbers, not estimates, once they're measurable.**
    Phase 0.2's batch size, Phase 4/5's Recall@K — these get the
    actual measured value written into the docs, replacing any
    placeholder, as soon as they exist (ROADMAP.md, ARCHITECTURE.md
    §6).
