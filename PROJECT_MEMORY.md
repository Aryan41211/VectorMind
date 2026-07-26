# PROJECT_MEMORY.md — VectorMind

The engineering journal. Unlike ARCHITECTURE.md (what the design *is*)
or ROADMAP.md (what's *planned*), this document is about what actually
happened while getting here: decisions made and why, problems hit and
how they were resolved, and what's still genuinely open. Append to
this over time — don't rewrite history in it.

---

## Important / Architectural Decisions

### Decision: Train entirely from scratch — no pretrained CLIP/OpenCLIP weights
**Why:** the explicit point of the project (CLAUDE.md §1,
PROJECT_CONTEXT.md §3) is to demonstrate understanding of *why*
CLIP-style architectures are shaped the way they are, not to
demonstrate fine-tuning API usage. This is the single decision every
other architecture choice gets checked against.

### Decision: Small ResNet-18-style CNN over a from-scratch ViT for the image encoder
**Why:** ViTs lack the convolutional inductive bias (locality,
translation equivariance) that lets CNNs learn efficiently from small
datasets. A from-scratch ViT on ~30k images and 6GB VRAM would likely
underperform a CNN and cost more to train — documented in
ARCHITECTURE.md §2/§8.

### Decision: Small Transformer (not LSTM) for the text encoder
**Why:** more demonstrative of current ML engineering understanding
for a portfolio; self-attention handles variable-length captions
without an RNN's sequential bottleneck, and remains trainable at 4-6
layers on 6GB VRAM. Documented in ARCHITECTURE.md §3/§8.

### Decision: Pretrained BPE tokenizer, used strictly as preprocessing
**Why:** tokenization defines vocabulary, not learned language
understanding — using an existing tokenizer doesn't compromise the
"trained from scratch" claim, since no *embeddings* or encoder weights
are borrowed. Explicitly reasoned through in ARCHITECTURE.md §3 and
re-justified in TECH_STACK.md's tokenizer section when the question of
"why not train a custom tokenizer too" came up during documentation.

### Decision: In-batch negatives + MoCo-style memory queue, not in-batch alone
**Why:** the 6GB VRAM ceiling caps the physically achievable batch
size far below what CLIP-style in-batch-negative training needs
(CLIP itself used batch size 32,768). The memory queue decouples
negative-sample count from physical batch size. This is the most
load-bearing decision in the whole architecture — ARCHITECTURE.md §6
calls it out explicitly as "the section that most needs explicit
design, because 6GB VRAM is the binding constraint on the entire
project."

### Decision: Learnable temperature, not fixed
**Why:** lets the model calibrate similarity sharpness during
training rather than requiring a correctly-guessed fixed value
upfront, following CLIP's own approach. ARCHITECTURE.md §5.

### Decision: Phase 3.5 (tiny-subset overfit sanity check) is a hard gate before Phase 4
**Why:** embedding collapse (all embeddings converging to
near-identical vectors while loss looks fine) is a silent failure mode
— it wouldn't necessarily show up as an obviously broken loss curve.
Proving the pipeline can memorize a tiny subset first catches this
(and other pipeline bugs) before spending compute on a full run.
CLAUDE.md §4 makes this non-optional.

### Decision: exact FAISS `IndexFlatIP`, not an approximate index, for serving
**Why:** at ~30k vectors, exact brute-force cosine similarity is still
sub-millisecond. The approximate-index speed/recall tradeoff has no
current upside — added during the Phase 6+ architecture expansion
(ARCHITECTURE.md §9), explicitly flagged as something to revisit only
if FUTURE_IDEAS.md's "larger datasets" item is ever pursued.

### Decision: FastAPI over Flask/Django for the serving layer
**Why:** native async, automatic OpenAPI schema generation, and
Pydantic validation fit the distinct typed request schemas needed for
text vs. image queries. TECH_STACK.md.

### Decision: keep the Phase 6+ serving/frontend/deployment stack additive, not a replacement of Phase 0-5 decisions
**Why:** when a much larger "production AI startup" documentation
template was requested (full React/FastAPI/Docker/K8s stack framed as
if reviewed by major AI labs), it directly conflicted in tone and
scope with the already-executed, hardware-validated Phase 0-5 plan
(RTX 4050, Flickr30k, from-scratch, "not trying to match published
CLIP numbers" per ROADMAP.md's Realistic Success Definition).
Overwriting the existing `ARCHITECTURE.md`/`ROADMAP.md`/`CLAUDE.md`
wholesale from a generic template risked silently erasing decisions
already validated on real hardware. Resolution (explicitly confirmed
with the user): layer the new stack onto Phase 6/7 as concrete
technology choices for serving/deployment, leave Phase 0-5's locked
training architecture untouched. Codified as PROJECT_RULES.md rule #27.

## Problems Encountered & Solutions

### Problem: `ROADMAP.md` contained two full duplicate copies of Phases 0-6
**Found during:** the documentation-expansion pass (second round).
**Impact:** the second (duplicate) copy was a *shorter* version that
was missing the Phase 3.5 sanity-check gate entirely — a real risk if
someone read only the second copy and concluded there was no
mandatory pre-Phase-4 check.
**Solution:** kept the first, more detailed copy (with the Phase 3.5
gate and the Known Risks table) and removed the duplicate.
**Lesson:** always grep/scan for structural duplication before editing
a long living document, not just visually skim it — this bug was easy
to miss by eye in a 400+ line file.

### Problem: shell brace-expansion (`mkdir -p src/{a,b,c}`) silently failed under `/bin/sh`
**Found during:** initial repo skeleton creation.
**Impact:** created a literal directory named `{src` instead of the
intended nested structure; not immediately obvious from the command's
exit code alone.
**Solution:** verified directory creation with `find`/`ls` after every
structural `mkdir`, and switched to explicit (non-brace) `mkdir -p`
calls once the issue was identified.
**Lesson:** when scripting environment setup, verify the actual
filesystem result, don't trust a zero exit code as proof the intended
structure was created.

## Lessons Learned

- **Generic "act as a team of 8 senior roles, generate a huge doc
  suite" prompts are useful for coverage but dangerous for
  consistency** — they tend to assume a scope (full production
  startup stack) that may not match a project's actual,
  hardware-validated constraints. Always reconcile against existing
  locked documents before generating, rather than regenerating
  wholesale.
- **"Ran everything" from the user is not the same as having the
  actual numbers.** The user confirmed running the Phase 0.2 profiling
  script, but the specific measured batch size/VRAM/GPU name was never
  actually provided in this conversation. Don't record a phase as
  numerically complete based on a general confirmation alone — get the
  actual value. (See PROJECT_STATUS.md's "Known Issues.")
- **A document that claims to reflect "current reality" needs an
  explicit UNKNOWN-marking convention**, or it silently degrades into
  a document that reflects "what seemed plausible at write time" —
  which is worse than not writing it at all for onboarding purposes.

## Future Reminders

- Once the real Phase 0.2 numbers arrive: update ARCHITECTURE.md §6
  (replacing the placeholder), ROADMAP.md's Phase 0 status/checklist,
  and this file (add a new decision entry once the number is real,
  since "measured batch size = N" is itself a decision point worth
  recording, not just a config value).
- Confirm whether the second and third rounds of documentation
  (7 supplementary docs + this fourth round of 4) have actually been
  pushed to GitHub — PROJECT_STATUS.md currently marks this UNKNOWN.
- Re-profile VRAM (Phase 0.2) once the real Phase 2 model exists — the
  current profiling script uses stand-in encoder classes
  (`ImageEncoderStub`/`TextEncoderStub`) that approximate but don't
  exactly match the eventual real architecture's memory footprint
  (noted as a limitation in `scripts/profile_vram.py`'s own docstring).

## Open Questions

- What is the actual measured Phase 0.2 batch size and peak VRAM
  usage? **(blocking Phase 0 completion — see PROJECT_STATUS.md)**
- Which experiment-tracking tool will actually be used — Weights &
  Biases or TensorBoard? TECH_STACK.md documents both as viable;
  CLAUDE.md §5 names W&B as the default, but no account/setup has been
  confirmed in this conversation.
- Has a license been chosen for the repository itself (distinct from
  the Flickr30k dataset's own license, covered in DATASETS.md)?
  README.md currently marks this "not yet chosen."
- Is a single-machine/VM deployment target (Docker Compose, per
  ARCHITECTURE.md §11) still the intended Phase 7 target, or might
  scope grow — and if it does, does PROJECT_RULES.md #27's
  "additive, not contradicting" principle still hold, or does it need
  revisiting at that point?
