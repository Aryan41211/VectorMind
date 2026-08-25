# PROJECT_MEMORY.md — VectorMind

The engineering journal. Unlike ARCHITECTURE.md (what the design *is*)
or ROADMAP.md (what's *planned*), this document is about what actually
happened while getting here: decisions made and why, problems hit and
how they were resolved, and what's still genuinely open. Append to
this over time — don't rewrite history in it.

---

## Important / Architectural Decisions

> **Entries below the Phase 0 block were added on 2026-08-25.** This
> journal stopped at Phase 0 while the project made most of its
> consequential decisions; the decisions were recorded in
> ARCHITECTURE.md, EXPERIMENTS.md and KNOWN_ISSUES.md at the time, and
> are summarized back into the journal here so it is a journal again.

### Decision (2026-08-23): clamp the learnable logit scale at 100
**Why:** the scalar is unconstrained, and inflating it lowers
contrastive loss without improving the representation — the cheapest
direction in the landscape, and the optimizer took it. Phase 4 ran
without a ceiling and the scale went 14.3 → 55 → 500+ while the space
collapsed into a cone. A correctness guard, not a hyperparameter.
ARCHITECTURE.md §5.1.

### Decision (2026-08-24): remove the memory queue — reversing the most load-bearing decision above
**Why:** a controlled A/B from a shared checkpoint measured the queue
costing **87% of Recall@10 and 81% of separation**. Without a momentum
encoder its 4096 stale entries outnumber the in-batch negatives 32:1,
and the cheapest way to fit them is to sharpen the similarity
distribution — which is what drove the logit scale up. The mitigation
was the cause. EXPERIMENTS.md 006, KNOWN_ISSUES.md §11.

This overturned the decision recorded above as "the most load-bearing
decision in the whole architecture", and it overturned this project's
own published +18.2% claim, which came from a single epoch measured
before the collapse reached the metric.

### Decision (2026-08-24): grade embedding health on separation, not variance
**Why:** per-dimension variance cannot distinguish a cone from a spread
on a unit hypersphere, so the Phase 4 reports called a collapsed space
HEALTHY. Separation (matched minus unmatched similarity) can, and now
runs every epoch beside Recall@K rather than being reconstructed
afterwards. The grade has three states, because "usable but not clean"
and "matched and unmatched are indistinguishable" are different
problems that were being reported identically.

### Decision (2026-08-25): ship the uniformity-regularized model at weight 0.2
**Why:** the converged InfoNCE-only model failed one health threshold —
‖mean embedding‖ 0.62 against a 0.5 ceiling. A weight sweep put the cost
of fixing it at **0.13pp of R@10** (29.30% → 29.17%) while separation
went 0.347 → 0.49 and ‖mean‖ 0.62 → 0.15. At w=0.5 the same fix cost
2.0pp, which is why the first attempt was not shipped. EXPERIMENTS.md
008 and 009.

### Decision (2026-08-25): defaults live in config, not in shell history
**Why:** the queue had been diagnosed, removed from the recipe and
written up in three documents, and `scripts/train.py` still defaulted it
**on** — the only thing keeping it out of every run was a hand-typed
flag. Knowledge that lives in a flag lives in one person's memory.
KNOWN_ISSUES.md §13a, DEBUGGING_STORY.md §12.


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

- ~~Once the real Phase 0.2 numbers arrive: update ARCHITECTURE.md §6,
  ROADMAP.md's Phase 0 status, and this file.~~ **Done** — the measured
  ceiling is batch size 256 under AMP; 128 is used in practice to leave
  headroom for validation. Recorded in docs/PHASE_0_REPORT.md.
- ~~Confirm whether the documentation rounds were pushed to GitHub.~~
  **Done** — everything is on `origin/main`; history was rewritten on
  2026-08-24 to remove 42 empty commits, with a backup at
  `backup-pre-rewrite`.
- ~~Re-profile VRAM once the real Phase 2 model exists.~~ Superseded by
  evidence from real runs: the binding constraint on this machine turned
  out to be **system RAM**, not VRAM, twice as often as VRAM
  (DEBUGGING_STORY.md §11). `num_workers` and `prefetch_factor` were cut
  from 4 to 2 as a result.

## Resolved Questions

Kept rather than deleted — the answers are part of the record.

- **Measured Phase 0.2 batch size?** 256 under AMP; 128 used, for
  validation headroom.
- **W&B or TensorBoard?** TensorBoard, exclusively. Nothing in `src/` or
  `scripts/` has ever imported wandb. It needs no account and writes
  locally, which suits a laptop that trains offline and gets
  interrupted. ARCHITECTURE.md §12 said the opposite until 2026-08-25.
- **Repository license?** MIT (`LICENSE`, 2026). Distinct from
  Flickr30k's own terms — see DATASETS.md.
- **Is single-machine Docker Compose still the Phase 7 target?** Yes.
  Both images build, the stack has been run end to end, and Kubernetes
  and managed endpoints are explicitly out of scope
  (docs/FUTURE_IDEAS.md). What remains is a host, not a design.

## Open Questions

- **Would a momentum encoder make the memory queue usable?** The A/B
  (EXPERIMENTS.md 006) shows the queue harming this model badly, but
  without a momentum encoder that is a statement about *this* queue, not
  about MoCo. Answering it costs a second copy of both towers in VRAM.
- **Where on the uniformity curve should the model sit?** Answered for
  now at weight 0.2 (EXPERIMENTS.md 009), but the term was applied by
  resuming into it. Training from scratch with it enabled is untested
  and is the obvious next comparison.
- **What is the model's ceiling on 31k images?** "Train longer" is
  exhausted (EXPERIMENTS.md 007) and the remaining levers — more data, a
  larger encoder — are both bounded by the constraint that defines the
  project.
