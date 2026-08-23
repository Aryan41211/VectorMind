# The Debugging Story — VectorMind

Every project has things that break. This document records what went
wrong during VectorMind's development, how each issue was found, and
how it was fixed. Written for portfolio readers and interviewers who
want to see engineering judgment, not just green test suites.

---

## 1. The Test/Val Discrepancy That Wasn't a Bug

**What happened:** After completing Phase 5 evaluation, the test
Recall@10 (19.63%) looked slightly lower than validation (20.23%).
The question: was this a genuine generalization gap, or a bug?

**How it was found:** While auditing `evaluate_test_set.py`, the
destructuring `_, eval_loader, _` was noticed — it unpacked three
values from `create_dataloaders()`, which returns five. The third
value (`eval_loader`) was actually the *val* loader, not the test
loader. The test evaluation had been running on the validation set
all along, silently.

**How it was fixed:** Changed the destructuring to
`_, _, test_loader, _, _` to correctly extract the test loader. The
"discrepancy" was actually the correct test metrics being revealed
for the first time: R@1=4.62%, R@5=13.43%, R@10=19.63%.

**Lesson:** A suspiciously small discrepancy between val and test
metrics should trigger code inspection, not just acceptance. The bug
was a single underscore in a destructuring pattern — easy to miss in
review, impossible to miss if you ask "why are these numbers so
close?"

---

## 2. The Memory Queue "Bug" That Was Just a Missing Flag

**What happened:** A code review flagged the memory queue as
"always disabled" because of `.detach()` on embeddings. The concern:
detaching prevents gradients from flowing through the queue, making
it useless.

**How it was found:** Manual code review during the Phase 6 audit.

**What actually happened:** `.detach()` is *correct* — it prevents
gradient computation through stale embeddings (which would be wrong
and waste memory). The queue was disabled because the CLI flag
`--use-memory-queue` wasn't being passed during the training run,
not because of `.detach()`. The code was correct; the invocation was
incomplete.

**How it was fixed:** Added `--use-memory-queue` to the training
command. R@10 improved by 18.2%.

**Lesson:** "This code looks wrong" and "this code is being invoked
wrong" are different hypotheses. Check both before changing code.

---

## 3. The Async Blocking Disaster

**What happened:** The FastAPI search endpoints (`/search/text` and
`/search/image`) were defined as `async def` but contained
synchronous blocking operations: tokenization, PyTorch inference,
and FAISS search. This blocked the event loop, causing the Vite
proxy to timeout with HTTP 502 Bad Gateway.

**How it was found:** The frontend showed "Search Error: 502 Bad
Gateway" on every query. The backend logs showed no errors — it was
processing requests, just not concurrently.

**How it was fixed:** Wrapped blocking calls in
`asyncio.to_thread()` to run them in a thread pool, freeing the
event loop. Alternatively, the endpoints could be changed to `def`
(sync), but `to_thread` preserves the async routing pattern for
future use.

**Lesson:** FastAPI's `async def` is not just a signature — it
changes the execution model. Any blocking call inside an async
endpoint blocks the entire event loop. Always check: is this
endpoint actually doing I/O, or is it doing CPU work?

---

## 4. The `encode_text` Signature Mismatch

**What happened:** `text_search.py` called
`app_state.model.encode_text(text_input)` where `text_input` was a
HuggingFace tokenizer output dictionary
(`{"input_ids": tensor, "attention_mask": tensor}`). But
`encode_text()` expects separate tensor arguments.

**How it was found:** Runtime error — the model raised a TypeError
when the dict was passed instead of tensors.

**How it was fixed:** Unpacked the dictionary into separate
arguments:
```python
app_state.model.encode_text(
    text_input["input_ids"],
    text_input["attention_mask"]
)
```

**Lesson:** The API contract between the serving layer and the model
was never formally tested. A simple integration test (pass a real
tokenizer output through `encode_text`) would have caught this
during Phase 6 development, not during manual testing.

---

## 5. The Stale Metric Scattered Across 13 Files

**What happened:** The number `20.26` appeared in 13 files across
the repository — README, training logs, project status, evaluation
reports. It was the *validation* Recall@10, but it was being cited
in contexts where the *test* Recall@10 (19.63%) was appropriate, or
where neither metric applied.

**How it was found:** A grep for `20.26` across the entire repo
after fixing the test eval bug revealed the scope of the problem.

**How it was fixed:** Each occurrence was individually assessed:
- Contexts referring to validation metrics → changed to `20.23%`
  (the corrected val number)
- Contexts referring to test metrics → changed to `19.63%`
- Contexts where neither applied → removed or reworded

**Lesson:** Metrics are not just numbers — they have context (val
vs test, which checkpoint, which epoch). A metric without its context
is misleading. When the ground truth changes, grep is your friend.

---

## 6. The Missing `python-multipart` Dependency

**What happened:** The FastAPI image search endpoint uses
`UploadFile`, which requires `python-multipart`. It wasn't in
`requirements.txt`. It worked locally because it was installed as a
transitive dependency, but a fresh `pip install -r requirements.txt`
would have failed on image upload.

**How it was found:** Manually reviewing `requirements.txt` against
actual imports during Phase 6.5.

**How it was fixed:** Added `python-multipart>=0.0.6` to
`requirements.txt` under a new "Serving" section.

**Lesson:** Transitive dependencies are not guaranteed. Every direct
import should have a direct dependency entry. `pip install` in a
fresh environment is the only reliable test.

---

## 7. The SPA Catch-All Breaking Backend Tests

**What happened:** Adding a SPA catch-all route
`/{full_path:path}` to serve `index.html` for client-side routing
caused two existing tests to fail:
- `test_root_returns_api_info` expected JSON from `GET /`, got HTML
- `test_404_for_unknown_endpoint` expected 404 from `GET /nonexistent`,
  got 200 (SPA serves index.html for all paths)

**How it was found:** Running the full test suite after the change.

**How it was fixed:**
- Renamed the API info endpoint to `GET /api/info` (separation of
  concerns — API metadata vs static serving)
- Updated tests to reflect new routing behavior: unknown paths now
  serve the SPA (correct for client-side routing)

**Lesson:** Adding routes has test side effects. A catch-all route
changes the semantics of "unknown path" — it's no longer an error,
it's a SPA navigation. Tests must match the new contract.

---

*Last updated: Phase 7 (ROADMAP.md)*

---

## 8. The Healthy Embedding Space That Wasn't

**What happened:** `reports/phase5_embedding_diagnostics.json` said
`"overall_status": "HEALTHY"`, `"collapse_risk": "LOW"`, and reported a
matched-vs-unmatched similarity separation of `0.33`. The model was
shipped, indexed, and served on that basis.

**How it was found:** by not trusting the file. The shipped
`image_embeddings.npy` was still on disk, so the claim was directly
checkable:

```python
cross = image_embeds @ text_embeds.T
matched   = cross.diagonal().mean()          # 0.937
unmatched = off_diagonal(cross).mean()       # 0.843
separation = matched - unmatched             # 0.094, not 0.33
```

Mean off-diagonal image-image cosine was **0.810**, and the norm of the
mean embedding was **0.900** on a scale where 1.0 is total collapse.
Every embedding sat inside a narrow cone. The reported `0.33` did not
reproduce from any subset or seed — it appears to have been written
rather than computed.

**Why it went unnoticed:** the only automated health metric was
per-dimension variance, and variance is the wrong instrument. Embeddings
are L2-normalized onto the unit hypersphere, so per-dimension variance
conflates "spread out" with "spread across dimensions" — a cone of
vectors all pointing the same way still shows nonzero variance. The
recorded 0.00075 looked merely "slightly below threshold" and was
explained away in the report's own notes field.

**The fix:** `src/vectormind/evaluation/embedding_health.py`, which
measures the questions that actually matter — separation, mean
off-diagonal cosine, and the norm of the mean embedding — and is logged
beside Recall@K at every validation. Its test suite includes a synthetic
cone tuned to reproduce the real Phase 4 geometry (mean cosine 0.813
against the measured 0.810), so a regression fails a test rather than a
shipped report.

**Lesson:** a metric that cannot fail is not a check. Variance never
crossed its threshold, so it never raised an alarm — and its inability
to raise one was read as evidence of health.

---

## 9. The Mitigation That Was the Cause

**What happened:** `ARCHITECTURE.md` §6 named the MoCo-style memory
queue "the key mitigation for negative sample count".
`docs/TRAINING_LOG.md` credited it with lifting Recall@10 from 17.12% to
20.23%, +18.2%. Separately, the same log recorded "embedding collapse
after epoch 7" and attributed it to "temperature overgrowth" — the
learned logit scale running from 55 to over 500.

Two entries, two pages apart, describing one phenomenon.

**How it was found:** by re-running the experiment properly. The
retrained model improved monotonically for six epochs — R@10 2.99 to
17.46%, separation 0.102 to 0.329 — and then the queue activated at
epoch 7:

| Epoch 7 from `epoch_006.pt` | Queue active | Queue inactive |
|---|---|---|
| Val R@10 | 10.51% | **19.63%** |
| Separation | 0.062 | **0.322** |
| Logit scale | 67.6 | **18.6** |

One epoch of queue negatives undid six epochs of improvement.

**Why the original conclusion survived.** Three failures compounding,
and this is the interesting part:

1. **The measurement window hid the effect.** 20.23% was epoch 7, the first epoch after activation. Recall lags a collapse by several epochs, so the metric was read at exactly the moment the queue looked good.
2. **It was never a controlled comparison.** `--no-queue` substituted a size-1 stub queue, and `load_checkpoint` rejects a size-1 queue against a 4096-entry checkpoint — so the baseline arm could only ever run *from scratch*. The recorded "A/B" was epoch 6 of one run against epoch 7 of another. Nobody noticed, because the flag appeared to work.
3. **The contradicting evidence was filed under a different heading.** The collapse was observed, measured, and written down — as a separate problem. Two true observations, never connected.

**The mechanism:** MoCo pairs its queue with a momentum encoder, whose
slow EMA update keeps queued keys comparable to the live encoder's
output. This implementation borrowed the queue and not the momentum
encoder. With 4,096 stale negatives against a batch of 128, the cheapest
way to lower the loss is to sharpen the similarity distribution rather
than improve the representation — and an unbounded logit scale is the
cheapest way to sharpen. The scale runs away, the space collapses,
recall falls.

**The fix:** disable the queue; clamp the logit scale at CLIP's 100 as a
belt-and-braces bound on the symptom. `--no-queue` now deactivates the
queue instead of stubbing it, so the comparison is actually runnable.

**Lesson:** when borrowing a technique, identify which part is
load-bearing. The queue is the visible half of MoCo. The momentum
encoder is the half that makes it work.

---

## 10. The Resume That Ate the Better Checkpoint

**What happened:** a resumed run finished epoch 7 at 10.51% Recall@10
and saved it as `best_model.pt`, overwriting a checkpoint at 17.46%.

**Cause:** one line.

```python
best_val_recall10 = 0.0     # unconditional, executed after --resume too
```

Best-so-far reset to zero on every resume, so the first epoch after a
resume always won the comparison — whatever its score. Nothing in the
log said anything was wrong; the run reported "New best model saved" and
was telling the truth about what it had computed.

**Why it was survivable:** the periodic `epoch_NNN.pt` saves still held
the good weights. That was luck, not design.

**The fix:** `save_checkpoint` now records the validation metrics that
earned a checkpoint, and a resumed run reads the best-so-far back out of
`best_model.pt`. Checkpoints written before the change have no metrics
block, so the reader warns and returns a default rather than failing —
and a corrupt file cannot stop training from starting.

**Lesson:** "best" is state, and resuming restores state. Every variable
that survives across epochs has to survive across restarts too, or the
restart quietly resets the thing it was supposed to preserve.

---

## 11. The GPU That Also Draws Your Desktop

**What happened:** training died twice in ninety minutes, at epoch 5 and
epoch 7, with peak usage around 4.6GB of the card's 6GB.

The first was `CUDA error: out of memory`. The second was
`cuDNN error: CUDNN_STATUS_INTERNAL_ERROR_HOST_ALLOCATION_FAILED` — and
that one is not a VRAM failure at all. It is cuDNN failing to allocate
**system** RAM, with 4.5GB free of 16GB, because the dataloader was
holding 1.15GB of *pinned* buffers that cannot be swapped out:

```
num_workers x prefetch_factor x batch_size x 3 x 224 x 224 x 4 bytes
     4      x        4        x     128                    = 1.15 GB
```

**Why it matters:** the project's stated constraint, repeated throughout
its documentation, is 6GB of VRAM. On this machine the binding
constraint turned out to be system RAM twice as often — because the same
GPU drives the display, and the browser, editor, and compositor take
what they need without asking.

**The fix:** `src/vectormind/training/oom.py` releases torch's allocator
cache (which it otherwise holds rather than returning to the driver) and
retries the step. Detection matches on message as well as type, because
torch reports allocation failures as `OutOfMemoryError`,
`AcceleratorError`, or a bare `RuntimeError` depending on the path — and
the host-side message contains no "out of memory" at all, which is why
the first version of this fix did not catch the second failure.

A genuine capacity problem is still fatal, and the message distinguishes
the two cases: a VRAM shortage needs `batch_size`, a RAM shortage needs
`num_workers` and `prefetch_factor`. Those were cut 4 to 2, taking the
pinned footprint to 0.29GB.

**Lesson:** two things. Losing hours of training to a neighbouring
process is an infrastructure failure, not a modelling one, and it should
not be fatal. And the constraint you wrote on the tin is not necessarily
the one that stops you.
