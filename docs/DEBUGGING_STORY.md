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
