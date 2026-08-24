# Results — VectorMind

Every number here is produced by `scripts/generate_reports.py`
in a single run against a single checkpoint. Regenerate with:

```bash
python scripts/generate_reports.py --checkpoint checkpoints/train/best_model.pt
```

**Checkpoint:** `checkpoints/train/best_model.pt` (epoch 10, step 10923)  
**Learned logit scale:** 23.38  
**Generated:** 2026-08-24 08:27 UTC

---

## Retrieval

Recall@K against the random-chance baseline for each direction.
Chance is computed as the complement of drawing K non-relevant
items without replacement, not the `k/n` shortcut, which
overstates chance when an image has five valid captions.

### Val split (3178 images, 15890 captions)

| Direction | K | Measured | Chance | vs chance |
|---|---|---|---|---|
| image → text | 1 | 5.92% | 0.03% | 188.0× |
| image → text | 5 | 17.68% | 0.16% | 112.5× |
| image → text | 10 | 25.77% | 0.31% | 82.0× |
| text → image | 1 | 5.00% | 0.03% | 159.0× |
| text → image | 5 | 15.36% | 0.16% | 97.6× |
| text → image | 10 | 23.14% | 0.31% | 73.5× |

### Test split (3179 images, 15895 captions)

| Direction | K | Measured | Chance | vs chance |
|---|---|---|---|---|
| image → text | 1 | 6.29% | 0.03% | 200.0× |
| image → text | 5 | 17.77% | 0.16% | 113.1× |
| image → text | 10 | 25.64% | 0.31% | 81.6× |
| text → image | 1 | 5.06% | 0.03% | 160.8× |
| text → image | 5 | 15.50% | 0.16% | 98.6× |
| text → image | 10 | 23.07% | 0.31% | 73.3× |

## Embedding health

Recall@K alone cannot tell you whether a contrastive model has
learned a usable space. Phase 4 shipped a checkpoint whose
embeddings all sat inside a narrow cone at separation 0.094,
and whose report called it HEALTHY. These are the numbers that
would have caught it.

| Metric | Value | Healthy |
|---|---|---|
| Matched similarity | 0.5993 | high |
| Unmatched similarity | 0.2551 | ≈ 0 |
| **Separation** | **0.3442** | **large** |
| Mean image–image cosine | 0.3536 | ≈ 0 |
| Mean text–text cosine | 0.3846 | ≈ 0 |
| ‖mean image embedding‖ | 0.5948 | ≈ 0 |
| Per-dimension variance | 0.002525 | — |

**Grade:** ANISOTROPIC

ANISOTROPIC — separation 0.344 is above the 0.25 floor, so retrieval is meaningful, but ||mean embedding|| 0.620 > 0.5

Reference points from this project's own runs: the Phase 3.5
tiny-subset overfit reached separation 0.964; the unclamped
Phase 4 checkpoint measured 0.094. See
[KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md) §1.
