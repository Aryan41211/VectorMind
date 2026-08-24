# Results — VectorMind

Every number here is produced by `scripts/generate_reports.py`
in a single run against a single checkpoint. Regenerate with:

```bash
python scripts/generate_reports.py --checkpoint checkpoints/train/best_model.pt
```

**Checkpoint:** `checkpoints/train/best_model.pt` (epoch 11, step 11916)  
**Learned logit scale:** 24.33  
**Generated:** 2026-08-24 11:00 UTC

---

## Retrieval

Recall@K against the random-chance baseline for each direction.
Chance is computed as the complement of drawing K non-relevant
items without replacement, not the `k/n` shortcut, which
overstates chance when an image has five valid captions.

### Val split (3178 images, 15890 captions)

| Direction | K | Measured | Chance | vs chance |
|---|---|---|---|---|
| image → text | 1 | 7.17% | 0.03% | 228.0× |
| image → text | 5 | 20.01% | 0.16% | 127.3× |
| image → text | 10 | 29.30% | 0.31% | 93.2× |
| text → image | 1 | 5.64% | 0.03% | 179.2× |
| text → image | 5 | 16.94% | 0.16% | 107.6× |
| text → image | 10 | 24.92% | 0.31% | 79.2× |

### Test split (3179 images, 15895 captions)

| Direction | K | Measured | Chance | vs chance |
|---|---|---|---|---|
| image → text | 1 | 7.71% | 0.03% | 245.0× |
| image → text | 5 | 20.60% | 0.16% | 131.1× |
| image → text | 10 | 28.91% | 0.31% | 92.0× |
| text → image | 1 | 5.71% | 0.03% | 181.4× |
| text → image | 5 | 17.33% | 0.16% | 110.2× |
| text → image | 10 | 25.20% | 0.31% | 80.1× |

## Embedding health

Recall@K alone cannot tell you whether a contrastive model has
learned a usable space. Phase 4 shipped a checkpoint whose
embeddings all sat inside a narrow cone at separation 0.094,
and whose report called it HEALTHY. These are the numbers that
would have caught it.

| Metric | Value | Healthy |
|---|---|---|
| Matched similarity | 0.6036 | high |
| Unmatched similarity | 0.2569 | ≈ 0 |
| **Separation** | **0.3467** | **large** |
| Mean image–image cosine | 0.3452 | ≈ 0 |
| Mean text–text cosine | 0.3859 | ≈ 0 |
| ‖mean image embedding‖ | 0.5877 | ≈ 0 |
| Per-dimension variance | 0.002558 | — |

**Grade:** ANISOTROPIC

ANISOTROPIC — separation 0.347 is above the 0.25 floor, so retrieval is meaningful, but ||mean embedding|| 0.621 > 0.5

Reference points from this project's own runs: the Phase 3.5
tiny-subset overfit reached separation 0.964; the unclamped
Phase 4 checkpoint measured 0.094. See
[KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md) §1.
