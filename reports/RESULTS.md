# Results — VectorMind

Every number here is produced by `scripts/generate_reports.py`
in a single run against a single checkpoint. Regenerate with:

```bash
python scripts/generate_reports.py --checkpoint checkpoints/train/best_model.pt
```

**Checkpoint:** `checkpoints/train/best_model.pt` (epoch 9, step 9930)  
**Learned logit scale:** 22.10  
**Generated:** 2026-08-23 20:39 UTC

---

## Retrieval

Recall@K against the random-chance baseline for each direction.
Chance is computed as the complement of drawing K non-relevant
items without replacement, not the `k/n` shortcut, which
overstates chance when an image has five valid captions.

### Val split (3178 images, 15890 captions)

| Direction | K | Measured | Chance | vs chance |
|---|---|---|---|---|
| image → text | 1 | 5.63% | 0.03% | 179.0× |
| image → text | 5 | 15.89% | 0.16% | 101.1× |
| image → text | 10 | 23.06% | 0.31% | 73.4× |
| text → image | 1 | 4.56% | 0.03% | 145.0× |
| text → image | 5 | 14.40% | 0.16% | 91.5× |
| text → image | 10 | 21.94% | 0.31% | 69.7× |

### Test split (3179 images, 15895 captions)

| Direction | K | Measured | Chance | vs chance |
|---|---|---|---|---|
| image → text | 1 | 6.04% | 0.03% | 192.0× |
| image → text | 5 | 16.04% | 0.16% | 102.1× |
| image → text | 10 | 23.91% | 0.31% | 76.1× |
| text → image | 1 | 5.06% | 0.03% | 161.0× |
| text → image | 5 | 14.63% | 0.16% | 93.0× |
| text → image | 10 | 21.53% | 0.31% | 68.4× |

## Embedding health

Recall@K alone cannot tell you whether a contrastive model has
learned a usable space. Phase 4 shipped a checkpoint whose
embeddings all sat inside a narrow cone at separation 0.094,
and whose report called it HEALTHY. These are the numbers that
would have caught it.

| Metric | Value | Healthy |
|---|---|---|
| Matched similarity | 0.5928 | high |
| Unmatched similarity | 0.2633 | ≈ 0 |
| **Separation** | **0.3295** | **large** |
| Mean image–image cosine | 0.3834 | ≈ 0 |
| Mean text–text cosine | 0.3765 | ≈ 0 |
| ‖mean image embedding‖ | 0.6194 | ≈ 0 |
| Per-dimension variance | 0.002408 | — |

**Grade:** ANISOTROPIC

ANISOTROPIC — separation 0.330 is above the 0.25 floor, so retrieval is meaningful, but ||mean embedding|| 0.619 > 0.5

Reference points from this project's own runs: the Phase 3.5
tiny-subset overfit reached separation 0.964; the unclamped
Phase 4 checkpoint measured 0.094. See
[KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md) §1.
