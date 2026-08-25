# Results — VectorMind

Every number here is produced by `scripts/generate_reports.py`
in a single run against a single checkpoint. Regenerate with:

```bash
python scripts/generate_reports.py --checkpoint checkpoints/train/best_model.pt
```

**Checkpoint:** `checkpoints/train/best_model.pt` (epoch 15, step 14895)  
**Learned logit scale:** 17.07  
**Generated:** 2026-08-25 13:46 UTC

---

## Retrieval

Recall@K against the random-chance baseline for each direction.
Chance is computed as the complement of drawing K non-relevant
items without replacement, not the `k/n` shortcut, which
overstates chance when an image has five valid captions.

### Val split (3178 images, 15890 captions)

| Direction | K | Measured | Chance | vs chance |
|---|---|---|---|---|
| image → text | 1 | 6.80% | 0.03% | 216.0× |
| image → text | 5 | 20.04% | 0.16% | 127.5× |
| image → text | 10 | 29.17% | 0.31% | 92.8× |
| text → image | 1 | 5.71% | 0.03% | 181.6× |
| text → image | 5 | 17.72% | 0.16% | 112.6× |
| text → image | 10 | 26.23% | 0.31% | 83.4× |

### Test split (3179 images, 15895 captions)

| Direction | K | Measured | Chance | vs chance |
|---|---|---|---|---|
| image → text | 1 | 7.64% | 0.03% | 243.0× |
| image → text | 5 | 20.79% | 0.16% | 132.3× |
| image → text | 10 | 28.91% | 0.31% | 92.0× |
| text → image | 1 | 5.98% | 0.03% | 190.0× |
| text → image | 5 | 18.24% | 0.16% | 116.0× |
| text → image | 10 | 26.22% | 0.31% | 83.4× |

## Embedding health

Recall@K alone cannot tell you whether a contrastive model has
learned a usable space. Phase 4 shipped a checkpoint whose
embeddings all sat inside a narrow cone at separation 0.094,
and whose report called it HEALTHY. These are the numbers that
would have caught it.

| Metric | Value | Threshold |
|---|---|---|
| Matched similarity | 0.4920 | high |
| Unmatched similarity | 0.0105 | ≈ 0 |
| **Separation** | **0.4815** | > 0.25 |
| Mean image–image cosine | 0.0268 | < 0.5 |
| Mean text–text cosine | 0.0130 | < 0.5 |
| ‖mean image embedding‖ | 0.1646 | < 0.5 |
| ‖mean text embedding‖ | 0.1155 | < 0.5 |
| Per-dimension variance | 0.003802 | — |

The two norm rows are graded as their maximum, which is why the
verdict below quotes the larger of them.

**Grade:** HEALTHY

HEALTHY — separation 0.481, max mean-cosine 0.027

Reference points from this project's own runs: the Phase 3.5
tiny-subset overfit reached separation 0.964; the unclamped
Phase 4 checkpoint measured 0.094. See
[KNOWN_ISSUES.md](../docs/KNOWN_ISSUES.md) §1.
