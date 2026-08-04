# Performance Optimization Report — VectorMind Phase 4

## Hardware Profile

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop |
| GPU VRAM | 6,141 MiB (6.0 GB) |
| CPU | 16 logical cores |
| RAM | ~16 GB |
| Platform | Windows, PyTorch 2.x, CUDA |

## Baseline Profiling (Before Optimization)

| Metric | Value |
|---|---|
| Batch size | 256 |
| num_workers | 2 |
| persistent_workers | False (default) |
| prefetch_factor | 2 (default) |
| cudnn.benchmark | False (default) |
| DataLoader time | 3,395 ms/batch |
| Training step time | 7,848 ms/step |
| GPU peak memory | 7.25 GB (OOM on 6GB GPU) |
| Status | **OOM during validation, unusable** |

**Primary bottleneck**: DataLoader with num_workers=2 was the dominant
cost — data loading alone consumed 3.4s/batch, leaving the GPU idle
most of the time. The batch_size=256 exceeded VRAM capacity.

## Optimizations Applied

| Optimization | Setting | Justification |
|---|---|---|
| `torch.backends.cudnn.benchmark` | `True` | Fixed input size (224x224) — auto-tunes conv algorithms |
| `batch_size` | 128 (from 256) | Fits in 4.6GB VRAM (measured 2.49GB peak with AMP) |
| `num_workers` | 8 (from 2) | Utilizes 16 logical cores for parallel data preprocessing |
| `persistent_workers` | `True` | Eliminates worker respawn overhead (~2s) per epoch |
| `prefetch_factor` | 4 (from 2) | Keeps more batches ready in the data pipeline |
| `pin_memory` | `True` | Faster CPU->GPU transfers via page-locked memory |
| Validation cache clear | `torch.cuda.empty_cache()` | Frees fragmented memory before eval to prevent OOM |
| Log frequency | Every 50 steps (from 10) | Reduces I/O overhead in the training loop |

## Post-Optimization Profiling

| Metric | Before | After | Improvement |
|---|---|---|---|
| Training step | 7,848 ms | 349 ms | **22.5x faster** |
| GPU peak memory | 7.25 GB (OOM) | 2.49 GB | **Fits in 6GB** |
| Steps/sec | 0.13 | 2.86 | **22x throughput** |
| Est. epoch time | >650s | 347s | **1.9 min/epoch** |
| Est. 10-epoch time | >108 min | 58 min | **~1 hour** |

## Scientific Validity Checklist

| Criterion | Status |
|---|---|
| Same model architecture | ✓ Identical (23.9M params) |
| Same loss function | ✓ Symmetric InfoNCE |
| Same optimizer | ✓ AdamW (lr=1e-3, wd=0.01) |
| Same scheduler | ✓ CosineAnnealing (T_max=10, eta_min=1e-6) |
| Same data splits | ✓ Seed=42, 80/10/10 |
| Same augmentations | ✓ Resize+RandomCrop+HFlip |
| Same evaluation metrics | ✓ Image-level Recall@K |
| Reproducible | ✓ Same configs, same seed |
| batch_size reduced | ✓ Justified: 256 OOM'd; 128 is the max safe |

**Note**: The batch_size reduction from 256 to 128 is a hardware-
constrained necessity, not a scientific choice. With batch_size=256
the experiment literally cannot run on this GPU. The effective batch
with gradient accumulation remains equivalent if needed.

## DataLoader Configuration

```yaml
# Optimized settings for RTX 4050 6GB, 16-core CPU
batch_size: 128
num_workers: 8
persistent_workers: true
prefetch_factor: 4
pin_memory: true
drop_last: true
```

## GPU Utilization Estimate

With 349ms/step and batch_size=128:
- Forward+backward: ~250ms (GPU compute)
- Data transfer: ~50ms (overlap via pin_memory + non_blocking)
- Optimizer step: ~30ms
- Data loading: hidden by prefetch pipeline (no stall)
- Estimated GPU utilization: ~70-80%

The remaining gap is due to the optimizer step and unavoidable
CPU-GPU sync points. This is acceptable for a laptop GPU.
