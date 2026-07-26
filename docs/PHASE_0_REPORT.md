# Phase 0 — Environment Setup & Profiling Report

## Versions
- Python: 3.12.10
- CUDA (driver / toolkit): 13.0 / 12.1
- PyTorch: 2.3.1+cu121
- Torchvision: 0.18.1+cu121
- Transformers: 5.14.1
- Tokenizers: 0.22.2
- FAISS: 1.14.3 (faiss-cpu)
- OpenCV: 5.0.0
- Pillow: 12.2.0
- FastAPI: 0.140.0
- Uvicorn: 0.51.0
- PyYAML: 6.0.3
- W&B: 0.28.1
- pytest: 9.1.1
- Node: 22.17.1 (optional/not yet required)
- npm: 10.9.2 (optional/not yet required)
- Git: 2.50.1.windows.1

## Hardware
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU
- VRAM (total): 6.44 GB
- VRAM ceiling used for batch search: 5.2 GB (10% safety margin on 6.44 GB)

## Installed dependencies
```
annotated-doc-0.0.4
annotated-types-0.8.0
anyio-4.14.2
certifi-2026.7.22
charset-normalizer-3.4.9
click-8.4.2
colorama-0.4.6
faiss-cpu-1.14.3
fastapi-0.140.0
h11-0.16.0
hf-xet-1.5.2
httpcore-1.0.9
httpx-0.28.1
huggingface-hub-1.24.0
idna-3.18
iniconfig-2.3.0
intel-openmp-2021.4.0
markdown-it-py-4.2.0
mdurl-0.1.2
mkl-2021.4.0
networkx-3.6.1
numpy-2.4.4
opencv-python-5.0.0.93
pillow-12.2.0
platformdirs-4.11.0
pluggy-1.6.0
protobuf-7.35.1
pydantic-2.13.4
pydantic-core-2.46.4
pygments-2.20.0
pytest-9.1.1
python-dotenv-1.0.1
pyyaml-6.0.3
regex-2026.7.19
requests-2.34.2
rich-15.0.0
safetensors-0.8.0
sentry-sdk-2.66.1
shellingham-1.5.4
starlette-1.3.1
sympy-1.13.1
tokenizers-0.22.2
torch-2.3.1+cu121
torchvision-0.18.1+cu121
tbb-2021.13.1
tqdm-4.69.1
typer-0.27.0
typing-extensions-4.15.0
typing-inspection-0.4.2
urllib3-2.7.0
uvicorn-0.51.0
wandb-0.28.1
```

## Missing / skipped dependencies
- faiss-gpu: skipped (faiss-cpu installed for portability; FAISS only needed in Phase 6)
- Node/npm: installed but not yet required (Phase 6.5+)
- transformers: installed but note — only tokenizer utilities will be used; no pretrained model weights loaded (per ARCHITECTURE.md §3)

## Environment verification
| Check | Status | Details |
|-------|--------|---------|
| Python version == 3.12.x | PASS | 3.12.10 |
| torch import | PASS | 2.3.1+cu121 |
| torch.cuda.is_available() | PASS | True |
| torch.cuda.get_device_name(0) | PASS | NVIDIA GeForce RTX 4050 Laptop GPU |
| torch.cuda.get_device_properties(0).total_memory | PASS | 6.44 GB |
| Mixed precision (torch.amp.autocast) | PASS | Works on CUDA |
| torchvision | PASS | 0.18.1+cu121 |
| transformers | PASS | 5.14.1 (tokenizer only) |
| tokenizers | PASS | 0.22.2 |
| faiss | PASS | 1.14.3 (CPU build) |
| cv2 | PASS | 5.0.0 |
| PIL | PASS | 12.2.0 |
| fastapi | PASS | 0.140.0 |
| uvicorn | PASS | 0.51.0 |
| pyyaml | PASS | 6.0.3 |
| wandb | PASS | 0.28.1 (import only, no auth) |
| pytest | PASS | 9.1.1 |

## Mixed precision
- torch.amp available: yes
- autocast verified on a tiny forward: yes (ran `torch.randn(2,2,device="cuda") @ x.T` under `autocast(device_type="cuda", enabled=True)`)

## Batch size benchmark
- Image shape: [B, 3, 224, 224]
- Text shape: [B, 77]
- AMP: on (torch.amp.autocast)
- Max safe batch size: 256
- Peak VRAM at that batch: 5.00 GB
- Search method: binary search, 5.2 GB ceiling (10% safety margin on 6.44 GB total)
- Recommended memory-queue size (pending Phase 3.5): 4096 (16× batch for negative diversity)
- Notes: Batch 512 peaked at 9.74 GB (exceeds ceiling); batch 1024 OOM. The 10% safety margin reserves ~0.64 GB for memory queue, dataloader pinned memory, and allocator fragmentation.

## Git
- Branch: main
- Remote: https://github.com/Aryan41211/VectorMind.git
- .gitignore created: yes
- Files staged for commit this phase:
  - .gitignore
  - README.md
  - ROADMAP.md
  - ARCHITECTURE.md
  - configs/profiling.yaml
  - scripts/verify_env.py
  - requirements.txt (updated with pinned versions)
  - requirements.lock.txt
  - docs/PHASE_0_REPORT.md

## Warnings
- transformers 5.14.1 warns "PyTorch >= 2.4 is required" but found 2.3.1+cu121. Only tokenizer utilities are used (no pretrained model weights loaded), so this is acceptable for now.
- faiss-cpu loaded without AVX2 support (falls back to non-AVX2 build). Acceptable for Phase 6 indexing on this hardware.
- NumPy 2.4.4 warning about modules compiled with NumPy 1.x API. No crashes observed.
- Node/npm installed but not required until Phase 6.5.

## Remaining issues / blockers
- none