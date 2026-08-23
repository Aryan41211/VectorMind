# Deployment

How to run VectorMind outside a development shell, and what has actually
been verified versus what has only been written.

**Verification status (2026-08-24): both images built and the full
stack run end to end on this machine.** What was actually observed:

| Check | Result |
|---|---|
| Backend image builds | ✅ 2.17GB, with `import backend.app` as a build gate |
| Frontend image builds | ✅ 93.1MB, with `tsc`, `oxlint` and `nginx -t` as build gates |
| Backend reaches healthy | ✅ `/ready` 200 — model, both indices and both index maps loaded |
| Index served correctly | ✅ `num_indexed_images: 3179`, not the pre-fix 15,895 |
| SPA served through nginx | ✅ `GET /` → 200 |
| API proxied through nginx | ✅ `/health`, `/ready`, `/search/text` all 200 same-origin |
| Search returns distinct images | ✅ 10 unique filenames of 10 results, across five queries |
| Warm latency | ✅ 8–19ms through the proxy |
| Cold first query | 2.07s — the tokenizer and first forward pass |
| Security headers | ✅ all four present, exactly once |
| Request correlation | ✅ `X-Request-ID` on every response |
| Rate limiting | ✅ 429 after the configured 30/minute |

Still not verified: CI has never been observed green on GitHub, and
there is no public deployment.

---

## What runs where

```
                  :8080
                     │
        ┌────────────▼────────────┐
        │  frontend (nginx)       │
        │  · serves the built SPA │
        │  · proxies /search      │
        │    /health /ready       │
        │    /images /docs        │
        └────────────┬────────────┘
                     │ compose network
        ┌────────────▼────────────┐
        │  backend (uvicorn)      │
        │  · FastAPI + model      │
        │  · FAISS indices        │
        │  · serves dataset images│
        └─────────────────────────┘
                     │ read-only volume mounts
        checkpoints/ · backend/indices/ · data/raw/flickr30k/images/
```

Everything is reached through the one published port. The browser never
talks to the backend directly, so the deployed app is same-origin and
CORS is not part of the deployment at all.

---

## Prerequisites

The backend image is built from source, but three artifacts are mounted
rather than baked in — the checkpoint alone is ~278MB and the image set
is 1.3GB:

| Path | How to obtain |
|---|---|
| `checkpoints/train/best_model.pt` | `python scripts/train.py` |
| `backend/indices/` | `python -m backend.index_builder --checkpoint checkpoints/train/best_model.pt` |
| `data/raw/flickr30k/images/` | See [DATASETS.md](DATASETS.md) |

The index must be rebuilt whenever the checkpoint changes. Embeddings
from one model and an index from another produce confident, wrong
results — the app validates that each index map matches its index, but it
cannot tell that an index came from a different checkpoint.

---

## Running it

```bash
docker compose -f deployment/docker-compose.yml up --build
```

Then open <http://localhost:8080>.

Port 80 is frequently already taken on a development machine, so the
frontend publishes on 8080 by default. Override it if that clashes too:

```bash
FRONTEND_PORT=8081 docker compose -f deployment/docker-compose.yml up --build
```

The backend healthcheck probes `/ready`, and the frontend waits on it, so
`up` does not report healthy until the model and both indices are loaded.
Expect roughly 30–90 seconds on first start; `start_period: 90s` exists
so the container is not killed while starting normally.

### Without Docker

```bash
# Backend
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Frontend, in another shell
cd frontend && npm run dev     # http://localhost:3000, proxies to :8000
```

---

## Probes

Two endpoints, answering different questions. Confusing them is the
usual cause of a deployment that returns 503 to every early request.

| Endpoint | Question | Returns |
|---|---|---|
| `/health` | Is the process alive? | 200 as soon as the app is up |
| `/ready` | Can it serve searches? | 503 until model **and** both indices load |

Route traffic on `/ready`. `/health` is for restart decisions.

`/ready` names the specific artifact that is missing, so a wrong volume
mount is diagnosable from the probe alone:

```json
{
  "ready": false,
  "model_loaded": true,
  "image_index_loaded": false,
  "text_index_loaded": false,
  "index_maps_loaded": false
}
```

That response means the checkpoint mounted and the index directory did
not.

---

## Configuration

All of it lives in `configs/serving.yaml` — paths, CORS origins, request
limits, and tokenizer settings. Nothing is read from the environment
except `PYTHONUNBUFFERED`.

Two values to revisit before exposing this publicly:

- `cors.allow_origins` lists localhost only. Add the real origin, or leave it — with the nginx proxy the app is same-origin and CORS is unused.
- `limits.rate_limit_requests` is 30/minute per client. It is enforced **in-process**, so N workers means N times the budget. Run one worker, or put a real limiter upstream.

---

## Known gaps

Honest list of what this deployment is not.

1. **No TLS.** Nothing here terminates HTTPS, and the security headers deliberately omit HSTS, which belongs on whatever does. Put this behind a TLS terminator before it faces the internet.
2. **Single machine, single worker.** The rate limiter holds per-process state, so horizontal scaling silently multiplies the effective limit. Kubernetes and managed endpoints are explicitly out of scope — see [FUTURE_IDEAS.md](FUTURE_IDEAS.md).
3. **No metrics or tracing.** Structured logs with request ids are all the observability there is. Adequate for a demo, not for anything on call.
4. **CPU inference.** The image installs CPU torch, and only the packages serving actually imports (`requirements-serving.txt`). Measured through the proxy: 8-19ms warm, 2.07s on the first query while the tokenizer and first forward pass warm up. Fine at this corpus size; GPU serving would need the CUDA base image and a runtime with device access.
5. **No authentication.** Every endpoint is public. There is nothing to protect but the GPU, which is what the rate limit is for.

---

## Troubleshooting

**`/ready` never returns 200.** Check which artifact it names as missing,
then check the corresponding volume mount. The app deliberately starts in
a degraded state rather than crash-looping, so the logs stay readable.

**Searches return 503 while `/health` returns 200.** Working as intended
— the process is alive and the model is not loaded yet. Route on
`/ready`.

**Every result shows the same image repeatedly.** The index predates the
deduplication fix. Rebuild it:
`python -m backend.index_builder --checkpoint <path>`.

**First search takes seconds, later ones are instant.** Cold model load.
The nginx `proxy_read_timeout` is 120s specifically so this does not
surface as a 504.

**Results look confident but wrong.** The index was probably built from a
different checkpoint than the one loaded. Rebuild it.
