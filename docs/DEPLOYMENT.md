# Deployment

How to run VectorMind outside a development shell, and what has actually
been verified versus what has only been written.

**Verification status (2026-08-25): both images built, the full stack
run end to end, and the TLS and auth overlays exercised — all on this
machine.** What was actually observed:

| Check | Result |
|---|---|
| Backend image builds | ✅ with `import backend.app` and an offline tokenizer load as build gates |
| Frontend image builds | ✅ 93.1MB, with `tsc`, `oxlint` and `nginx -t` as build gates |
| Backend reaches healthy | ✅ `/ready` 200 — model, both indices and both index maps loaded |
| Index served correctly | ✅ `num_indexed_images: 31783` — the full corpus, from `--split all` |
| SPA served through nginx | ✅ `GET /` → 200 |
| API proxied through nginx | ✅ `/health`, `/ready`, `/search/text` all 200 same-origin |
| Search returns distinct images | ✅ 10 unique filenames of 10 results, across five queries |
| Image→text search | ✅ 10 distinct captions for an uploaded photo |
| Cold first query | 1.06s — the first forward pass, with the tokenizer now loaded offline |
| Warm latency | 37–102ms through the proxy, p50 51ms (see note below) |
| Security headers | ✅ all four, exactly once, on **every** path — see below |
| Request correlation | ✅ `X-Request-ID` on every API response (the SPA document is served by nginx and never passes through the backend middleware that sets it) |
| Rate limiting | ✅ 429 after the configured 30/minute |
| Basic auth overlay | ✅ 401 on `/` and `/search`, 200 on `/health`, `/ready`, `/nginx-health`; correct credentials → 200 |
| TLS overlay | ✅ Caddy → nginx → backend chain serves HTTPS, redirects HTTP→HTTPS (308), and adds HSTS once |

On warm latency. The 8–19ms recorded on 2026-08-24 was measured on an
otherwise idle machine; the range above was measured with two unrelated
container stacks running alongside it. Both are real; neither is a
property of the application alone, which is why the number is given with
its conditions rather than as a specification.

**The security-header row previously read "✅ all four present, exactly
once" and was wrong.** It had been checked on a proxied API path, which
is one of the few locations that inherited them. `GET /` — the SPA
document itself — carried none of the four. See "What was fixed on
2026-08-25" below.

CI has since been observed green (run 32756952454, all five jobs). A
sixth job has been added that validates everything in `deployment/`:
the Caddyfile, every documented compose combination, and the response
headers of a running container.

Still not verified: there is no public deployment. The TLS chain has now
been run end to end locally against Caddy's internal CA, so what remains
untested is specifically ACME issuance against a real domain — not the
proxy topology.

**Deploy tooling added 2026-08-28.** The "Going public" path is now
scripted for a Windows host: `deployment/preflight.ps1` checks what a
public instance needs, `deployment/deploy.ps1` brings up the TLS stack,
and `deployment/verify.ps1` asserts the checks this table records. What
was actually run that day, on this machine:

| Check | Result |
|---|---|
| `preflight.ps1` | ✅ no blockers — artifacts present (278MB checkpoint, indices, 31,783 images), Docker 29.7.2 + compose v5 up, public IP `106.221.214.121` is a real public address (not CGNAT/private), outbound HTTPS fine |
| Compose validation | ✅ all four combinations (base, `+tls`, `+auth`, `+tls +auth`) pass `config -q` on this machine |
| Full stack rebuilt | ✅ both images built from current source; backend healthy; `/ready` green |
| `verify.ps1 -Local` | ✅ full corpus (`num_indexed_images: 31783`), 5 distinct text results, four security headers, `X-Request-ID` on API responses |

Still open, and none of it is code: a domain (free duckdns or bought),
the router port-forward proof (the `-TestPublicPort` step, which needs a
phone on cellular), and the DNS cutover.

---

## What was fixed on 2026-08-25

Three defects in the never-executed public-deployment path, plus one
piece of unnecessary runtime coupling. Each was found by running the
thing rather than by reading it.

| Defect | Effect | Fix |
|---|---|---|
| `deployment/Caddyfile` did not parse | `transport` was written as a site-level directive; it is a subdirective of `reverse_proxy`. Caddy exits with `unrecognized directive: transport`, so the TLS stack could never have started and the site could never have been issued a certificate. | Nested inside `reverse_proxy`; `caddy validate` now runs in CI |
| The TLS overlay did not rebind the app to loopback | Compose merges sequences by **appending**, so the overlay's `127.0.0.1:8080` mapping was added to the base file's `0.0.0.0:8080` rather than replacing it. The app's nginx stayed bound on every interface behind the TLS terminator, and the two entries also collided at bind time. | `ports: !override`; CI asserts exactly one mapping, on loopback |
| No security headers on any static response | nginx drops the inherited `add_header` set in any location that declares an `add_header` of its own. Every such location — including `location /` — silently lost all four. `/assets/` also returned two `Cache-Control` headers and `/nginx-health` two `Content-Type` headers. | Headers moved to `deployment/security-headers.inc`, included per location; CI asserts on real responses |
| The tokenizer reached the Hub on first query | The cache is baked into the image, but `transformers` still made an outbound request before falling back to it — slow when the network is up, and a needless dependency when it is not. | `HF_HUB_OFFLINE=1`, set after the download step, with an offline load as a build gate. Cold first query 2.07s → 1.06s |

The common thread is the same one this repository has recorded before: a
configuration file that is never executed is a document, not a
configuration. `nginx -t` at image-build time was the only gate over
`deployment/`, and it checks one file's syntax.

---

## Windows host — quick start

The deployment is scripted for a Windows machine (Docker Desktop). Four
commands, run at the repository root in PowerShell:

```powershell
# 1. The gate. Verifies the artifacts, Docker, and that the network can
#    host something public. Read every FAIL; WARN items are your part.
powershell -ExecutionPolicy Bypass -File deployment/preflight.ps1

# Proves the router forwards port 80: starts a listener, then load
# http://<public-ip>/preflight from a PHONE ON CELLULAR (not home WiFi).
powershell -ExecutionPolicy Bypass -File deployment/preflight.ps1 -TestPublicPort

# 2. Tell deploy.ps1 the public hostname (copy once, then fill in).
#    Free name option: duckdns.org — a real DNS record, so the
#    Let's Encrypt HTTP-01 challenge works with it.
Copy-Item deployment\.env.example deployment\.env   # then edit: DOMAIN, TLS_EMAIL

# 3. Deploy: validates compose, opens firewall 80/443, `up -d --build`,
#    waits for /ready, then runs the verification checks.
powershell -ExecutionPolicy Bypass -File deployment/deploy.ps1

# 4. Re-verify after DNS points the name at this machine.
powershell -ExecutionPolicy Bypass -File deployment/verify.ps1

# Logs, when you need them:
docker compose -f deployment/docker-compose.yml -f deployment/docker-compose.tls.yml logs -f
```

`deploy.ps1` fails fast and tells you why: missing `.env`, Docker engine
down, artifacts absent, compose invalid, or the stack not reaching
`/ready` within 180s. Nothing starts half-configured.

For a private demo instead of a public one (e.g., a CV link), add the
auth overlay — see [auth.conf.example](../deployment/auth.conf.example)
and the Basic auth steps in the manual section below.

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
| `backend/indices/` | `python -m backend.index_builder --checkpoint checkpoints/train/best_model.pt --split all` |
| `data/raw/flickr30k/images/` | See [DATASETS.md](DATASETS.md) |

**`--split all` matters.** It indexes the whole 31,783-image corpus,
which is what the demo should serve. Building from `--split test` gives
the index only 3,179 images — a tenth of the corpus — so most queries
have nothing relevant to match against and the model looks far worse
than it is. Reported metrics are unaffected either way:
`scripts/generate_reports.py` measures Recall@K on the test split and
never reads this index.

The index is **not committed** — at 234MB, with `text_index.faiss` alone
above GitHub's 100MB file limit, it cannot be. Budget ~10 minutes on a
GPU to build it.

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
limits, and the serve-time device. One thing is read from the
environment: `VECTORMIND_DEVICE` overrides the config's `server.device`
(that is how the GPU compose overlay pins `cuda`); `PYTHONUNBUFFERED` is
the other env read.

Two values to revisit before exposing this publicly:

- `cors.allow_origins` lists localhost only. Add the real origin, or leave it — with the nginx proxy the app is same-origin and CORS is unused.
- `limits.rate_limit_requests` is 30/minute per client. It is enforced **in-process**, so N workers means N times the budget. Run one worker, or put a real limiter upstream.

---

## Known gaps

Honest list of what this deployment is not.

1. **No TLS in the default stack.** The base compose file serves plain HTTP, and its security headers deliberately omit HSTS, which belongs on whichever hop terminates TLS. `deployment/docker-compose.tls.yml` adds Caddy for that and sets HSTS there — run locally against Caddy's internal CA, not yet against a real domain (see "Going public").
2. **Single machine, single worker.** The rate limiter holds per-process state, so horizontal scaling silently multiplies the effective limit. Kubernetes and managed endpoints are explicitly out of scope — see [FUTURE_IDEAS.md](FUTURE_IDEAS.md).
3. **Metrics endpoint yes, metrics stack no.** A Prometheus-format `/metrics` endpoint exists (request counters and a latency histogram, added 2026-08-29) but nothing scrapes it — there is still no Prometheus/Grafana/alerting deployment. Structured logs with request ids remain the primary observability. The endpoint is the hook an operator's Prometheus would point at.
4. **CPU inference by default.** The shipped image installs CPU torch, and only the packages serving actually imports (`requirements-serving.txt`). Measured through the proxy: 8-19ms warm, 2.07s on the first query while the tokenizer and first forward pass warm up. Fine at this corpus size. GPU serving is now a documented option — `configs/serving.yaml` `server.device` plus `deployment/docker-compose.gpu.yml`, which rebuilds with CUDA torch and pins `VECTORMIND_DEVICE=cuda` — but the GPU image, like ACME issuance, has not yet been exercised against real CUDA hardware.
5. **No authentication.** Every endpoint is public. There is nothing to protect but the GPU, which is what the rate limit is for.

---

## Going public

Everything above runs the stack on a machine you are already sitting at.
This section is the remaining Phase 7 deliverable: the same stack, on a
host with a public name and a certificate.

On a Windows host, the scripted path above ("Windows host — quick
start") is the how. What follows is the manual route, which also covers
the case of doing this on a Linux VM; the scripts encapsulate exactly
these steps.

**Status: the stack has been run, the host has not.** As of 2026-08-27
the TLS overlay in this section has been executed locally end to end —
Caddy in front of the app's nginx, HTTPS served, HTTP redirected, HSTS
set once — using Caddy's internal CA in place of a public domain. Doing
that is what surfaced the two defects in the table at the top of this
file, either of which would have stopped this section's Step 3 dead. On
2026-08-28 the targets of every step below were exercised on the host
(see the deploy-tooling verification table at the top).

What has **not** been run: any of it against a real domain or
Let's Encrypt, and the router port-forward proof (the
`-TestPublicPort` step). ACME issuance in particular has never been
exercised — the internal CA proves the proxy topology, not the
certificate path.

Marked this precisely on purpose: this project has a history of
documents describing intentions in the same voice as results
(docs/KNOWN_ISSUES.md §5), and "the TLS overlay works" would be exactly
that mistake again.

### What the host needs

Sized from what was actually measured locally, not guessed:

| Resource | Needed | Where it goes |
|---|---|---|
| RAM | **4GB minimum**, 8GB comfortable | The backend container is capped at 4G in compose; the model plus both FAISS indices sit around 1.5GB resident |
| Disk | **~12GB** | 2.17GB backend image, 93MB frontend image, 1.3GB Flickr30k images, 234MB indices, 278MB checkpoint, plus Docker overhead |
| vCPU | 2 | Inference is CPU-only here: 8-19ms warm per query |
| Ports | 80, 443 | Caddy. The app's own port stays on loopback |

A €4-6/month VM (Hetzner CX22, DigitalOcean 2GB/2vCPU with an upgraded
disk, Lightsail 4GB) is enough. **Do not size for 2GB RAM** — the
backend limit alone is 4G, and the container will be OOM-killed
mid-request rather than failing at startup where it would be obvious.

Serverless and scale-to-zero platforms fit this badly: 1.3GB of images
and a 234MB index have to live somewhere persistent, and a cold start
that loads a 278MB checkpoint is a poor fit for a request-triggered
container. A plain VM is both cheaper and simpler here.

### Step 1 — provision and prepare the host

For a Windows host these prerequisites replace the commands below:
Windows Firewall (or a rule added by `deploy.ps1` when elevated) must
allow inbound TCP 80/443, the router must forward 80/443 to this
machine, and the machine needs a public IP that is not CGNAT —
`deployment/preflight.ps1 -TestPublicPort` is the honest test of all of
this, because only an outside client (a phone on cellular) can prove it.

For a Linux VM:

```bash
# On the VM, as a sudo-capable user
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker "$USER" && exec su -l "$USER"

sudo ufw allow OpenSSH && sudo ufw allow 80 && sudo ufw allow 443
sudo ufw enable
```

Point the domain's A record (and AAAA, if the host has IPv6) at the VM
**before** starting Caddy. The HTTP-01 challenge is answered on port 80,
so issuance fails while DNS is still propagating, and the failure reads
like a certificate problem rather than a DNS one.

### Step 2 — get the code and the artifacts there

The repository clones; the three mounted artifacts do not, because none
of them are in git (see Prerequisites above).

```bash
git clone https://github.com/<you>/VectorMind.git && cd VectorMind

# From your development machine, in the repo root:
rsync -avz --progress checkpoints/train/best_model.pt     <user>@<host>:VectorMind/checkpoints/train/
rsync -avz --progress backend/indices/     <user>@<host>:VectorMind/backend/indices/
rsync -avz --progress data/raw/flickr30k/images/     <user>@<host>:VectorMind/data/raw/flickr30k/images/
```

The images are the slow part — 1.3GB, and `rsync` resumes, which
`scp` does not.

**Alternative: build the index on the host.** If the upload is painful,
copy only the checkpoint and the images, then run
`python -m backend.index_builder --checkpoint checkpoints/train/best_model.pt --split all`
there. On CPU that takes considerably longer than the ~10 minutes it
takes on a GPU, so uploading is usually the better trade.

### Step 3 — start it, with TLS

```bash
DOMAIN=demo.example.com TLS_EMAIL=you@example.com   docker compose -f deployment/docker-compose.yml                  -f deployment/docker-compose.tls.yml up -d --build
```

`deployment/docker-compose.tls.yml` rebinds the app's own port to
loopback and puts Caddy on 80/443 with automatic certificate issuance
and renewal. HSTS is set there, on the hop that actually terminates TLS.

For a private demo, add HTTP Basic auth as well — worth doing if the
link is going into a CV rather than onto the open internet:

```bash
docker run --rm httpd:alpine htpasswd -nbB <user> '<passphrase>'   > deployment/.htpasswd
cp deployment/auth.conf.example deployment/auth.conf

DOMAIN=demo.example.com TLS_EMAIL=you@example.com   docker compose -f deployment/docker-compose.yml                  -f deployment/docker-compose.tls.yml                  -f deployment/docker-compose.auth.yml up -d
```

Both credential files are gitignored. Basic auth is only meaningful
behind TLS, which is why the overlays are used together.

### Step 4 — verify, and record what you saw

The claim in ROADMAP.md is "a reader can reach a live deployed instance",
so the check is a real query from a browser on another network, not a
`curl` from the host:

```bash
curl -sS https://demo.example.com/ready | python -m json.tool
curl -sS -X POST https://demo.example.com/search/text   -H 'Content-Type: application/json'   -d '{"query": "a dog running through grass", "top_k": 5}'   | python -m json.tool
```

| Check | Expected |
|---|---|
| `/ready` | `"ready": true`, model and both indices loaded |
| `num_indexed_images` | 31783 — the full corpus, not 3179 |
| Text search | 5 results, distinct filenames, plausible for the query |
| Image search | Upload from the browser returns ranked results |
| Certificate | Valid, issued by Let's Encrypt, auto-renewing |
| First query | Seconds (cold start); subsequent queries fast |

Then update `ROADMAP.md` Phase 7, `docs/PROJECT_STATUS.md` and the
verification table at the top of this file with the URL and the date —
and only then. The point of the status tables in this repository is that
they record what was observed.

### Keeping it running

- `restart: unless-stopped` is set on every service, so the stack comes back after a reboot. Verify it once with `sudo reboot` rather than assuming.
- Watch disk. Docker image layers accumulate on rebuilds: `docker system prune -f` after a deploy.
- The rate limiter is in-process at 30 requests/minute (`configs/serving.yaml`). Keep one backend worker, or the limit multiplies by worker count.
- There is no metrics stack, and that is deliberate. `docker compose logs -f backend`, the request ids in each response, and the `/metrics` endpoint (scraped by hand, or by an external Prometheus if one is ever added) are the observability, which is proportionate for a demo. Run the concurrent load test (`scripts/load_test_api.py`) before load to see the latency curve under contention rather than the single-serial benchmark.

### Cost, honestly

A €4-6/month VM running continuously, plus a domain. If that is not
worth it, the defensible alternative is to run the demo on request:
record a screen capture (`docs/screenshots/`), keep the stack
reproducible with one compose command, and say plainly in the README
that the demo runs locally. What is not defensible is leaving a dead
link in a portfolio.

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
