# Pre-Launch / Go-Live Checklist

A checklist for taking a VectorMind instance public. It consolidates the
machine-checkable gates (`deployment/preflight.ps1`, `deploy.ps1`,
`verify.ps1`) with the human checks that no script can do, so a launch is
either reproducible or not done. Work top to bottom; a FAIL stops the
line.

The honest status is recorded in `docs/DEPLOYMENT.md` and
`docs/PROJECT_STATUS.md`, and this checklist's checks are the same ones
those documents' verification tables assert.

---

## 1. Preconditions (run `deployment/preflight.ps1`)

The gate. It checks artifacts, Docker, and that this host can actually
host something public.

- [ ] Checkpoint present and ≥200MB (`checkpoints/train/best_model.pt`)
- [ ] FAISS indices present and non-trivial (`image_index.faiss`,
      `text_index.faiss`)
- [ ] Full corpus cached — 31,783 Flickr30k images (a partial cache
      indexes a partial corpus and reports it as success)
- [ ] Docker CLI + engine reachable, Compose v2+
- [ ] Host has a real public IP (not CGNAT/private) and outbound HTTPS
- [ ] Firewall allows inbound 80/443
- [ ] **Manually:** prove the router forwards port 80 with
      `preflight.ps1 -TestPublicPort`, from a phone on cellular

Read every FAIL; WARN items are your responsibility to resolve.

## 2. Configure the deployment

- [ ] `Copy-Item deployment\.env.example deployment\.env`, then:
      `DOMAIN` = the public hostname that resolves to this machine;
      `TLS_EMAIL` = a real address for Let's Encrypt notice
- [ ] Choose the reachability: public (no overlay), or private demo
      (+ `docker-compose.auth.yml` behind Basic auth)
- [ ] Optional GPU serving: build the backend with the GPU overlay
      (`docker-compose.gpu.yml`) instead of the CPU image — see
      `docs/DEPLOYMENT.md`; only worth it if the host has a GPU
- [ ] Confirm `configs/serving.yaml` CORS origins include the deployed
      frontend origin (currently localhost only)

## 3. Deploy (`deployment/deploy.ps1`)

- [ ] Compose validates (`config -q`) for the chosen overlay combination
- [ ] Firewall rules applied for 80/443
- [ ] Stack `up -d --build`; backend passes its `/ready` healthcheck
- [ ] No `CrashLoopBackOff`; backend logs show model + both indices loaded

## 4. Verify (`deployment/verify.ps1`, then from another network)

The script hits the public `https://DOMAIN`; it is not a loopback-only
approval.

- [ ] TLS certificate valid, issued by Let's Encrypt, auto-renew volume mount
- [ ] `http://DOMAIN` redirects to `https://DOMAIN` (308)
- [ ] `/ready` → `"ready": true`
- [ ] `num_indexed_images` == 31,783 (full corpus)
- [ ] Text search returns 5+ distinct filenames across repeated queries
- [ ] All four security headers on API paths; `X-Request-ID` on API
      responses
- Optional, from a phone/other network: the load test
      `scripts/load_test_api.py --base-url https://DOMAIN` shows a sane
      error rate and p95 — the single serial benchmark cannot see
      contention

## 5. Observability and smoke before calling it live

- [ ] `scripts/smoke_test_api.py --base-url <url>` passes end to end
- [ ] `/metrics` on the backend returns a Prometheus-format body with a
      latency histogram and request counters (added 2026-08-29) — scrape
      it once by hand before wiring any monitoring
- [ ] `docker compose logs -f backend` shows clean access logs with
      request ids (the project runs **no** metric stack by design; this
      endpoint is the hook an external Prometheus would point at)

## 6. Record what you saw — and only then

- [ ] Update the verification table at the top of `docs/DEPLOYMENT.md`
      with the URL and date
- [ ] Update `ROADMAP.md` Phase 7: the "live deployed demo at a public
      URL" item flips from *Not started* **only after** the checks above
      pass against the real public URL
- [ ] Update `docs/PROJECT_STATUS.md`
- [ ] Confirm no dead link: load the public URL from another network

---

## Known gaps that constrain a launch

- **Single worker / single node.** The rate limiter is per-process
  (30/min in `configs/serving.yaml`); a multi-worker backend multiplies
  the budget. Kubernetes is out of scope (`docs/DEPLOYMENT.md`,
  `ROADMAP.md`).
- **No metrics stack.** Only the `/metrics` endpoint exists, not a
  Prometheus/Grafana deployment — deliberate for a demo.
- **GPU is optional.** The default image is CPU; `docker-compose.gpu.yml`
  is written but, like the ACME issuance, not exercised against a real
  GPU host.
