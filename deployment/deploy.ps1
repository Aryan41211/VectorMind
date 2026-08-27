# VectorMind - one-command public deployment (Windows host)
#
# Brings up the public stack: backend + frontend + the Caddy TLS overlay,
# after the preflight gate (deployment/preflight.ps1) has confirmed the
# machine can do it. Reads the deployment configuration from
# deployment/.env (copy deployment/.env.example and fill in DOMAIN and
# TLS_EMAIL first).
#
# What it does, in order:
#   1.  parses deployment/.env and exports DOMAIN/TLS_EMAIL for compose;
#   2.  fails fast if Docker/compose are missing or the serving artifacts
#       are not in place (the same floors preflight.ps1 uses);
#   3.  validates the compose combination before starting anything;
#   4.  creates Windows Firewall inbound allow rules for TCP 80/443;
#   5.  docker compose up -d --build (or -SkipBuild to reuse images);
#   6.  waits for the frontend's /ready through nginx on loopback;
#   7.  hands off to deployment/verify.ps1 for the public checks.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File deployment/deploy.ps1
#   powershell -ExecutionPolicy Bypass -File deployment/deploy.ps1 -SkipBuild
#
# Run this from the repo root. The TLS overlay binds host ports 80/443,
# so administrative rights are needed for the firewall step; if the
# current session is not elevated the script prints the exact elevated
# command to run once instead of failing.
#
# Exit codes: 0 = deployed and healthy; 1 = an error occurred.

param(
    # Reuse existing vectormind-backend/frontend images instead of
    # rebuilding from source.
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$envFile = Join-Path $PSScriptRoot ".env"

# --- 0. Configuration ------------------------------------------------------
if (-not (Test-Path -LiteralPath $envFile)) {
    Write-Host "deployment/.env is missing." -ForegroundColor Red
    Write-Host "  Copy deployment/.env.example to deployment/.env and set DOMAIN" -ForegroundColor Yellow
    Write-Host "  and TLS_EMAIL first (see deployment/preflight.ps1 -Domain for the DNS check)."
    exit 1
}
foreach ($line in Get-Content -LiteralPath $envFile) {
    if ($line -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$") {
        $key = $matches[1]
        $value = $matches[2].Trim().Trim('"').Trim("'")
        Set-Item -Path "Env:$key" -Value $value
    }
}
$domain = $env:DOMAIN
$tlsEmail = $env:TLS_EMAIL
$frontendPort = $env:FRONTEND_PORT
if (-not $frontendPort) { $frontendPort = "8080" }

if (-not $domain) {
    Write-Host "DOMAIN is not set in deployment/.env. TLS needs a name resolving to" -ForegroundColor Red
    Write-Host "this machine's public IP (see deployment/preflight.ps1 -Domain)."
    exit 1
}
if (-not $tlsEmail) {
    Write-Host "TLS_EMAIL is not set in deployment/.env. Let's Encrypt needs an address" -ForegroundColor Red
    Write-Host "for certificate-expiry notices."
    exit 1
}
Write-Host "Deploying $domain (let's encrypt notices to $tlsEmail)" -ForegroundColor Cyan

# --- 1. Docker and artifacts ------------------------------------------------
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "docker is not installed. Install Docker Desktop first." -ForegroundColor Red
    exit 1
}
$serverVersion = docker version --format "{{.Server.Version}}" 2>$null
if (-not $serverVersion) {
    Write-Host "The Docker engine is not running. Start Docker Desktop and retry." -ForegroundColor Red
    exit 1
}

function Test-Artifact {
    # Fail fast on a missing serving artifact, mirroring preflight.ps1's
    # floors so `up` does not start a half-working stack.
    param([string]$Path, [long]$MinBytes, [string]$Label)

    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "Missing ${Label}: $Path" -ForegroundColor Red
        Write-Host "  See 'Prerequisites' in docs/DEPLOYMENT.md for how to obtain it."
        exit 1
    }
    if ($MinBytes -gt 0) {
        $len = (Get-Item -LiteralPath $Path).Length
        if ($len -lt $MinBytes) {
            Write-Host "$Label looks truncated ($len bytes) at $Path" -ForegroundColor Red
            exit 1
        }
    }
}

Test-Artifact -Path (Join-Path $repoRoot "checkpoints\train\best_model.pt") `
    -MinBytes 200MB -Label "checkpoint"
Test-Artifact -Path (Join-Path $repoRoot "backend\indices\text_index.faiss") `
    -MinBytes 100MB -Label "text index"
Test-Artifact -Path (Join-Path $repoRoot "backend\indices\image_index.faiss") `
    -MinBytes 20MB -Label "image index"
Test-Artifact -Path (Join-Path $repoRoot "data\raw\flickr30k\images") `
    -MinBytes 0 -Label "Flickr30k images"

# --- 2. Validate the compose combination before starting anything -----------
Write-Host "Validating compose (base + tls overlay)... "
$composeArgs = @(
    "compose", "-f", "deployment/docker-compose.yml",
    "-f", "deployment/docker-compose.tls.yml", "config", "-q"
)
& docker @composeArgs 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "docker compose config failed. Run the deployment CI job's `"Compose files`"" -ForegroundColor Red
    Write-Host "step, or check the overlays for the error." -ForegroundColor Red
    exit 1
}
Write-Host "OK" -ForegroundColor Green

# --- 3. Windows Firewall ----------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
foreach ($port in 80, 443) {
    $ruleName = "VectorMind public demo TCP $port"
    $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
    if ($existing) { continue }
    if ($isAdmin) {
        netsh advfirewall firewall add rule name="VectorMind public demo TCP $port" dir=in action=allow protocol=TCP localport=$port | Out-Null
        Write-Host "Firewall: allowed inbound TCP $port" -ForegroundColor Green
    }
    else {
        Write-Host "Firewall: not elevated, so not adding TCP $port" -ForegroundColor Yellow
        Write-Host "  Run once as Administrator to allow inbound 80/443:" -ForegroundColor Yellow
        Write-Host "    netsh advfirewall firewall add rule name=`"VectorMind public demo TCP 80`" dir=in action=allow protocol=TCP localport=80"
        Write-Host "    netsh advfirewall firewall add rule name=`"VectorMind public demo TCP 443`" dir=in action=allow protocol=TCP localport=443"
    }
}

# --- 4. Bring it up ----------------------------------------------------------
$upArgs = @(
    "compose", "-f", "deployment/docker-compose.yml",
    "-f", "deployment/docker-compose.tls.yml", "up", "-d"
)
if (-not $SkipBuild) { $upArgs += "--build" }
Write-Host "Starting the stack (this builds two images the first time)..."
& docker @upArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "docker compose up failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

# --- 5. Wait for readiness ---------------------------------------------------
# The TLS overlay publishes the app's nginx on loopback
# (127.0.0.1:$frontendPort), so poll that: it proxies /ready to the
# backend, which 503s until the checkpoint and indices are loaded.
Write-Host ("Waiting for the stack to become ready at http://127.0.0.1:{0}/ready ..." -f $frontendPort)
$readyUrl = "http://127.0.0.1:$frontendPort/ready"
$deadline = (Get-Date).AddSeconds(180)
$ready = $false
while ((Get-Date) -lt $deadline -and -not $ready) {
    try {
        $r = Invoke-RestMethod -Uri $readyUrl -TimeoutSec 5
        $ready = $r.ready -eq $true
    }
    catch {
        Start-Sleep -Seconds 5
    }
}
if (-not $ready) {
    Write-Host "The stack did not become ready within 180s." -ForegroundColor Red
    Write-Host "  docker compose -f deployment/docker-compose.yml -f deployment/docker-compose.tls.yml logs --tail=50 backend"
    exit 1
}
Write-Host ("Ready on loopback. Public site: https://{0}" -f $domain) -ForegroundColor Green

# --- 6. Verify ----------------------------------------------------------------
Write-Host "Now running the public verification checks..."
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "verify.ps1")
exit $LASTEXITCODE