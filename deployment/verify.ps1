# VectorMind - post-deployment verification (Windows host)
#
# The evidence step for ROADMAP Phase 7: asserts, against the running
# public site, that a reader can actually reach the deployed instance,
# that it serves the full corpus, and that search returns real results.
#
# Defaults to the public origin (https://$DOMAIN from deployment/.env).
# -Local checks the same stack over plain HTTP on the loopback port
# instead, which is how you verify before the DNS cutover.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File deployment/verify.ps1
#   powershell -ExecutionPolicy Bypass -File deployment/verify.ps1 -Local
#
# Exit codes: 0 = all checks passed; 1 = something failed.

param(
    # Verify against http://127.0.0.1:FRONTEND_PORT instead of the public
    # https://$DOMAIN (useful before DNS points at this machine).
    [switch]$Local
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

function Read-DotEnv {
    # Minimal .env reader: exports KEY=VALUE lines into the process env.
    param([string]$Path)

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match "^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$") {
            Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2].Trim().Trim('"').Trim("'")
        }
    }
}

function Write-Check {
    # One verdict line, book-keeping the exit code.
    param([string]$Name, [string]$Result, [string]$Detail = "")

    $color = "Gray"
    if ($Result -eq "PASS") { $color = "Green" }
    elseif ($Result -eq "FAIL") { $color = "Red"; $script:failed = $true }
    else { $color = "Yellow" }
    Write-Host ("[{0}] {1}" -f $Result.PadRight(4), $Name) -ForegroundColor $color
    if ($Detail) { Write-Host ("       {0}" -f $Detail) -ForegroundColor "DarkGray" }
}

$script:failed = $false

$envFile = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path -LiteralPath $envFile)) {
    Write-Host "deployment/.env is missing; copy deployment/.env.example and set DOMAIN." -ForegroundColor Red
    exit 1
}
Read-DotEnv -Path $envFile
$domain = if ($Local) { "" } else { $env:DOMAIN }
if (-not $domain -and -not $Local) {
    Write-Host "DOMAIN is not set in deployment/.env." -ForegroundColor Red
    exit 1
}
$port = $env:FRONTEND_PORT
if (-not $port) { $port = "8080" }

$origin = if ($Local) { "http://127.0.0.1:$port" } else { "https://$domain" }
Write-Host ("Verifying {0}" -f $origin) -ForegroundColor Cyan
Write-Host ""

# --- 1. TLS and readiness ----------------------------------------------------
if ($Local) {
    Write-Check "Readiness (/ready)" "SKIP" "loopback run; /ready is exercised by deploy.ps1's wait"
}
else {
    $name = "Certificate and readiness (https://$domain/ready)"
    try {
        $readyJson = Invoke-RestMethod -Uri "$origin/ready" -TimeoutSec 30
        if ($readyJson.ready -eq $true) {
            Write-Check $name "PASS" "valid certificate chain; model and both indices loaded"
        }
        else {
            Write-Check $name "FAIL" ("ready:false - {0}" -f ($readyJson | ConvertTo-Json -Compress))
        }
    }
    catch {
        Write-Check $name "FAIL" $_.Exception.Message
    }
}

# --- 2. Full corpus -----------------------------------------------------------
$name = "Full corpus indexed (/health num_indexed_images)"
try {
    $healthJson = Invoke-RestMethod -Uri "$origin/health" -TimeoutSec 30
    if ($healthJson.num_indexed_images -eq 31783) {
        Write-Check $name "PASS" "31783 images"
    }
    else {
        Write-Check $name "FAIL" ("{0} images; expected 31783 (build the index with `--split all`)" -f $healthJson.num_indexed_images)
    }
}
catch {
    Write-Check $name "FAIL" $_.Exception.Message
}

# --- 3. Real search ------------------------------------------------------------
$name = "Text search returns distinct results"
try {
    $body = @{ query = "a dog running through a grassy field"; top_k = 5 } | ConvertTo-Json
    $searchJson = Invoke-RestMethod -Uri "$origin/search/text" -Method Post -Body $body `
        -ContentType "application/json" -TimeoutSec 60
    $names = @($searchJson.results | ForEach-Object { $_.filename })
    $distinct = @($names | Select-Object -Unique)
    if (@($searchJson.results).Count -eq 5 -and $distinct.Count -eq 5) {
        Write-Check $name "PASS" ("5 distinct results: {0}" -f (($names -join ", ").Substring(0, [Math]::Min(80, ($names -join ", ").Length))))
    }
    else {
        Write-Check $name "FAIL" ("{0} results, {1} distinct (want 5 and 5)" -f @($searchJson.results).Count, $distinct.Count)
    }
}
catch {
    Write-Check $name "FAIL" $_.Exception.Message
}

# --- 4. HTTP -> HTTPS redirect -------------------------------------------------
if (-not $Local) {
    $name = "HTTP on port 80 redirects to HTTPS (308)"
    try {
        $resp = Invoke-WebRequest -Uri "http://$domain/" -MaximumRedirection 0 -TimeoutSec 30 `
            -UseBasicParsing -ErrorAction Stop
        Write-Check $name "FAIL" ("expected a redirect, got {0}" -f $resp.StatusCode)
    }
    catch {
        $loc = ""
        if ($_.Exception.Response -and $_.Exception.Response.Headers) {
            $loc = $_.Exception.Response.Headers["Location"]
        }
        if ($loc -and $loc -match "https://$domain") {
            Write-Check $name "PASS" ("Location: {0}" -f $loc)
        }
        elseif ($_.Exception.Response -and $_.Exception.Response.StatusCode -eq 308) {
            Write-Check $name "FAIL" "308 but no Location header pointing at the HTTPS origin"
        }
        else {
            Write-Check $name "FAIL" $_.Exception.Message
        }
    }
}
else {
    Write-Check "HTTP -> HTTPS redirect" "SKIP" "loopback run has no Caddy TLS hop"
}

# --- 5. Security headers and HSTS ----------------------------------------------
$name = "Security headers on the SPA document"
try {
    $resp = Invoke-WebRequest -Uri "$origin/" -TimeoutSec 30 -UseBasicParsing
    $need = "X-Content-Type-Options", "X-Frame-Options", "Referrer-Policy", "Permissions-Policy"
    $missing = @($need | Where-Object { -not $resp.Headers[$_] })
    if ($missing.Count -eq 0) {
        Write-Check $name "PASS" ("all of {0} present" -f ($need -join ", "))
    }
    else {
        Write-Check $name "FAIL" ("missing: {0}" -f ($missing -join ", "))
    }
}
catch {
    Write-Check $name "FAIL" $_.Exception.Message
}

$name = "Request correlation (X-Request-ID) on API responses"
try {
    $resp = Invoke-WebRequest -Uri "$origin/health" -TimeoutSec 30 -UseBasicParsing
    if ($resp.Headers["X-Request-ID"]) {
        Write-Check $name "PASS" "present (id: $($resp.Headers['X-Request-ID']))"
    }
    else {
        Write-Check $name "FAIL" "no X-Request-ID header on /health"
    }
}
catch {
    Write-Check $name "FAIL" $_.Exception.Message
}

if ($Local) {
    Write-Check "HSTS (Strict-Transport-Security)" "SKIP" "HSTS is set on the Caddy TLS hop, not on plain HTTP"
}
else {
    $name = "HSTS via Caddy"
    try {
        $resp = Invoke-WebRequest -Uri "https://$domain/" -TimeoutSec 30 -UseBasicParsing
        if ($resp.Headers["Strict-Transport-Security"]) {
            Write-Check $name "PASS" $resp.Headers["Strict-Transport-Security"]
        }
        else {
            Write-Check $name "FAIL" "no Strict-Transport-Security header from the TLS hop"
        }
    }
    catch {
        Write-Check $name "FAIL" $_.Exception.Message
    }
}

# --- Verdict --------------------------------------------------------------------
Write-Host ""
if ($script:failed) {
    Write-Host "VERDICT: some checks failed - not ready to record as deployed." -ForegroundColor Red
    exit 1
}
Write-Host "VERDICT: all checks passed. The demo is reachable at the public origin." -ForegroundColor Green
exit 0