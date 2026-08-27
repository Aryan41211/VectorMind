# VectorMind - Deployment Preflight (Windows host)
#
# The precondition gate for turning this machine into a public demo host.
# It does not deploy anything. It tells you, with evidence, whether the
# three ingredients a public instance needs are actually in place:
#
#   1. the serving artifacts exist (checkpoint, indices, images);
#   2. Docker can run the stack;
#   3. a browser on the open internet can reach this machine (public IP,
#      not CGNAT, inbound port 80/443 routeable).
#
# A home connection usually fails 3. The point of this file is to find
# that out before you have paid for a domain name.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File deployment/preflight.ps1
#   powershell -ExecutionPolicy Bypass -File deployment/preflight.ps1 `
#       -TestPublicPort -Domain demo.duckdns.org
#
# -TestPublicPort starts a temporary listener on port 80 and waits for a
# connection, so you can prove inbound routing by browsing to
# http://<public-ip>/preflight from a phone on cellular (not home WiFi).
# Only that test can answer whether the router forwards correctly.
#
# Exit codes: 0 = no blockers found, 1 = a blocker found, 2 = could not
# determine (some checks require the interactive port test or a domain).

param(
    # Start the temporary port-80 listener and wait for an outside client.
    [switch]$TestPublicPort,
    # How long -TestPublicPort listens before giving up.
    [int]$PortTestSeconds = 90,
    # A name your DNS (or a DDNS updater) is working toward; checked
    # against the detected public IP when supplied.
    [string]$Domain = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version 3.0

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# --- The artifacts and their sanity floors, from what was actually
# --- measured on 2026-08-28 (checkpoint ~278MB, indices ~234MB total).
$checkpointPath = "checkpoints/train/best_model.pt"
$checkpointMinBytes = 200MB
$indexMinBytes = 20MB
$textIndexMinBytes = 100MB
$minImageCount = 30000

$privateResult = 0   # count of FAIL
$unclearResult = 0   # count of SKIP/WARN that block the "ready" verdict

function Write-Check {
    # Emit one verdict line and book-keep the exit code.
    param([string]$Name, [string]$Result, [string]$Detail = "")

    $color = "Gray"
    switch ($Result) {
        "PASS"  { $color = "Green" }
        "FAIL"  { $color = "Red"; $script:privateResult++ }
        "WARN"  { $color = "Yellow"; $script:unclearResult++ }
        default { $color = "Gray"; $script:unclearResult++ }
    }
    Write-Host ("[{0}] {1}" -f $Result.PadRight(4), $Name) -ForegroundColor $color
    if ($Detail) { Write-Host ("       {0}" -f $Detail) -ForegroundColor "DarkGray" }
}

function Test-PrivateAddress {
    # True if the dotted-quad string is in a range no router on the
    # internet can reach: private, CGNAT, loopback, or link-local.
    param([string]$Ip)

    $octets = $Ip -split "\."
    if ($octets.Count -ne 4) { return $true }
    foreach ($o in $octets) {
        $n = 0
        if (-not [int]::TryParse($o, [ref]$n)) { return $true }
        if ($n -lt 0 -or $n -gt 255) { return $true }
    }
    $a = [int]$octets[0]; $b = [int]$octets[1]
    # 10.0.0.0/8
    if ($a -eq 10) { return $true }
    # 172.16.0.0/12
    if ($a -eq 172 -and $b -ge 16 -and $b -le 31) { return $true }
    # 192.168.0.0/16
    if ($a -eq 192 -and $b -eq 168) { return $true }
    # 100.64.0.0/10 (CGNAT)
    if ($a -eq 100 -and $b -ge 64 -and $b -le 127) { return $true }
    # Loopback and link-local
    if ($a -eq 127 -or ($a -eq 169 -and $b -eq 254)) { return $true }
    return $false
}

function Get-PublicIpv4 {
    # The public IP as seen from outside, or "" if the services are
    # unreachable. Tries three providers so one quota/outage is not fatal.
    foreach ($url in @("https://api.ipify.org", "https://ipv4.icanhazip.com", "https://ifconfig.me/ip")) {
        try {
            $ip = (Invoke-RestMethod -Uri $url -TimeoutSec 15).ToString().Trim()
            if ($ip) { return $ip }
        }
        catch {
            continue
        }
    }
    return ""
}

Write-Host "VectorMind deployment preflight (repo root: $repoRoot)" -ForegroundColor Cyan
Write-Host ""

# --- 1. Serving artifacts ------------------------------------------------
$name = "Checkpoint present ($checkpointPath)"
$ckpt = Join-Path $repoRoot ($checkpointPath -replace "/", "\")
if (Test-Path -LiteralPath $ckpt) {
    $len = (Get-Item -LiteralPath $ckpt).Length
    if ($len -ge $checkpointMinBytes) {
        Write-Check $name "PASS" ("{0:N1} MB" -f ($len / 1MB))
    }
    else {
        Write-Check $name "FAIL" ("only {0:N1} MB; a real checkpoint is ~278MB" -f ($len / 1MB))
    }
}
else {
    Write-Check $name "FAIL" "not found. Build it first:  python scripts/train.py"
}

$name = "FAISS indices present (backend/indices)"
$idxDir = Join-Path $repoRoot "backend\indices"
if (Test-Path -LiteralPath $idxDir) {
    $meta = Join-Path $idxDir "index_metadata.json"
    $img  = Join-Path $idxDir "image_index.faiss"
    $txt  = Join-Path $idxDir "text_index.faiss"
    $missing = @(@($meta, $img, $txt) | Where-Object { -not (Test-Path -LiteralPath $_) })
    if ($missing.Count -eq 0) {
        $imgLen = (Get-Item -LiteralPath $img).Length
        $txtLen = (Get-Item -LiteralPath $txt).Length
        if ($imgLen -ge $indexMinBytes -and $txtLen -ge $textIndexMinBytes) {
            Write-Check $name "PASS" ("image {0:N1}MB, text {1:N1}MB, metadata present" -f ($imgLen / 1MB), ($txtLen / 1MB))
        }
        else {
            Write-Check $name "FAIL" ("index files look truncated (image {0:N1}MB, text {1:N1}MB)" -f ($imgLen / 1MB), ($txtLen / 1MB))
        }
    }
    else {
        Write-Check $name "FAIL" ("missing: {0}" -f (($missing | ForEach-Object { Split-Path -Leaf $_ }) -join ", "))
    }
}
else {
    Write-Check $name "FAIL" "backend/indices not found. Build it first:  python -m backend.index_builder --checkpoint <path> --split all"
}

$name = "Flickr30k images mounted (data/raw/flickr30k/images)"
$imgDir = Join-Path $repoRoot "data\raw\flickr30k\images"
if (Test-Path -LiteralPath $imgDir) {
    $count = (Get-ChildItem -LiteralPath $imgDir -File -ErrorAction SilentlyContinue).Count
    if ($count -ge $minImageCount) {
        Write-Check $name "PASS" ("{0} images" -f $count)
    }
    else {
        Write-Check $name "FAIL" ("only {0} images; the demo needs the full ~31,783" -f $count)
    }
}
else {
    Write-Check $name "FAIL" "image directory not found (see docs/DATASETS.md)"
}

# --- 2. Docker -----------------------------------------------------------
$name = "docker CLI present"
$docker = Get-Command docker -ErrorAction SilentlyContinue
if ($docker) {
    Write-Check $name "PASS" $docker.Source
}
else {
    Write-Check $name "FAIL" "install Docker Desktop (https://www.docker.com/products/docker-desktop/)"
}

$name = "docker daemon reachable (Docker Desktop running)"
$serverVersion = ""
if ($docker) {
    try {
        $serverVersion = docker version --format "{{.Server.Version}}" 2>$null
    }
    catch {
        $serverVersion = ""
    }
    if ($serverVersion) {
        Write-Check $name "PASS" ("server {0}" -f $serverVersion)
    }
    else {
        Write-Check $name "FAIL" "start Docker Desktop and wait for the engine to come up"
    }
}
else {
    Write-Check $name "SKIP" "no docker CLI; nothing to check"
}

$name = "compose v2 available"
if ($docker) {
    $composeVersion = docker compose version 2>$null
    if ($composeVersion -and $composeVersion -match "version") {
        Write-Check $name "PASS" ("{0} (the docker compose subcommand answers)" -f $composeVersion.Trim())
    }
    else {
        Write-Check $name "FAIL" "docker compose v2 is required (comes with current Docker Desktop)"
    }
}
else {
    Write-Check $name "SKIP" "no docker CLI; nothing to check"
}

# --- 3. Reachability from the open internet -------------------------------
Write-Host ""
$name = "Public IP visible to the internet"
$publicIp = Get-PublicIpv4
if ($publicIp) {
    if (Test-PrivateAddress $publicIp) {
        Write-Check $name "FAIL" ("detected IP {0} is a private/CGNAT range - no inbound route exists" -f $publicIp)
        Write-Host "       Check with your ISP whether they use CGNAT; this is the most common" -ForegroundColor "DarkGray"
        Write-Host "       reason a home connection cannot host anything public." -ForegroundColor "DarkGray"
    }
    else {
        Write-Check $name "PASS" ("{0} (a browser outside your network can find this machine by that IP)" -f $publicIp)
    }
}
else {
    Write-Check $name "FAIL" "could not reach any IP-discovery service (api.ipify.org etc.)"
}

$name = "Outbound HTTPS (reach Caddy/Let's Encrypt/DDNS later)"
try {
    $probe = Invoke-WebRequest -Uri "https://duckdns.org" -Method Head -TimeoutSec 15 `
        -UseBasicParsing -ErrorAction SilentlyContinue
    Write-Check $name "PASS" ("duckdns.org answered {0}" -f $probe.StatusCode)
}
catch {
    Write-Check $name "FAIL" ("duckdns.org unreachable: {0}" -f $_.Exception.Message)
}

$name = "Windows Firewall allows inbound TCP 80/443"
if (Get-Command Get-NetFirewallPortFilter -ErrorAction SilentlyContinue) {
    try {
        $fwAllow80 = Get-NetFirewallPortFilter -Protocol TCP -ErrorAction SilentlyContinue |
            Where-Object { $_.LocalPort -contains 80 -or $_.LocalPort -eq 80 } |
            ForEach-Object { Get-NetFirewallRule -AssociatedNetFirewallPortFilter $_ -ErrorAction SilentlyContinue } |
            Where-Object { $_.Enabled -eq "True" -and $_.Action -eq "Allow" -and $_.Direction -eq "Inbound" }
        $has80 = @($fwAllow80).Count -gt 0
        $fwAllow443 = Get-NetFirewallPortFilter -Protocol TCP -ErrorAction SilentlyContinue |
            Where-Object { $_.LocalPort -eq 443 } |
            ForEach-Object { Get-NetFirewallRule -AssociatedNetFirewallPortFilter $_ -ErrorAction SilentlyContinue } |
            Where-Object { $_.Enabled -eq "True" -and $_.Action -eq "Allow" -and $_.Direction -eq "Inbound" }
        $has443 = @($fwAllow443).Count -gt 0
        if ($has80 -and $has443) {
            Write-Check $name "PASS" "TCP 80 and 443 both have inbound allow rules"
        }
        elseif ($has80 -or $has443) {
            $present = if ($has80) { "80" } else { "443" }
            $absent = if ($has80) { "443" } else { "80" }
            Write-Check $name "WARN" ("TCP {0} has an inbound allow rule; TCP {1} does not" -f $present, $absent)
        }
        else {
            Write-Check $name "WARN" "no inbound allow rules for TCP 80/443; the deploy step adds them (elevated)"
        }
    }
    catch {
        Write-Check $name "SKIP" ("could not inspect firewall: {0}" -f $_.Exception.Message)
    }
}
else {
    Write-Check $name "SKIP" "Get-NetFirewallPortFilter unavailable (pre-Windows 8?)"
}

# --- 4. Domain / DNS ------------------------------------------------------
Write-Host ""
if ($Domain) {
    $name = "Domain resolves toward the public IP"
    try {
        $resolved = Resolve-DnsName -Name $Domain -Type A -ErrorAction Stop |
            Where-Object { $_.IPAddress } | Select-Object -ExpandProperty IPAddress
        if ($resolved) {
            if ($resolved -contains $publicIp) {
                Write-Check $name "PASS" ("{0} -> {1}" -f $Domain, ($resolved -join ", "))
            }
            else {
                Write-Check $name "FAIL" ("{0} -> {1}; the machine is at {2}. Fix DNS (or the DDNS updater) before starting Caddy" -f $Domain, ($resolved -join ", "), $publicIp)
            }
        }
        else {
            Write-Check $name "FAIL" "no A record for $Domain"
        }
    }
    catch {
        Write-Check $name "WARN" ("could not resolve {0}: {1}" -f $Domain, $_.Exception.Message)
    }
}
else {
    $name = "Domain supplied for DNS check"
    Write-Check $name "WARN" "no -Domain supplied so DNS was not checked. You need a name resolving to the public IP (buy a domain, or use free duckdns.org DDNS) before TLS works."
}

# --- 5. Interactive inbound-route proof ------------------------------------
Write-Host ""
if ($TestPublicPort) {
    Write-Host "Listening on 0.0.0.0:80 for $PortTestSeconds seconds..." -ForegroundColor Cyan
    Write-Host "  On a phone with Wi-Fi OFF (cellular), open:" -ForegroundColor Cyan
    Write-Host "  http://$publicIp/preflight" -ForegroundColor Cyan

    $listener = $null
    try {
        $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Any, 80)
        $listener.Start()
    }
    catch {
        Write-Check "Port 80 bind" "FAIL" ("{0}. Run PowerShell as Administrator and retry." -f $_.Exception.Message)
        $privateResult++
        $listener = $null
    }

    if ($listener) {
        $deadline = (Get-Date).AddSeconds($PortTestSeconds)
        $hit = $false
        while ((Get-Date) -lt $deadline -and -not $hit) {
            if (-not $listener.Pending()) {
                Start-Sleep -Milliseconds 500
                continue
            }
            $client = $listener.AcceptTcpClient()
            $remote = $client.Client.RemoteEndPoint.ToString()
            $bytes = [System.Text.Encoding]::ASCII.GetBytes(
                "HTTP/1.1 200 OK`r`nContent-Length: 0`r`nConnection: close`r`n`r`n")
            $client.GetStream().Write($bytes, 0, $bytes.Length)
            $client.Close()
            $hit = $true
            $outside = ($remote -split ":")[0]
            if ($outside -eq $publicIp -or (Test-PrivateAddress $outside)) {
                Write-Check "Inbound via port 80" "FAIL" ("connection from {0} (same-network private source) does not prove external routing" -f $remote)
                $privateResult++
            }
            else {
                Write-Check "Inbound via port 80" "PASS" ("connection from {0} - an outside client reached this machine through the router" -f $remote)
            }
        }
        $listener.Stop()
        if (-not $hit) {
            Write-Check "Inbound via port 80" "WARN" ("no connection in {0}s. Until a cellular phone can load http://{1}/preflight, the router is not forwarding port 80." -f $PortTestSeconds, $publicIp)
        }
    }
}
else {
    Write-Check "Inbound via port 80" "WARN" "not tested. Re-run with -TestPublicPort and load http://<public-ip>/preflight from a phone on cellular to prove the router forwards port 80."
}

# --- Verdict ---------------------------------------------------------------
Write-Host ""
if ($privateResult -gt 0) {
    Write-Host ("VERDICT: {0} blocker(s) - this machine is not deployable yet; fix the FAIL items above." -f $privateResult) -ForegroundColor Red
    exit 1
}
if ($unclearResult -gt 0) {
    Write-Host "VERDICT: no hard blockers found, but some checks are pending (port test, DNS). Decide after those are resolved." -ForegroundColor Yellow
    exit 2
}
Write-Host "VERDICT: no blockers. This machine is ready to attempt a public deployment." -ForegroundColor Green
exit 0