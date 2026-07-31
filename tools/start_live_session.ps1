<#
.SYNOPSIS
  PC→Pi live session launcher (ADR 0002 two-machine path).

.DESCRIPTION
  Shared orchestrator for start_live_session.bat (full: receiver + Pi) and
  start_pi_session.bat (Pi-only). Starts the capture receiver when needed,
  SSHes to the Pi with --kiosk, optionally launches Chromium on the Pi
  display, and optionally opens an SSH tunnel so the laptop can peek at
  http://127.0.0.1:8080.

.PARAMETER Mode
  full = start receiver, parse SESSION, then SSH Pi
  pi   = SSH Pi only (requires --session)

.EXAMPLE
  .\tools\start_live_session.bat
  .\tools\start_live_session.bat --resume 3 --no-tunnel
  .\tools\start_pi_session.bat --session 3
  .\tools\start_pi_session.bat --session 3 --receiver-url http://192.168.50.1:8077 --no-chromium
#>
[CmdletBinding()]
param(
    [ValidateSet('full', 'pi')]
    [string]$Mode = 'full',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Defaults (locked by design handoff 2026-07-09)
# ---------------------------------------------------------------------------
$SshUser = 'esears'
$SshHost = '192.168.50.2'
$PiRepo = '/home/esears/ljiblockcheck'
$ReceiverRoot = 'outputs\live_session'
$ReceiverUrl = 'http://192.168.50.1:8077'
$KioskUrl = 'http://127.0.0.1:8080'
$LocalTunnelPort = 8080
$RemoteKioskPort = 8080
$ReceiverTimeoutSec = 30
$ChromiumDelaySec = 4
$NoChromium = $false
$NoTunnel = $false
$KeepPiSession = $false
$ReviewCaptures = $false
$OpenRetrieval = $false
$Hybrid = $false
$HybridShadow = $false
$ProfileMode = $false
$Resume = $null
$Session = $null
$ShowHelp = $false

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot 'venv\Scripts\python.exe'

function Show-Usage {
    @"
PC→Pi live session launcher

Usage:
  tools\start_live_session.bat [options]
  tools\start_pi_session.bat --session N [options]
  powershell -File tools\start_live_session.ps1 -Mode full|pi [options]

Options:
  --session N              Pi session number (required for pi mode)
  --resume N               Full mode: run_receiver.py --resume N (not --start-session)
  --receiver-url URL       Receiver URL passed to the Pi (default: $ReceiverUrl)
  --receiver-root PATH     Receiver --root relative to repo (default: $ReceiverRoot)
  --ssh-user USER          SSH user (default: $SshUser)
  --ssh-host HOST          SSH host (default: $SshHost)
  --pi-repo PATH           Absolute path to repo on Pi (default: $PiRepo)
  --kiosk-url URL          Chromium URL on Pi (default: $KioskUrl)
  --local-tunnel-port N    Laptop local forward port (default: $LocalTunnelPort)
  --remote-kiosk-port N    Pi kiosk port to forward (default: $RemoteKioskPort)
  --no-chromium            Skip Chromium on Pi DISPLAY=:0
  --no-tunnel              Skip laptop SSH -L tunnel
  --keep-pi-session        Do not kill an existing run_pi_session.py on the Pi
  --review-captures        Enable ADR 0006 debug capture review gate on the Pi
  --open-retrieval          Enable open-retrieval mode (issue #147-#151) on the Pi
  --hybrid                 Enable Hybrid session scoring mode (#247 mode plumbing only) on the Pi
  --hybrid-shadow          Enable Hybrid Shadow session scoring mode (#247 mode plumbing only) on the Pi
  --profile                Print per-stage capture timings + persist profile_summary.csv (#168) on the Pi

  --open-retrieval/--hybrid/--hybrid-shadow are mutually exclusive.
  -h, --help               Show this help

Defaults match ADR 0002 (PC 192.168.50.1 / Pi 192.168.50.2).
"@
}

function Get-NextArg {
    param(
        [string[]]$List,
        [int]$Index,
        [string]$Flag
    )
    if ($Index + 1 -ge $List.Count) {
        throw "Missing value after $Flag"
    }
    return $List[$Index + 1]
}

function Parse-RestArgs {
    param([string[]]$List)
    # -File with only -Mode can leave a single empty remaining arg; ignore blanks.
    $List = @($List | Where-Object { $_ -and $_.Trim().Length -gt 0 })
    if ($List.Count -eq 0) { return }
    $i = 0
    while ($i -lt $List.Count) {
        $a = $List[$i]
        switch -Regex ($a) {
            '^(-\?|-h|--help)$' {
                $script:ShowHelp = $true
                $i++
            }
            '^--session$' {
                $script:Session = [int](Get-NextArg $List $i $a)
                $i += 2
            }
            '^--resume$' {
                $script:Resume = [int](Get-NextArg $List $i $a)
                $i += 2
            }
            '^--receiver-url$' {
                $script:ReceiverUrl = Get-NextArg $List $i $a
                $i += 2
            }
            '^--receiver-root$' {
                $script:ReceiverRoot = Get-NextArg $List $i $a
                $i += 2
            }
            '^--ssh-user$' {
                $script:SshUser = Get-NextArg $List $i $a
                $i += 2
            }
            '^--ssh-host$' {
                $script:SshHost = Get-NextArg $List $i $a
                $i += 2
            }
            '^--pi-repo$' {
                $script:PiRepo = Get-NextArg $List $i $a
                $i += 2
            }
            '^--kiosk-url$' {
                $script:KioskUrl = Get-NextArg $List $i $a
                $i += 2
            }
            '^--local-tunnel-port$' {
                $script:LocalTunnelPort = [int](Get-NextArg $List $i $a)
                $i += 2
            }
            '^--remote-kiosk-port$' {
                $script:RemoteKioskPort = [int](Get-NextArg $List $i $a)
                $i += 2
            }
            '^--no-chromium$' {
                $script:NoChromium = $true
                $i++
            }
            '^--no-tunnel$' {
                $script:NoTunnel = $true
                $i++
            }
            '^--keep-pi-session$' {
                $script:KeepPiSession = $true
                $i++
            }
            '^--review-captures$' {
                $script:ReviewCaptures = $true
                $i++
            }
            '^--open-retrieval$' {
                $script:OpenRetrieval = $true
                $i++
            }
            '^--hybrid$' {
                $script:Hybrid = $true
                $i++
            }
            '^--hybrid-shadow$' {
                $script:HybridShadow = $true
                $i++
            }
            '^--profile$' {
                $script:ProfileMode = $true
                $i++
            }
            default {
                throw "Unknown argument: $a`n$(Show-Usage)"
            }
        }
    }
}

function Assert-SafeInt {
    param([int]$Value, [string]$Name)
    if ($Value -lt 1 -or $Value -gt 2147483647) {
        throw "$Name must be a positive integer (got $Value)"
    }
}

function Assert-SafeUrl {
    param([string]$Value, [string]$Name)
    if ($Value -notmatch '^https?://[A-Za-z0-9._~:/?#\[\]@!$&()*+,;=%-]+$') {
        throw "$Name is not a safe http(s) URL: $Value"
    }
    if ($Value -match "['`"`$\\;|&<>]") {
        throw "$Name contains characters that are not allowed: $Value"
    }
}

function Assert-SafeSshToken {
    param([string]$Value, [string]$Name)
    if ($Value -notmatch '^[A-Za-z0-9._:-]+$') {
        throw "$Name contains unsafe characters: $Value"
    }
}

function Assert-SafeUnixPath {
    param([string]$Value, [string]$Name)
    if ($Value -notmatch '^/[A-Za-z0-9._/-]+$') {
        throw "$Name must be an absolute Unix path with safe characters: $Value"
    }
}

function Assert-SafeReceiverRoot {
    param([string]$Value, [string]$Name)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "$Name must not be empty"
    }
    if ([System.IO.Path]::IsPathRooted($Value)) {
        throw "$Name must be relative to the repo root (got absolute path: $Value)"
    }
    if ($Value -match '(^|[\\/])\.\.([\\/]|$)') {
        throw "$Name must not contain '..' segments: $Value"
    }
    if ($Value -match '[;$`|&<>"]') {
        throw "$Name contains characters that are not allowed: $Value"
    }
}

function ConvertTo-BashSingleQuoted {
    param([string]$Text)
    return "'" + ($Text -replace "'", "'\''") + "'"
}

function ConvertTo-PsSingleQuoted {
    param([string]$Text)
    return "'" + ($Text -replace "'", "''") + "'"
}

function Test-LocalPortInUse {
    param([int]$Port)
    try {
        $listener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Loopback,
            $Port
        )
        $listener.Start()
        $listener.Stop()
        return $false
    }
    catch {
        return $true
    }
}

function Assert-ReceiverPortAvailable {
    <#
    .SYNOPSIS
      Refuse to start a second capture receiver on the same local port.

    .DESCRIPTION
      `run_receiver.py` uses an address-reuse socket option, so Windows can
      otherwise leave multiple detached receiver processes listening on the
      same address and dispatch Pi requests between them.  They then race to
      write one `outputs\live_session` tree.  The launcher owns exactly one
      receiver per full run; an existing listener must be stopped or resumed
      intentionally before another full run begins.
    #>
    param([string]$Url)

    $uri = [System.Uri]$Url
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $uri.Port `
        -ErrorAction SilentlyContinue)
    if ($listeners.Count -eq 0) {
        return
    }

    $pids = ($listeners | Select-Object -ExpandProperty OwningProcess -Unique) -join ', '
    throw @"
Receiver port $($uri.Port) is already in use (PID(s): $pids).
Do not start another full session against the same outputs\\live_session tree.
Close the existing receiver window or stop those receiver processes, then retry.
"@
}

function Stop-ManagedProcessTree {
    <#
    .SYNOPSIS
      Stop a launcher-owned receiver and every child it started.

    `Start-ReceiverWindow` launches a PowerShell wrapper, which launches
    Python plus Tee-Object. `Stop-Process` on the wrapper alone can orphan the
    Python receiver, so use taskkill's `/T` tree boundary instead.
    #>
    param(
        [System.Diagnostics.Process]$Process,
        [string]$Label
    )
    if ($null -eq $Process) { return }
    try {
        if (-not $Process.HasExited) {
            Write-Host "Stopping $Label (pid $($Process.Id))..."
            & taskkill.exe /PID $Process.Id /T /F 2>$null | Out-Null
        }
    }
    catch {
        # Best effort only: shutdown must not mask the original SSH error.
    }
}

function Wait-ReceiverReady {
    param(
        [string]$LogPath,
        [int]$TimeoutSec
    )
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    $sessionNumber = $null
    $boundUrl = $null
    while ((Get-Date) -lt $deadline) {
        if (Test-Path -LiteralPath $LogPath) {
            $text = Get-Content -LiteralPath $LogPath -Raw -ErrorAction SilentlyContinue
            if ($text) {
                if (-not $boundUrl) {
                    $m = [regex]::Match($text, 'Capture receiver bound to (https?://\S+)')
                    if ($m.Success) {
                        $boundUrl = $m.Groups[1].Value.Trim()
                    }
                }
                if (-not $sessionNumber) {
                    $m = [regex]::Match($text, '(?m)^SESSION\s+(\d+)\b')
                    if ($m.Success) {
                        $sessionNumber = [int]$m.Groups[1].Value
                    }
                }
                if ($sessionNumber -and $boundUrl) {
                    return @{
                        Session = $sessionNumber
                        Url     = $boundUrl
                    }
                }
            }
        }
        Start-Sleep -Milliseconds 250
    }
    throw @"
Receiver did not print SESSION N within ${TimeoutSec}s.
Check the receiver console window and log:
  $LogPath
Common causes: venv missing, --root unwritable, bind address unreachable,
or neither --start-session nor --resume was passed.
"@
}

function Get-ReceiverSessionArgs {
    <#
    .SYNOPSIS
      Build run_receiver.py's session-lifecycle argument string.

    .DESCRIPTION
      #269: one flag on this top-level launcher must reach BOTH the receiver
      (this function's output, consumed by Start-ReceiverWindow) and the Pi
      (Build-RemotePiCommand, below) -- not require typing it twice. The mode
      flags are only meaningful for --start-session (a fresh session mints
      sessions.session_mode once); --resume attaches to an already-created
      session whose mode is ALREADY durable and cannot be changed by a flag.
      A confirmed HIGH-severity bug (adversarial review, post-#269): this
      function used to append the mode flag to "--resume N" too. run_receiver.py
      resolves --hybrid/etc unconditionally but only threads the result into
      store.start_session on the --start-session branch, so on --resume the
      flag was silently discarded -- NOT harmless, because an operator relying
      on it would believe the resumed session was in that mode when its durable
      sessions.session_mode says otherwise. `tools/run_pi_session.py::main` now
      hard-refuses to start on exactly that mismatch (#269 hardening); this
      function no longer manufactures one on the receiver side. Kept as its own
      function (mirrors Build-RemotePiCommand) so the argument-assembly logic is
      testable without launching a real process window.
    #>
    param(
        [object]$ResumeSession,
        [bool]$OpenRetrieval = $false,
        [bool]$Hybrid = $false,
        [bool]$HybridShadow = $false
    )
    if ($null -ne $ResumeSession) {
        return "--resume $ResumeSession"
    }
    $sessionArgs = '--start-session'
    if ($OpenRetrieval) { $sessionArgs = "$sessionArgs --open-retrieval" }
    elseif ($Hybrid) { $sessionArgs = "$sessionArgs --hybrid" }
    elseif ($HybridShadow) { $sessionArgs = "$sessionArgs --hybrid-shadow" }
    return $sessionArgs
}

function Start-ReceiverWindow {
    param(
        [string]$PythonPath,
        [string]$RootRel,
        [object]$ResumeSession,
        [string]$LogPath,
        [bool]$OpenRetrieval = $false,
        [bool]$Hybrid = $false,
        [bool]$HybridShadow = $false
    )
    if (-not (Test-Path -LiteralPath $PythonPath)) {
        throw "Project venv python not found: $PythonPath"
    }

    Assert-ReceiverPortAvailable -Url $ReceiverUrl

    $rootAbs = Join-Path $RepoRoot $RootRel
    New-Item -ItemType Directory -Force -Path $rootAbs | Out-Null

    if (Test-Path -LiteralPath $LogPath) {
        Remove-Item -LiteralPath $LogPath -Force
    }
    New-Item -ItemType File -Path $LogPath -Force | Out-Null

    $sessionArgs = Get-ReceiverSessionArgs -ResumeSession $ResumeSession `
        -OpenRetrieval $OpenRetrieval -Hybrid $Hybrid -HybridShadow $HybridShadow

    # New console: run receiver and tee stdout/stderr into the log we poll.
    # PYTHONUNBUFFERED is required: when stdout is piped, Python block-buffers
    # and SESSION would never appear within the timeout.
    $inner = @"
`$ErrorActionPreference = 'Continue'
`$env:PYTHONUNBUFFERED = '1'
Set-Location -LiteralPath $(ConvertTo-PsSingleQuoted $RepoRoot)
& $(ConvertTo-PsSingleQuoted $PythonPath) -u tools\run_receiver.py --root $(ConvertTo-PsSingleQuoted $RootRel) $sessionArgs 2>&1 |
  Tee-Object -FilePath $(ConvertTo-PsSingleQuoted $LogPath)
Write-Host ''
Write-Host 'Receiver exited. You can close this window.'
"@

    $proc = Start-Process -FilePath 'powershell.exe' -WorkingDirectory $RepoRoot -ArgumentList @(
        '-NoProfile',
        '-NoExit',
        '-ExecutionPolicy', 'Bypass',
        '-Command', $inner
    ) -PassThru

    Write-Host "Started receiver in a new window (pid $($proc.Id), log: $LogPath)"
    return $proc
}

function Write-ManualTunnelHint {
    param(
        [string]$Target,
        [int]$LocalPort,
        [int]$RemotePort
    )
    $forward = "${LocalPort}:127.0.0.1:${RemotePort}"
    Write-Host @"
Manual tunnel (run in a separate PowerShell window):
  ssh -N -L $forward $Target
Then open: http://127.0.0.1:$LocalPort
"@
}

function Start-LaptopTunnel {
    param(
        [string]$Target,
        [int]$LocalPort,
        [int]$RemotePort
    )
    $forward = "${LocalPort}:127.0.0.1:${RemotePort}"
    $manual = "ssh -N -L $forward $Target"

    if (Test-LocalPortInUse -Port $LocalPort) {
        Write-Host @"

ERROR: Local port $LocalPort is already in use.
  - Close the other process using $LocalPort, or
  - Re-run with --local-tunnel-port <free-port>
    then open http://127.0.0.1:<free-port> on this PC.
Tunnel was NOT started.
"@
        Write-ManualTunnelHint -Target $Target -LocalPort $LocalPort -RemotePort $RemotePort
        return $null
    }

    # Visible console + BatchMode: auth/host failures must surface (not hang
    # minimized waiting for a password). ConnectTimeout keeps QA from stalling.
    $proc = Start-Process -FilePath 'ssh' -ArgumentList @(
        '-N',
        '-o', 'BatchMode=yes',
        '-o', 'ConnectTimeout=10',
        '-L', $forward,
        $Target
    ) -PassThru

    Start-Sleep -Seconds 2
    if ($proc.HasExited) {
        Write-Host @"

ERROR: Laptop tunnel exited immediately (exit $($proc.ExitCode)).
Likely SSH auth/host failure, or port forward rejected.
"@ -ForegroundColor Red
        Write-ManualTunnelHint -Target $Target -LocalPort $LocalPort -RemotePort $RemotePort
        return $null
    }

    Write-Host "Laptop tunnel started (pid $($proc.Id)): $manual"
    Write-Host "Peek URL: http://127.0.0.1:$LocalPort"
    return $proc
}

function ConvertTo-UnixNewlines {
    param([string]$Text)
    # Windows PowerShell here-strings are CRLF; bash then sees `set -e\r` /
    # `sleep 4\r` and fails with "invalid option" / "numeric argument required".
    return ($Text -replace "`r`n", "`n" -replace "`r", "`n")
}

function Build-RemotePiCommand {
    param(
        [string]$Repo,
        [string]$Url,
        [int]$SessionNumber,
        [string]$ChromiumUrl,
        [bool]$StartChromium,
        [int]$DelaySec,
        [bool]$KillStalePiSession = $true,
        [bool]$ReviewCaptures = $false,
        [bool]$OpenRetrieval = $false,
        [bool]$Hybrid = $false,
        [bool]$HybridShadow = $false,
        [bool]$ProfileMode = $false
    )

    $startChromiumFlag = if ($StartChromium) { '1' } else { '0' }
    $killStaleFlag = if ($KillStalePiSession) { '1' } else { '0' }
    $reviewCapturesFlag = if ($ReviewCaptures) { ' --review-captures' } else { '' }
    $openRetrievalFlag = if ($OpenRetrieval) { ' --open-retrieval' } else { '' }
    $hybridFlag = if ($Hybrid) { ' --hybrid' } else { '' }
    $hybridShadowFlag = if ($HybridShadow) { ' --hybrid-shadow' } else { '' }
    $profileFlag = if ($ProfileMode) { ' --profile' } else { '' }
    # Values are embedded inside a single-quoted bash -lc string; validate first.
    $script = @"
set -e
cd $(ConvertTo-BashSingleQuoted $Repo) || {
  echo "Pi repo not found: $Repo" >&2
  exit 1
}
export DISPLAY=:0
START_CHROMIUM=$startChromiumFlag
KILL_STALE=$killStaleFlag
KIOSK_URL=$(ConvertTo-BashSingleQuoted $ChromiumUrl)
TUNNEL_HINT=$(ConvertTo-BashSingleQuoted "http://127.0.0.1:$LocalTunnelPort")
# Only one process can hold the Pi camera. Closing the laptop SSH window does
# not always kill a prior run_pi_session.py, which then causes
# "Device or resource busy" / "Pipeline handler in use by another process".
if [ "`$KILL_STALE" = "1" ]; then
  STALE_PIDS=`$(pgrep -f 'tools/run_pi_session.py' || true)
  if [ -n "`$STALE_PIDS" ]; then
    echo "Stopping leftover run_pi_session.py (pids: `$STALE_PIDS) so the camera is free..."
    # TERM first so ActionLogger / camera close can run; then KILL stragglers.
    kill `$STALE_PIDS 2>/dev/null || true
    sleep 1
    STALE_PIDS=`$(pgrep -f 'tools/run_pi_session.py' || true)
    if [ -n "`$STALE_PIDS" ]; then
      echo "Force-killing leftover run_pi_session.py (pids: `$STALE_PIDS)..."
      kill -9 `$STALE_PIDS 2>/dev/null || true
      sleep 1
    fi
  fi
fi
if [ "`$START_CHROMIUM" = "1" ]; then
  (
    sleep $DelaySec
    # The SSH shell is not the graphical login shell.  DISPLAY alone usually
    # works, but Xwayland also needs the login user's Xauthority cookie.
    export XAUTHORITY="`${XAUTHORITY:-`$HOME/.Xauthority}"
    export XDG_RUNTIME_DIR="`${XDG_RUNTIME_DIR:-/run/user/`$(id -u)}"
    CHROME_BIN=""
    if command -v chromium-browser >/dev/null 2>&1; then
      CHROME_BIN=chromium-browser
    elif command -v chromium >/dev/null 2>&1; then
      CHROME_BIN=chromium
    fi
    if [ -z "`$CHROME_BIN" ]; then
      echo "WARN: chromium-browser/chromium not found." >&2
      echo "Session continues. Use laptop tunnel: `$TUNNEL_HINT" >&2
      exit 0
    fi
    # A normal Chromium invocation exits after handing its URL to an existing
    # browser process.  Use a kiosk-owned profile and inspect that process,
    # rather than treating the short-lived handoff PID as a browser failure.
    KIOSK_PROFILE=/tmp/lji-kiosk-chromium-profile
    KIOSK_LOG=/tmp/lji-kiosk-chromium.log
    : > "`$KIOSK_LOG"
    attempt=1
    while [ "`$attempt" -le 3 ]; do
      `$CHROME_BIN --kiosk --ozone-platform=x11 \
        --user-data-dir="`$KIOSK_PROFILE" --no-first-run \
        --disable-session-crashed-bubble "`$KIOSK_URL" >>"`$KIOSK_LOG" 2>&1 &
      sleep 2
      KIOSK_PIDS=`$(pgrep -f -- "--user-data-dir=`$KIOSK_PROFILE" || true)
      if [ -n "`$KIOSK_PIDS" ]; then
        echo "Kiosk Chromium ready (pids: `$KIOSK_PIDS)."
        exit 0
      fi
      echo "WARN: Kiosk Chromium was not running after attempt `$attempt/3; retrying..." >&2
      attempt=`$((attempt + 1))
      sleep 1
    done
    echo "WARN: Kiosk Chromium did not start after 3 attempts." >&2
    echo "Last log lines:" >&2
    tail -n 20 "`$KIOSK_LOG" 2>/dev/null >&2 || true
    echo "Session continues. Use laptop tunnel: `$TUNNEL_HINT" >&2
  ) &
fi
echo "NOTE: Pi must have run_pi_session.py --kiosk (KioskServer). Without it, argparse exits immediately."
echo "Starting Pi session $SessionNumber against $Url (--kiosk)"
exec ./venv/bin/python tools/run_pi_session.py --receiver-url $(ConvertTo-BashSingleQuoted $Url) --session $SessionNumber --kiosk$reviewCapturesFlag$openRetrievalFlag$hybridFlag$hybridShadowFlag$profileFlag
"@
    return (ConvertTo-UnixNewlines -Text $script)
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
try {
    Parse-RestArgs -List $Rest
}
catch {
    Write-Host $_ -ForegroundColor Red
    exit 2
}

if ($ShowHelp) {
    Show-Usage
    exit 0
}

# #247: --open-retrieval/--hybrid/--hybrid-shadow are mutually exclusive
# session scoring modes. Reject any two-or-more combination here, before
# SSHing to the Pi at all -- the Python-side `resolve_session_mode` is the
# single source of truth for this rule; this is a fail-fast mirror so a bad
# launch command never reaches the Pi.
$scoringModeFlagCount = @($OpenRetrieval, $Hybrid, $HybridShadow) |
    Where-Object { $_ } | Measure-Object | Select-Object -ExpandProperty Count
if ($scoringModeFlagCount -gt 1) {
    Write-Host "Only one of --open-retrieval, --hybrid, --hybrid-shadow may be set." -ForegroundColor Red
    exit 2
}

$UserReceiverUrl = $ReceiverUrl

try {
    Assert-SafeSshToken -Value $SshUser -Name 'SSH user'
    Assert-SafeSshToken -Value $SshHost -Name 'SSH host'
    Assert-SafeUnixPath -Value $PiRepo -Name 'Pi repo'
    Assert-SafeUrl -Value $ReceiverUrl -Name 'Receiver URL'
    Assert-SafeUrl -Value $KioskUrl -Name 'Kiosk URL'
    Assert-SafeReceiverRoot -Value $ReceiverRoot -Name 'Receiver root'
    Assert-SafeInt -Value $LocalTunnelPort -Name 'Local tunnel port'
    Assert-SafeInt -Value $RemoteKioskPort -Name 'Remote kiosk port'

    if ($null -ne $Resume) {
        Assert-SafeInt -Value $Resume -Name '--resume'
    }
    if ($null -ne $Session) {
        Assert-SafeInt -Value $Session -Name '--session'
    }
}
catch {
    Write-Host $_ -ForegroundColor Red
    exit 2
}

if ($Mode -eq 'pi' -and $null -eq $Session) {
    Write-Host "Pi mode requires --session N`n$(Show-Usage)" -ForegroundColor Red
    exit 2
}
if ($Mode -eq 'full' -and ($null -ne $Session) -and ($null -eq $Resume)) {
    Write-Host "Note: --session is ignored in full mode; session number comes from the receiver." -ForegroundColor Yellow
}
if ($Mode -eq 'full' -and ($null -ne $Resume) -and ($null -ne $Session)) {
    Write-Host "Pass only one of --resume (full launcher) or use start_pi_session.bat --session." -ForegroundColor Red
    exit 2
}

$sshTarget = "$SshUser@$SshHost"
$tunnelProc = $null
$receiverProc = $null
$sshExit = 0
$failed = $false
$receiverLog = Join-Path $env:TEMP ("lji-receiver-{0:yyyyMMdd-HHmmss}.log" -f (Get-Date))

Write-Host "=== LJI PC→Pi session launcher ==="
Write-Host "Mode: $Mode"
Write-Host "SSH:  $sshTarget"
Write-Host "Pi:   $PiRepo"

try {
    if ($Mode -eq 'full') {
        $receiverProc = Start-ReceiverWindow -PythonPath $Python -RootRel $ReceiverRoot `
            -ResumeSession $Resume -LogPath $receiverLog `
            -OpenRetrieval $OpenRetrieval -Hybrid $Hybrid -HybridShadow $HybridShadow
        Write-Host "Waiting up to ${ReceiverTimeoutSec}s for SESSION line..."
        $ready = Wait-ReceiverReady -LogPath $receiverLog -TimeoutSec $ReceiverTimeoutSec
        $Session = $ready.Session
        # Prefer the URL the receiver actually bound, not only the default knob.
        $ReceiverUrl = $ready.Url
        Assert-SafeUrl -Value $ReceiverUrl -Name 'Bound receiver URL'
        Write-Host "Receiver ready: SESSION $Session at $ReceiverUrl"
        if ($UserReceiverUrl -ne $ReceiverUrl) {
            Write-Host "Note: using bound receiver URL ($ReceiverUrl); ignoring --receiver-url ($UserReceiverUrl) in full mode." -ForegroundColor Yellow
        }
    }

    if (-not $NoTunnel) {
        $tunnelProc = Start-LaptopTunnel -Target $sshTarget `
            -LocalPort $LocalTunnelPort -RemotePort $RemoteKioskPort
        if ($tunnelProc) {
            $peek = "http://127.0.0.1:$LocalTunnelPort"
            Write-Host "Peek URL: $peek (browser opens once the kiosk answers)"
            # Don't open the browser yet — KioskServer starts only after SSH
            # launches run_pi_session.py --kiosk. Poll in the background.
            Start-Job -ScriptBlock {
                param($Url)
                for ($i = 0; $i -lt 60; $i++) {
                    Start-Sleep -Seconds 1
                    try {
                        $r = Invoke-WebRequest -Uri $Url -TimeoutSec 1 -UseBasicParsing
                        if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) {
                            Start-Process $Url | Out-Null
                            return
                        }
                    }
                    catch {
                        # keep waiting
                    }
                }
            } -ArgumentList $peek | Out-Null
        }
    }
    else {
        Write-Host "Skipping laptop tunnel (--no-tunnel)."
        Write-Host "Manual tunnel: ssh -N -L ${LocalTunnelPort}:127.0.0.1:${RemoteKioskPort} $sshTarget"
    }

    $remote = Build-RemotePiCommand -Repo $PiRepo -Url $ReceiverUrl `
        -SessionNumber $Session -ChromiumUrl $KioskUrl `
        -StartChromium (-not $NoChromium) -DelaySec $ChromiumDelaySec `
        -KillStalePiSession (-not $KeepPiSession) -ReviewCaptures $ReviewCaptures `
        -OpenRetrieval $OpenRetrieval -Hybrid $Hybrid -HybridShadow $HybridShadow `
        -ProfileMode $ProfileMode

    Write-Host ""
    Write-Host "Opening interactive SSH. Pi console (pi>) will take over."
    Write-Host "Exit: Ctrl+C or Ctrl+D at pi> (releases camera). Closing this SSH window also releases it."
    Write-Host "Close the receiver window when done."
    Write-Host ""

    # Encode the remote script as base64 so nested quotes / sshd double-quote
    # wrapping cannot break bash -lc parsing (Windows OpenSSH re-wraps the
    # remote command; a multiline single-quoted script with "" inside fails).
    #
    # Critical: do NOT pipe the script into bash's stdin. run_pi_session.py
    # reads the interactive pi> prompt from stdin; a pipe causes immediate EOF,
    # the session exits, and the laptop tunnel is torn down.
    $remoteBytes = [System.Text.Encoding]::UTF8.GetBytes($remote)
    $remoteB64 = [Convert]::ToBase64String($remoteBytes)
    # Single-quoted so PowerShell leaves bash's $$ (PID) alone.
    $remoteRunner = 'echo ' + $remoteB64 + ' | base64 -d > /tmp/lji-live-session.$$.sh && exec bash /tmp/lji-live-session.$$.sh'
    $bashLc = "bash -lc " + (ConvertTo-BashSingleQuoted $remoteRunner)
    & ssh -t $sshTarget $bashLc
    $sshExit = $LASTEXITCODE
}
catch {
    $failed = $true
    Write-Host $_ -ForegroundColor Red
}
finally {
    Stop-ManagedProcessTree -Process $tunnelProc -Label 'laptop tunnel'
    Stop-ManagedProcessTree -Process $receiverProc -Label 'receiver'
}

if ($failed) {
    exit 1
}
if ($sshExit -ne 0) {
    Write-Host "SSH exited with code $sshExit (auth/host failure or remote command error)." -ForegroundColor Red
    exit $sshExit
}

Write-Host "Done. Launcher-owned receiver has stopped."
exit 0
