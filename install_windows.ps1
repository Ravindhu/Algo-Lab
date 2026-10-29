<#
AlgoLab Windows installer: one command installs everything and starts the algo.

  Standalone (this PC only, no VPS):
    powershell -ExecutionPolicy Bypass -File .\install_windows.ps1 -Mode Standalone

  With a VPS (this PC also syncs data and account state to the VPS):
    powershell -ExecutionPolicy Bypass -File .\install_windows.ps1 -Mode Vps

  Other actions:
    ... -Action start | stop | status | uninstall

Requires: Windows 10/11, internet, and an MT5 broker login (you can finish login later).
#>
param(
  [ValidateSet("Standalone", "Vps")] [string]$Mode = "Standalone",
  [ValidateSet("install", "start", "stop", "status", "uninstall")] [string]$Action = "install",
  [int]$IntervalMinutes = 15,
  [int]$FrameworkMinutes = 5
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$Mt5Exe = "C:\Program Files\MetaTrader 5\terminal64.exe"
$TaskSync = "AlgoLab-Sync"
$TaskLive = "AlgoLab-Live"
$TaskFw = "AlgoLab-Framework"
$Tasks = @($TaskSync, $TaskLive, $TaskFw)

function Step([string]$msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

function Run {
  param([Parameter(ValueFromRemainingArguments = $true)] $Cmd)
  & $Cmd[0] @($Cmd[1..($Cmd.Count - 1)])
  if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $($Cmd -join ' ')" }
}

function Install-Python {
  Step "Python 3.13"
  & py -3.13 --version *> $null
  if ($LASTEXITCODE -eq 0) { Write-Host "Python 3.13 found."; return }
  winget install --id Python.Python.3.13 -e --silent --accept-package-agreements --accept-source-agreements
  & py -3.13 --version *> $null
  if ($LASTEXITCODE -ne 0) {
    throw "Python 3.13 installed. Open a NEW PowerShell window and run this installer again."
  }
}

function Install-Mt5 {
  Step "MetaTrader 5 terminal"
  if (Test-Path $Mt5Exe) { Write-Host "MT5 found."; return }
  $setup = Join-Path $env:TEMP "mt5setup.exe"
  Invoke-WebRequest -Uri "https://download.mql5.com/cdn/web/metaquotes.ltd/mt5/mt5setup.exe" -OutFile $setup
  Start-Process -FilePath $setup -ArgumentList "/auto" -Wait
  if (-not (Test-Path $Mt5Exe)) {
    Write-Warning "MT5 did not install silently. Run $setup by hand, or install your broker's MT5, then re-run this installer."
  }
}

function Install-WinScp {
  Step "WinSCP (for .env upload to VPS)"
  if ((Test-Path "C:\Program Files (x86)\WinSCP\WinSCP.com") -or (Test-Path "C:\Program Files\WinSCP\WinSCP.com")) {
    Write-Host "WinSCP found."; return
  }
  winget install --id WinSCP.WinSCP -e --silent --accept-package-agreements --accept-source-agreements
}

function Install-Python-Deps {
  Step "Python packages (requirements.txt + requirements-live.txt)"
  if (-not (Test-Path $Py)) { Run py -3.13 -m venv $Venv }
  Run $Py -m pip install --upgrade pip
  Run $Py -m pip install -r (Join-Path $Root "requirements.txt")
  Run $Py -m pip install -r (Join-Path $Root "requirements-live.txt")
  Run $Py -m pip install -e $Root
}

function Prepare-Folders-And-Env {
  Step "Folders and .env"
  foreach ($d in @("data", "reports", "logs", "state")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $d) | Out-Null
  }
  $envFile = Join-Path $Root ".env"
  if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $Root ".env.example") $envFile
    # Windows runs the local profile; the VPS profile is for the server only.
    (Get-Content $envFile) -replace '^ALGOLAB_PROFILE=.*', 'ALGOLAB_PROFILE=local' | Set-Content $envFile
    Write-Warning "Created .env. Fill in MT5_LOGIN, MT5_PASSWORD, MT5_SERVER (and VPS_* values for Vps mode)."
  } else {
    Write-Host ".env already exists, kept as is."
  }
}

function Check-Mt5 {
  Step "MetaTrader 5 connection check (never blocks the install)"
  & $Py (Join-Path $Root "scripts\check_mt5.py")
  if ($LASTEXITCODE -ne 0) { Write-Warning "MT5 check returned an error code. Review the output above." }
}

function Register-AlgoTasks {
  Step "Scheduled tasks (run while you are logged in)"
  $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
  $duration = New-TimeSpan -Days 3650

  # Sync (minute 0), then the full live cycle 4 minutes later. The framework has its own task so a
  # lower timeframe sees a fresh closed bar every 5 minutes.
  $defs = @(
    @{ Name = $TaskSync; Args = "-m algolab.bridge.windows_sync";       Start = (Get-Date).AddMinutes(1); Every = $IntervalMinutes },
    @{ Name = $TaskLive; Args = "-m algolab.live.pipeline";             Start = (Get-Date).AddMinutes(5); Every = $IntervalMinutes },
    @{ Name = $TaskFw;   Args = "-m algolab.live.pipeline --framework"; Start = (Get-Date).AddMinutes(2); Every = $FrameworkMinutes }
  )
  foreach ($d in $defs) {
    $repeat = New-TimeSpan -Minutes $d.Every
    $action = New-ScheduledTaskAction -Execute $Py -Argument $d.Args -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -Once -At $d.Start -RepetitionInterval $repeat -RepetitionDuration $duration
    Register-ScheduledTask -TaskName $d.Name -Action $action -Trigger $trigger `
      -Settings $settings -Principal $principal -Force | Out-Null
    Write-Host "registered $($d.Name) every $($d.Every) min"
  }
}

function Show-Vps-Upload {
  if ($Mode -ne "Vps") { return }
  Step "Upload .env to VPS"
  $script = Join-Path $Root "scripts\windows\upload_env.txt"
  if (Select-String -Path $script -Pattern '<' -Quiet) {
    Write-Warning "Edit scripts\windows\upload_env.txt first: replace the <...> placeholders (VPS user, host, key, fingerprint)."
    return
  }
  $answer = Read-Host "Upload .env to the VPS now? (y/N)"
  if ($answer -eq "y") { Run cmd /c (Join-Path $Root "scripts\windows\upload_env.bat") }
}

function Do-Install {
  Install-Python
  Install-Mt5
  if ($Mode -eq "Vps") { Install-WinScp }
  Install-Python-Deps
  Prepare-Folders-And-Env
  Check-Mt5
  Register-AlgoTasks
  Show-Vps-Upload
  Step "Start"
  Do-Start
  Write-Host "`nAlgoLab installed in $Mode mode." -ForegroundColor Green
  Write-Host "Next: log in to MT5 once in the terminal window, keep DRY_RUN=true, then check: .\install_windows.ps1 -Action status"
}

function Do-Start {
  foreach ($t in $Tasks) {
    Enable-ScheduledTask -TaskName $t | Out-Null
    Start-ScheduledTask -TaskName $t
    Write-Host "started $t"
  }
}

function Do-Stop {
  foreach ($t in $Tasks) {
    Stop-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
    Disable-ScheduledTask -TaskName $t | Out-Null
    Write-Host "stopped $t"
  }
}

function Do-Status {
  foreach ($t in $Tasks) {
    $task = Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
    if ($task) {
      $info = Get-ScheduledTaskInfo -TaskName $t
      Write-Host ("{0}: {1}, last result {2}, last run {3}" -f $t, $task.State, $info.LastTaskResult, $info.LastRunTime)
    } else {
      Write-Host "$t: not installed"
    }
  }
  $last = Join-Path $Root "state\live_last_cycle.json"
  if (Test-Path $last) { Step "Last live cycle"; Get-Content $last }
}

function Do-Uninstall {
  Do-Stop
  foreach ($t in $Tasks) {
    Unregister-ScheduledTask -TaskName $t -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "removed $t"
  }
  Write-Host "Scheduled tasks removed. Project files and .env were kept."
}

switch ($Action) {
  "install"   { Do-Install }
  "start"     { Do-Start }
  "stop"      { Do-Stop }
  "status"    { Do-Status }
  "uninstall" { Do-Uninstall }
}
