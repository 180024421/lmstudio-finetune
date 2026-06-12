param(
    [string]$TaskName = "lmstudio-finetune-inbox",
    [ValidateSet("inbox", "pipeline", "alignment")]
    [string]$Action = "inbox",
    [int]$IntervalMinutes = 5,
    [string]$Profile = ""
)

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Error "请先运行 .\run.ps1 -Setup"
    exit 1
}

$py = Join-Path $Root ".venv\Scripts\python.exe"
$env:LMSTUDIO_PROFILE = $Profile

switch ($Action) {
    "inbox" {
        $args = "`"$py`" `"$Root\watch_inbox.py`" --once"
    }
    "pipeline" {
        $args = "`"$py`" `"$Root\run_pipeline.py`""
    }
    "alignment" {
        $args = "`"$py`" `"$Root\run_alignment.py`""
    }
}

$actionCmd = "cmd /c cd /d `"$Root`" && set HTTP_PROXY= && set HTTPS_PROXY= && $args"
$existing = schtasks /Query /TN $TaskName 2>$null
if ($LASTEXITCODE -eq 0) {
    schtasks /Delete /TN $TaskName /F | Out-Null
}

schtasks /Create /TN $TaskName /SC MINUTE /MO $IntervalMinutes /TR $actionCmd /F
if ($LASTEXITCODE -ne 0) {
    Write-Error "创建计划任务失败，请以管理员身份运行 PowerShell"
    exit 1
}

Write-Host "[OK] 已注册计划任务: $TaskName"
Write-Host "  动作: $Action  间隔: 每 $IntervalMinutes 分钟"
Write-Host "  查看: schtasks /Query /TN $TaskName"
Write-Host "  删除: schtasks /Delete /TN $TaskName /F"
