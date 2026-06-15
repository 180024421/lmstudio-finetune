param(
    [ValidateSet("cpu", "gpu", "full", "web", "api", "all")]
    [string]$Mode = "full"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""

function Get-VenvPython {
    $py = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $py) {
        return $py
    }
    return $null
}

function Ensure-Venv {
    $py = Get-VenvPython
    if ($py) {
        return $py
    }
    Write-Host "[setup] 创建虚拟环境 .venv ..."
    $systemPy = Get-Command python -ErrorAction SilentlyContinue
    if (-not $systemPy) {
        throw "未找到系统 Python，请先安装 Python 3.10+ 并加入 PATH"
    }
    & python -m venv (Join-Path $Root ".venv")
    if ($LASTEXITCODE -ne 0) {
        throw "创建虚拟环境失败"
    }
    $py = Get-VenvPython
    if (-not $py) {
        throw "虚拟环境创建后未找到 python.exe"
    }
    return $py
}

function Invoke-VenvPip {
    param(
        [string]$PythonExe,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$PipArgs
    )
    & $PythonExe -m pip @PipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pip 失败: pip $($PipArgs -join ' ')"
    }
}

$py = Ensure-Venv

Write-Host "[setup] 使用 Python: $py"
Invoke-VenvPip $py install -U pip
Invoke-VenvPip $py cache purge 2>$null

switch ($Mode) {
    "cpu" {
        Invoke-VenvPip $py install --default-timeout=2000 --no-cache-dir -r requirements-cpu.txt
    }
    "gpu" {
        Invoke-VenvPip $py install --default-timeout=2000 --no-cache-dir -r requirements-gpu.txt
        Invoke-VenvPip $py install -r requirements.txt
    }
    "web" {
        Invoke-VenvPip $py install -r requirements-web.txt
    }
    "api" {
        Invoke-VenvPip $py install -r requirements-api.txt
    }
    "all" {
        Invoke-VenvPip $py install --default-timeout=2000 --no-cache-dir -r requirements-cpu.txt
        Invoke-VenvPip $py install -r requirements.txt -r requirements-web.txt -r requirements-api.txt
    }
    default {
        Write-Host "安装 CPU 版 PyTorch + 训练依赖..."
        Invoke-VenvPip $py install --default-timeout=2000 --no-cache-dir -r requirements-cpu.txt
        Invoke-VenvPip $py install -r requirements.txt
    }
}

if (-not (Test-Path "config.yaml")) {
    Copy-Item "config.example.yaml" "config.yaml"
}

Write-Host ""
Write-Host "运行环境诊断..."
& $py doctor.py 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[提示] Doctor 有未通过项（如 LM Studio 未启动），训练前请检查: .\run.ps1 -Doctor"
}

Write-Host ""
Write-Host "[OK] Setup ($Mode) 完成。运行: .\start.cmd 或 .\run.ps1 -Web"
