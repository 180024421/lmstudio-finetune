param(
    [ValidateSet("cpu", "gpu", "full", "web", "api", "all")]
    [string]$Mode = "full"
)

$Root = $PSScriptRoot
Set-Location $Root

# 清理可能干扰 pip 的代理（按需注释掉）
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$py = ".\.venv\Scripts\python.exe"
$pip = @($py, "-m", "pip")

& @pip install -U pip
& @pip cache purge 2>$null

switch ($Mode) {
    "cpu" {
        & @pip install --default-timeout=2000 --no-cache-dir -r requirements-cpu.txt
    }
    "gpu" {
        & @pip install --default-timeout=2000 --no-cache-dir -r requirements-gpu.txt
        & @pip install -r requirements.txt
    }
    "web" {
        & @pip install -r requirements-web.txt
    }
    "api" {
        & @pip install -r requirements-api.txt
    }
    "all" {
        & @pip install --default-timeout=2000 --no-cache-dir -r requirements-cpu.txt
        & @pip install -r requirements.txt -r requirements-web.txt -r requirements-api.txt
    }
    default {
        Write-Host "安装 CPU 版 PyTorch + 训练依赖..."
        & @pip install --default-timeout=2000 --no-cache-dir -r requirements-cpu.txt
        & @pip install -r requirements.txt
    }
}

if (-not (Test-Path "config.yaml")) {
    Copy-Item config.example.yaml config.yaml
}

Write-Host "`n运行环境诊断..."
& $py doctor.py 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[提示] Doctor 有未通过项（如 LM Studio 未启动），训练前请检查: .\run.ps1 -Doctor"
}

Write-Host "`n[OK] Setup ($Mode) 完成。运行: .\run.ps1 -Web 或 .\run.ps1 -Train"
Write-Host "Profile 示例: python train.py --profile dev"
