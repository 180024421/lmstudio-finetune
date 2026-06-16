param(

    [ValidateSet("cpu", "gpu", "full", "web", "api", "all")]

    [string]$Mode = "full",

    [ValidateSet("", "cn", "official")]

    [string]$Mirror = ""

)



$ErrorActionPreference = "Stop"

$Root = $PSScriptRoot

Set-Location $Root



$env:HTTP_PROXY = ""

$env:HTTPS_PROXY = ""

$env:ALL_PROXY = ""



function Get-VenvPython {

    $py = Join-Path $Root ".venv\Scripts\python.exe"

    if (Test-Path $py) { return $py }

    return $null

}



function Ensure-Venv {

    $py = Get-VenvPython

    if ($py) { return $py }

    Write-Host "[setup] 创建虚拟环境 .venv ..."

    $systemPy = Get-Command python -ErrorAction SilentlyContinue

    if (-not $systemPy) { throw "未找到系统 Python，请先安装 Python 3.10+ 并加入 PATH" }

    & python -m venv (Join-Path $Root ".venv")

    if ($LASTEXITCODE -ne 0) { throw "创建虚拟环境失败" }

    $py = Get-VenvPython

    if (-not $py) { throw "虚拟环境创建后未找到 python.exe" }

    return $py

}



function Get-PipIndexArgs {

    if ($Mirror -eq "cn") {

        return @("-i", "https://pypi.tuna.tsinghua.edu.cn/simple")

    }

    return @()

}



function Invoke-VenvPip {

    param(

        [string]$PythonExe,

        [Parameter(ValueFromRemainingArguments = $true)]

        [string[]]$PipArgs

    )

    $maxRetries = 3

    for ($attempt = 1; $attempt -le $maxRetries; $attempt++) {

        & $PythonExe -m pip @PipArgs

        if ($LASTEXITCODE -eq 0) { return }

        if ($attempt -lt $maxRetries) {

            Write-Host "[setup] pip 失败，5 秒后重试 ($attempt/$maxRetries)..."

            Start-Sleep -Seconds 5

        }

    }

    throw "pip 失败: pip $($PipArgs -join ' ')"

}



function Install-CpuTorch {

    param([string]$PythonExe)

    $idx = Get-PipIndexArgs

    if ($Mirror -eq "cn") {

        Write-Host "[setup] 国内镜像安装 CPU torch（PyPI）..."

        Invoke-VenvPip $PythonExe install --default-timeout=2000 --no-cache-dir @idx torch

    }

    else {

        Invoke-VenvPip $PythonExe install --default-timeout=2000 --no-cache-dir -r requirements-cpu.txt

    }

}



$py = Ensure-Venv

$pipIdx = Get-PipIndexArgs



Write-Host "[setup] 使用 Python: $py"

if ($Mirror -eq "cn") { Write-Host "[setup] 镜像: 清华大学 PyPI" }

Invoke-VenvPip $py install -U pip @pipIdx

Invoke-VenvPip $py cache purge 2>$null



switch ($Mode) {

    "cpu" { Install-CpuTorch $py }

    "gpu" {

        Invoke-VenvPip $py install --default-timeout=2000 --no-cache-dir -r requirements-gpu.txt

        Invoke-VenvPip $py install @pipIdx -r requirements.txt

    }

    "web" {

        Install-CpuTorch $py

        Invoke-VenvPip $py install --default-timeout=2000 --no-cache-dir @pipIdx -r requirements-web.txt

    }

    "api" {

        Invoke-VenvPip $py install @pipIdx -r requirements-api.txt

    }

    "all" {

        Install-CpuTorch $py

        Invoke-VenvPip $py install @pipIdx -r requirements.txt -r requirements-web.txt -r requirements-api.txt

    }

    default {

        Write-Host "安装 CPU 版 PyTorch + 训练依赖..."

        Install-CpuTorch $py

        Invoke-VenvPip $py install @pipIdx -r requirements.txt

    }

}



if (-not (Test-Path "config.yaml")) {

    Copy-Item "config.example.yaml" "config.yaml"

}



Write-Host ""

Write-Host "运行环境诊断..."

& $py doctor.py 2>$null

if ($LASTEXITCODE -ne 0) {

    Write-Host "[提示] Doctor 有未通过项，可运行: .\run.ps1 -Doctor"

    Write-Host "[提示] 国内网络安装失败可试: .\setup.ps1 -Mode web -Mirror cn"

}



Write-Host ""

Write-Host "[OK] Setup ($Mode) 完成。运行: .\start.cmd 或 .\run.ps1 -Web"

