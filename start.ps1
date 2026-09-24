param(
    [switch]$Setup,
    [switch]$NoBrowser,
    [switch]$SkipLmStudio,
    [ValidateSet("web", "api", "both")]
    [string]$Mode = "web",
    [string]$Profile = "dev"
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

$env:PYTHONIOENCODING = "utf-8"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""

function Write-Banner {
    param([string]$Text)
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
}

function Test-LmStudioApi {
    param([string]$PythonExe)
    $code = 'from src.lmstudio_client import check_lm_studio; import sys; sys.exit(0 if check_lm_studio() else 1)'
    & $PythonExe -c $code 2>$null
    return $LASTEXITCODE -eq 0
}

function Start-LmStudioApp {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\LM Studio\LM Studio.exe",
        "$env:LOCALAPPDATA\LM Studio\LM Studio.exe",
        "${env:ProgramFiles}\LM Studio\LM Studio.exe",
        "${env:ProgramFiles(x86)}\LM Studio\LM Studio.exe"
    )
    foreach ($path in $candidates) {
        if (Test-Path $path) {
            Write-Host "[启动] LM Studio -> $path"
            Start-Process $path | Out-Null
            return $true
        }
    }
    Write-Host '[提示] 未找到 LM Studio，请手动打开并开启 Local Server: http://127.0.0.1:1234'
    return $false
}

function Test-PythonPkg {
    param(
        [string]$PythonExe,
        [string]$Pkg
    )
    $code = "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('$Pkg') else 1)"
    & $PythonExe -c $code 2>$null
    return $LASTEXITCODE -eq 0
}

function Ensure-Dependencies {
    param([string]$PythonExe)
    if ($Mode -in @("web", "both")) {
        if (-not (Test-PythonPkg -PythonExe $PythonExe -Pkg "gradio") -or
            -not (Test-PythonPkg -PythonExe $PythonExe -Pkg "torch")) {
            Write-Host "[安装] 缺少 gradio / torch，正在安装 Web 依赖（国内镜像）..."
            & "$Root\setup.ps1" -Mode web -Mirror cn
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
    if ($Mode -in @("api", "both")) {
        if (-not (Test-PythonPkg -PythonExe $PythonExe -Pkg "fastapi")) {
            Write-Host "[安装] 缺少 fastapi，正在安装 API 依赖..."
            & "$Root\setup.ps1" -Mode api
            if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        }
    }
}

Write-Banner "lmstudio-finetune 一键启动"

if ($Setup -or -not (Test-Path ".venv")) {
    Write-Host "[1/4] 初始化环境，首次运行安装 Web 依赖..."
    & "$Root\setup.ps1" -Mode web
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
else {
    Write-Host "[1/4] 虚拟环境已就绪"
}

if (-not (Test-Path "config.yaml")) {
    Copy-Item "config.example.yaml" "config.yaml"
    Write-Host "[2/4] 已生成 config.yaml"
}
else {
    Write-Host "[2/4] 配置文件已就绪"
}

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    Write-Host "[错误] 未找到虚拟环境，请运行: .\start.cmd -Setup"
    exit 1
}

Ensure-Dependencies -PythonExe $py

if ($Profile) {
    $env:LMSTUDIO_PROFILE = $Profile
    Write-Host "[Profile] $Profile"
}

Write-Host "[3/4] 检查 LM Studio API..."
if (-not $SkipLmStudio) {
    if (Test-LmStudioApi -PythonExe $py) {
        Write-Host '       LM Studio 已连接: http://127.0.0.1:1234'
    }
    else {
        Write-Host "       LM Studio 未连接，尝试启动桌面客户端..."
        Start-LmStudioApp | Out-Null
        Write-Host "       请在 LM Studio 中: 加载模型 -> Local Server -> Start"
    }
}
else {
    Write-Host "       已跳过 LM Studio 检查"
}

$webUrl = "http://127.0.0.1:7860"
$apiUrl = "http://127.0.0.1:8000/docs"

Write-Host "[4/4] 启动服务 mode=$Mode ..."

switch ($Mode) {
    "api" {
        if (-not $NoBrowser) {
            Start-Process $apiUrl | Out-Null
        }
        Write-Host "       FastAPI http://127.0.0.1:8000"
        Write-Host "       按 Ctrl+C 停止"
        & $py serve_api.py
    }
    "both" {
        Write-Host "       Gradio  $webUrl"
        Write-Host "       FastAPI http://127.0.0.1:8000"
        Write-Host "       按 Ctrl+C 停止 Web，API 在后台继续运行"
        Start-Process -FilePath $py -ArgumentList "serve_api.py" -WorkingDirectory $Root -WindowStyle Minimized | Out-Null
        if ($NoBrowser) {
            Remove-Item Env:GRADIO_INBROWSER -ErrorAction SilentlyContinue
        }
        else {
            $env:GRADIO_INBROWSER = "1"
        }
        & $py web\app.py
    }
    default {
        Write-Host "       Gradio 控制台 $webUrl"
        Write-Host "       按 Ctrl+C 停止"
        if ($NoBrowser) {
            Remove-Item Env:GRADIO_INBROWSER -ErrorAction SilentlyContinue
        }
        else {
            $env:GRADIO_INBROWSER = "1"
        }
        & $py web\app.py
    }
}

exit $LASTEXITCODE