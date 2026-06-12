param(
    [switch]$Setup,
    [switch]$Train,
    [switch]$Export,
    [switch]$Chat,
    [string]$Prompt = "你好"
)

$Root = $PSScriptRoot
Set-Location $Root

if ($Setup) {
    if (-not (Test-Path ".venv")) { python -m venv .venv }
    .\.venv\Scripts\Activate.ps1
    pip install -U pip
    pip install -r requirements.txt
    if (-not (Test-Path "config.yaml")) { Copy-Item config.example.yaml config.yaml }
    Write-Host "Setup done. Edit config.yaml and data/*.jsonl"
    exit 0
}

.\.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING = "utf-8"

if ($Train) { python train.py; exit $LASTEXITCODE }
if ($Export) { python export.py --gguf; exit $LASTEXITCODE }
if ($Chat) { python chat.py $Prompt; exit $LASTEXITCODE }

Write-Host @"
用法:
  .\run.ps1 -Setup          # 创建 venv + 安装依赖
  .\run.ps1 -Train          # QLoRA 训练
  .\run.ps1 -Export         # 合并 LoRA + 尝试 GGUF
  .\run.ps1 -Chat -Prompt "你好"
"@
