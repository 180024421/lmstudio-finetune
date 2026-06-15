param(
    [switch]$Setup,
    [ValidateSet("cpu", "gpu", "full", "web")]
    [string]$SetupMode = "full",
    [switch]$Train,
    [switch]$Resume,
    [switch]$Export,
    [switch]$ExportAll,
    [switch]$DPO,
    [switch]$Chat,
    [switch]$Web,
    [switch]$Validate,
    [switch]$Stats,
    [switch]$Eval,
    [switch]$Judge,
    [switch]$Dedup,
    [switch]$Filter,
    [switch]$KTO,
    [switch]$Unsloth,
    [switch]$API,
    [switch]$Benchmark,
    [switch]$WatchInbox,
    [switch]$Pipeline,
    [switch]$DryRun,
    [switch]$Doctor,
    [switch]$MultiGpu,
    [switch]$Alignment,
    [switch]$Promo,
    [switch]$LoraAb,
    [switch]$Stream,
    [string]$Profile = "",
    [string]$Prompt = "你好"
)

$Root = $PSScriptRoot
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
if ($Profile) { $env:LMSTUDIO_PROFILE = $Profile }
$profileArg = @()
if ($Profile) { $profileArg = @("--profile", $Profile) }

if ($Setup) {
    & "$Root\setup.ps1" -Mode $SetupMode
    exit $LASTEXITCODE
}

if (-not (Test-Path ".venv")) {
    Write-Host "请先运行: .\run.ps1 -Setup"
    exit 1
}

$py = ".\.venv\Scripts\python.exe"

if ($Validate) { & $py validate_data.py; exit $LASTEXITCODE }
if ($Doctor)    { & $py doctor.py --markdown; exit $LASTEXITCODE }
if ($Stats)    { & $py stats_data.py; exit $LASTEXITCODE }
if ($Train) {
    if ($DryRun) { & $py train.py --dry-run @profileArg; exit $LASTEXITCODE }
    if ($Resume) { & $py train.py --resume @profileArg } else { & $py train.py @profileArg }
    exit $LASTEXITCODE
}
if ($MultiGpu) {
    if ($DryRun) { & $py train_multi.py --dry-run @profileArg; exit $LASTEXITCODE }
    & $py train_multi.py @profileArg $(if ($Resume) { "--resume" })
    exit $LASTEXITCODE
}
if ($Pipeline) { & $py run_pipeline.py @profileArg; exit $LASTEXITCODE }
if ($Alignment) { & $py run_alignment.py @profileArg; exit $LASTEXITCODE }
if ($Promo)     { & $py run_promo_pipeline.py @profileArg; exit $LASTEXITCODE }
if ($LoraAb)    { & $py lora_ab_report.py --tag promo @profileArg; exit $LASTEXITCODE }
if ($DPO)      { & $py dpo_train.py; exit $LASTEXITCODE }
if ($KTO)      { & $py kto_train.py; exit $LASTEXITCODE }
if ($Unsloth)  { & $py train_unsloth.py; exit $LASTEXITCODE }
if ($API)      { & $py serve_api.py; exit $LASTEXITCODE }
if ($Benchmark){ & $py run_benchmark.py --regression; exit $LASTEXITCODE }
if ($WatchInbox){ & $py watch_inbox.py; exit $LASTEXITCODE }
if ($ExportAll){ & $py export.py --all; exit $LASTEXITCODE }
if ($Export)   { & $py export.py --gguf; exit $LASTEXITCODE }
if ($Eval)     { & $py eval.py; exit $LASTEXITCODE }
if ($Judge)    { & $py eval.py --mode judge; exit $LASTEXITCODE }
if ($Dedup)    { & $py dedup_data.py data/examples/train.jsonl -o data/train_deduped.jsonl; exit $LASTEXITCODE }
if ($Filter)   { & $py filter_data.py data/generated.jsonl -o data/train_filtered.jsonl; exit $LASTEXITCODE }
if ($Chat) {
    if ($Stream) { & $py chat.py $Prompt --stream } else { & $py chat.py $Prompt }
    exit $LASTEXITCODE
}
if ($Web)      { & $py web\app.py; exit $LASTEXITCODE }

Write-Host @"
lmstudio-finetune 用法:
  .\run.ps1 -Setup [-SetupMode cpu|gpu|full|web]
  .\run.ps1 -Validate          # 校验数据
  .\run.ps1 -Doctor           # 环境诊断
  .\run.ps1 -Stats             # 数据统计
  .\run.ps1 -Train [-Resume] [-Profile dev]   # QLoRA 训练
  .\run.ps1 -MultiGpu [-DryRun]               # 多卡 torchrun
  .\run.ps1 -Alignment                        # SFT→DPO 串联
  .\run.ps1 -Promo                            # video-promo 深度联动
  .\run.ps1 -LoraAb                           # LoRA A/B HTML 报告
  .\run.ps1 -DPO               # DPO 偏好训练
  .\run.ps1 -Export            # 合并 + GGUF
  .\run.ps1 -ExportAll         # 合并 + 多量化 + Modelfile
  .\run.ps1 -Eval              # 规则评测
  .\run.ps1 -Judge             # LLM 评审评测
  .\run.ps1 -Dedup / -Filter   # 去重 / 质检
  .\run.ps1 -Chat -Prompt "你好"
  .\run.ps1 -Web               # Gradio 控制台 http://127.0.0.1:7860
  .\start.cmd                  # 一键启动（自动检查环境 + 打开浏览器）

其他 CLI:
  python download_model.py
  python import_video_promo.py ../video-promo-pipeline/output/jobs
  python experiment_cli.py compare
  python review_cli.py list
  python split_data.py data/all.jsonl -o data
  python convert_data.py faq.md -f faq_md -o data/train.jsonl
  python generate_data.py docs/*.md -o data/generated.jsonl
  python lora_cli.py list
  python kto_train.py / train_unsloth.py
  python lora_merge.py run1 run2
  python semantic_dedup.py data/train.jsonl -o data/train_deduped.jsonl
  python run_benchmark.py --judge --set-baseline
  python regression_cli.py check 0.85
  python prompt_ab.py --questions "什么是Redis"
  python watch_inbox.py --once
  python merge_data.py a.jsonl b.jsonl -o data/train.jsonl
  python augment_data.py data/train.jsonl -o data/aug.jsonl -n 2
  python diff_data.py data/v1.jsonl data/v2.jsonl
  python shuffle_data.py data/train.jsonl -o data/shuffled.jsonl -n 1000
  python version_cli.py list
  python train_report.py -o output/train_report.md
  python train_multi.py --dry-run -n 2
  python run_alignment.py
  python crawl_docs.py https://example.com --generate
  python upload_hub.py --repo-id user/model
  python usage_cli.py --markdown
  .\scripts\register_scheduled_task.ps1 -Action pipeline -IntervalMinutes 30
  python serve_api.py
  docker compose up web api
"@
