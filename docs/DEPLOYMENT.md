# 部署指南

## LM Studio（默认）

见 [LMSTUDIO.md](LMSTUDIO.md)。

## FastAPI 自托管

```powershell
pip install -r requirements-api.txt
# 转发 LM Studio（轻量）
$env:API_MODE="proxy"
python serve_api.py

# 或本地加载 HF + LoRA（需 GPU + torch）
$env:API_MODE="local"
python serve_api.py
```

- 健康检查：`GET /health`
- OpenAI 兼容：`POST /v1/chat/completions`（`stream: true` 支持 SSE）

## Docker

```powershell
copy config.example.yaml config.yaml
docker compose up web api
```

- Web：http://127.0.0.1:7860
- API：http://127.0.0.1:8000

容器内通过 `host.docker.internal` 访问宿主机 LM Studio。

## Ollama

```powershell
python export.py --all
python deploy_ollama.py --name my-finetuned
ollama run my-finetuned
```

## 收件箱自动训练

1. 将新 JSONL 放入 `data/inbox/`
2. `python watch_inbox.py` 或 Web「收件箱」页
3. `config.yaml` 中 `inbox.auto_train: true` 可自动触发训练

## 回归门禁

```powershell
python run_benchmark.py --judge --set-baseline
python run_benchmark.py --regression   # 低于基线则 exit 1
python regression_cli.py check 0.85
```

训练后自动检查：`train.regression_check: true`

## vLLM（可选，Linux 推荐）

```powershell
pip install -r requirements-vllm.txt
python export.py
python deploy_vllm.py --dry-run   # 查看命令
python deploy_vllm.py
```

## 一键 Pipeline

```powershell
python run_pipeline.py
python cli.py pipeline
```

## 知识蒸馏

```powershell
python distill_data.py docs/faq.md -o data/distilled.jsonl
python judge_filter.py data/distilled.jsonl -o data/train.jsonl
```

## 超参搜索

```powershell
python sweep_train.py --dry-run
python sweep_train.py --grid "{\"lora.r\":[8,16],\"train.learning_rate\":[1e-4,2e-4]}"
```

## DeepSpeed 多卡

`config.yaml` 中设置 `train.deepspeed: deepspeed/ds_config.json`，需安装 `deepspeed`。

## torchrun 多卡（DDP）

```powershell
python train_multi.py --dry-run -n 2
python train_multi.py -n 2
.\run.ps1 -MultiGpu
```

## SFT → DPO 串联

```powershell
python run_alignment.py
python cli.py alignment --skip-sft   # 仅 DPO
```

`alignment.use_sft_adapter_for_dpo: true` 时 DPO 从 SFT adapter 继续训练。

## Windows 计划任务

```powershell
.\scripts\register_scheduled_task.ps1 -Action inbox -IntervalMinutes 5
.\scripts\register_scheduled_task.ps1 -Action pipeline -IntervalMinutes 60 -Profile prod
```

## HuggingFace Hub 上传

```powershell
$env:HF_TOKEN = "hf_xxx"
python upload_hub.py --repo-id username/my-lora
```

## 网页抓取造数

```powershell
python crawl_docs.py https://docs.example.com --max-pages 20 --generate
```

## API 用量

造数/评审/蒸馏等 LM Studio 调用会写入 `output/api_usage.json`：

```powershell
python usage_cli.py --markdown
```

## 配置 Profile

```powershell
python train.py --profile dev
$env:LMSTUDIO_PROFILE = "prod"
```
