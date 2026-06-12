# lmstudio-finetune

本地 **QLoRA / DPO** 微调大语言模型，导出 **GGUF**，部署到 **LM Studio** 推理。含完整数据工具链、评测、Gradio 控制台。

```
造数据 → 校验/切分 → QLoRA/DPO → 合并导出 → LM Studio → 评测对比
```

## 功能一览

| 模块 | 能力 |
|------|------|
| **数据** | 校验、统计、切分、ShareGPT/CSV/FAQ 转换、LM Studio 自动造数 |
| **训练** | 4bit QLoRA、断点续训、早停、最佳 checkpoint、TensorBoard/WandB |
| **DPO** | 偏好对齐（chosen/rejected） |
| **导出** | LoRA 合并、多档 GGUF 量化、Model Card、Ollama Modelfile |
| **评测** | LM Studio / 本地 HF 评测、基座 vs 微调对比 |
| **LoRA 管理** | 多适配器注册、导出、列表 |
| **Web** | Gradio 全功能控制台（含实验/校对/导入） |
| **模板** | Qwen / Llama3 / ChatML / Mistral / Alpaca |
| **进阶** | LLM 评审、实验对比、回归门禁、Benchmark、去重/语义去重、PII、video-promo 导入 |
| **对齐** | DPO、KTO |
| **加速** | Unsloth（可选） |
| **部署** | FastAPI、Docker、Ollama、流式对话 |
| **运维** | 收件箱监视、Prompt A/B、多 LoRA 合并、RAG 数据、回归门禁、CI |
| **对齐+** | ORPO、知识蒸馏造数、LLM 双重质检 |
| **工程** | 统一 CLI、一键 Pipeline、超参 Sweep、Axolotl 导出、DeepSpeed、配置校验 |
| **推理+** | vLLM 启动助手、训练曲线、train --dry-run |
| **数据+** | 多文件合并、释义增强、数据集 diff、打乱抽样、版本登记 |
| **报告** | 训练报告 Markdown/HTML、评测 HTML 报告、环境诊断 doctor |
| **自动化** | 多卡 torchrun、SFT→DPO 串联、Windows 计划任务、收件箱 auto_pipeline |
| **协作** | HF Hub 上传、网页抓取造数、API 用量统计、配置 Profile (dev/prod) |
| **可视化** | Gradio 训练曲线 LinePlot |
| **联动** | video-promo 深度闭环、LoRA A/B HTML 报告、bridge 自动回写 |

## 环境要求

- Windows 10/11 + NVIDIA GPU（训练建议 8GB+ 显存）
- Python 3.10+
- [LM Studio](https://lmstudio.ai/)（推理、造数据、评测）
- 可选：[llama.cpp](https://github.com/ggerganov/llama.cpp)（转 GGUF）

## 快速开始

```powershell
cd lmstudio-finetune

# 安装（CPU 版 torch 体积小；有 GPU 用 -SetupMode gpu）
.\run.ps1 -Setup
copy config.example.yaml config.yaml   # 若 setup 未自动复制

.\run.ps1 -Validate
.\run.ps1 -Train
.\run.ps1 -ExportAll

# Gradio 控制台
.\run.ps1 -Web
# → http://127.0.0.1:7860

# LM Studio 加载 output/model.gguf 后
.\run.ps1 -Chat -Prompt "你好"
.\run.ps1 -Eval
```

## 项目结构

```
lmstudio-finetune/
  train.py / dpo_train.py / export.py / eval.py / chat.py
  validate_data.py / split_data.py / stats_data.py / convert_data.py
  generate_data.py / lora_cli.py
  web/app.py              # Gradio 控制台
  src/                    # 核心库
  data/examples/          # 示例 SFT + DPO 数据
  docs/                   # 数据集、LM Studio、项目联动
  setup.ps1               # 环境安装（处理代理/缓存）
  requirements*.txt       # cpu / gpu / web 拆分
```

## 配置要点

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct
dataset:
  chat_template: qwen
  train_file: data/examples/train.jsonl
train:
  resume_from_checkpoint: false
  load_best_model_at_end: true
  early_stopping: true
export:
  gguf_quants: [f16, q4_k_m]
  llama_cpp_dir: "D:/llama.cpp"
lm_studio:
  base_url: http://127.0.0.1:1234/v1
```

## CLI 速查

```powershell
.\run.ps1 -Setup -SetupMode gpu
python split_data.py data/all.jsonl -o data
python convert_data.py faq.md -f faq_md -o data/train.jsonl
python generate_data.py docs/ -o data/generated.jsonl
python train.py --resume
python export.py --all
python export.py --lora run1 --gguf --quant q4_k_m
python dpo_train.py
python eval.py --mode compare --base-model base --finetuned-model ft
python lora_cli.py list
python eval.py --mode judge
python dedup_data.py data/train.jsonl -o data/train_deduped.jsonl
python filter_data.py data/generated.jsonl -o data/train.jsonl --pii
python import_video_promo.py ../video-promo-pipeline/output/jobs
python download_model.py
python experiment_cli.py compare
python review_cli.py list
python kto_train.py
python train_unsloth.py
python lora_merge.py run1 run2
python semantic_dedup.py data/train.jsonl -o data/train_deduped.jsonl
python run_benchmark.py --judge --set-baseline --regression
python serve_api.py
python watch_inbox.py
python doctor.py --markdown
python merge_data.py data/a.jsonl data/b.jsonl -o data/train.jsonl
python augment_data.py data/train.jsonl -o data/aug.jsonl -n 2 --max-rows 50
python diff_data.py data/old.jsonl data/new.jsonl
python shuffle_data.py data/train.jsonl -o data/shuffled.jsonl -f 0.8
python version_cli.py register data/train.jsonl --note "v1"
python train_report.py -o output/train_report.md
python eval.py --html
python train.py --profile dev
python train_multi.py --dry-run -n 2
python run_alignment.py
python crawl_docs.py https://example.com/docs --generate
python upload_hub.py --repo-id username/my-lora
python usage_cli.py --markdown
.\scripts\register_scheduled_task.ps1 -Action inbox -IntervalMinutes 5
.\run.ps1 -Promo
python lora_ab_report.py --tag promo --file data/promo/promo_eval.jsonl
python lora_cli.py set-meta run1 --lm-studio-model my-promo-gguf
docker compose up web api
```

### 配置 Profile

```powershell
# 开发：小样本快速试跑
python train.py --profile dev

# 生产：长训 + 回归门禁 + 收件箱自动 Pipeline
$env:LMSTUDIO_PROFILE = "prod"
.\run.ps1 -Train
```

`config.dev.yaml` / `config.prod.yaml` 会深度合并到 `config.yaml` 之上。

## 文档

- [数据集格式](docs/DATASET.md)
- [LM Studio 部署](docs/LMSTUDIO.md)
- [与其他项目联动](docs/INTEGRATION.md)

## 许可

MIT
