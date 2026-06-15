# AGENTS.md — lmstudio-finetune

供 Cursor / Codex / Claude 等 AI Agent 使用的项目指南。

## 项目简介

本地 **QLoRA / DPO / KTO / ORPO** 微调大语言模型，导出 **GGUF**，部署到 **LM Studio** 推理。含数据工具链、评测、Gradio 控制台、video-promo 联动。

```
造数据 → 校验/切分 → QLoRA/DPO → 合并导出 → LM Studio → 评测对比
```

## 技术栈

- Python 3.10+、PyTorch、transformers、peft、trl
- Windows 10/11 + NVIDIA GPU（训练）
- LM Studio OpenAI 兼容 API（`http://127.0.0.1:1234/v1`）
- Gradio Web UI、FastAPI 推理代理
- pytest + ruff

## 模块地图

| 模块 | 路径 | 职责 |
|------|------|------|
| 配置 | `src/config_loader.py`, `config_schema.py` | YAML 加载、profile 合并、校验 |
| 数据 | `src/data_*.py`, `dataset.py` | 校验、转换、去重、增强、造数 |
| 训练 | `src/train_lora.py`, `train_dpo.py` 等 | QLoRA/DPO/KTO/ORPO |
| 流水线 | `src/pipeline.py`, `alignment_pipeline.py` | 一键 SFT→导出→评测 |
| 导出 | `src/export_model.py` | LoRA 合并、GGUF 量化 |
| 推理 | `src/lmstudio_client.py`, `api_server.py` | LM Studio 客户端、FastAPI |
| 评测 | `src/eval_runner.py`, `benchmark_runner.py` | 规则/LLM 评审、回归门禁 |
| 运维 | `src/doctor.py`, `vram_estimate.py`, `inbox_watcher.py` | 环境诊断、显存估算 |
| 联动 | `src/promo_integration.py` | video-promo-pipeline bridge |
| Web | `web/app.py` | Gradio 全功能控制台 |

## 开发工作流

```powershell
# 首次
.\run.ps1 -Setup
copy config.example.yaml config.yaml   # setup 可能已自动复制

# 日常
.\run.ps1 -Validate          # 校验数据
python doctor.py --markdown  # 环境诊断（含显存估算）
.\run.ps1 -Train             # QLoRA 训练
.\run.ps1 -ExportAll         # 合并 + GGUF
.\run.ps1 -Web               # Gradio → http://127.0.0.1:7860

# 质量
pytest tests/ -q
ruff check .
ruff format --check .
pre-commit run --all-files   # 若已 install
```

## Agent 行为准则

### 应该做

- 新逻辑放 `src/`，根目录只保留薄 CLI
- 改配置相关代码时同步更新 `config.example.yaml` 和 `config_schema.py`
- 新功能补 `tests/test_*.py`，用 mock 避免 GPU/LM Studio 依赖
- 训练相关改动考虑 `config.dev.yaml` 小样本试跑路径
- LM Studio 调用走 `src/lmstudio_client.py`

### 不应该做

- 不在 CI 中跑真实 GPU 训练或大模型下载
- 不硬编码 `http://127.0.0.1:1234`，用配置 `lm_studio.base_url`
- 不提交 `config.yaml`、`.venv/`、`output/` 到 git
- 不新增与 `cli.py` 重复的入口脚本
- 不修改 `~/.cursor/skills-cursor/`（Cursor 内置目录）

## 环境变量

见 `.env.example`：`HF_TOKEN`、`WANDB_API_KEY`、`LMSTUDIO_PROFILE` 等。

## 文档

- [README.md](README.md) — 功能一览与 CLI 速查
- [docs/DATASET.md](docs/DATASET.md) — 数据集格式
- [docs/LMSTUDIO.md](docs/LMSTUDIO.md) — LM Studio 部署
- [docs/INTEGRATION.md](docs/INTEGRATION.md) — video-promo 联动

## 测试策略

- 单元测试：纯函数、配置合并、数据校验
- Mock 测试：`pipeline`、`lmstudio_client` 用 mock 隔离外部依赖
- 集成测试：`test_promo_integration.py` 等，不依赖 GPU
- `train.py --dry-run` / `train_multi.py --dry-run` 用于本地冒烟
