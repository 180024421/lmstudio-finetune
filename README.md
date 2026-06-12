# lmstudio-finetune

在本地用 **自己的数据集** 微调大语言模型（QLoRA），并部署到 **LM Studio** 推理。

```
准备数据 → QLoRA 训练 → 合并导出 → LM Studio 加载 → OpenAI 兼容 API
```

## 环境要求

- Windows 10/11 + NVIDIA GPU（建议 8GB+ 显存，默认 `Qwen2.5-1.5B-Instruct`）
- Python 3.10+
- [LM Studio](https://lmstudio.ai/)（推理与 API）
- 可选：[llama.cpp](https://github.com/ggerganov/llama.cpp)（转 GGUF）

## 快速开始

```powershell
cd lmstudio-finetune
.\run.ps1 -Setup

# 编辑 config.yaml、准备 data/train.jsonl
copy config.example.yaml config.yaml
# 把你的 JSONL 放到 data/train.jsonl

.\run.ps1 -Train
.\run.ps1 -Export

# LM Studio 加载 output/model.gguf 或 merged 目录后：
.\run.ps1 -Chat -Prompt "你好"
```

## 项目结构

| 路径 | 说明 |
|------|------|
| `train.py` | QLoRA 微调 |
| `export.py` | 合并 LoRA → GGUF |
| `chat.py` | 测试 LM Studio API |
| `data/examples/` | 示例数据集 |
| `docs/DATASET.md` | 数据格式说明 |
| `docs/LMSTUDIO.md` | LM Studio 部署指南 |

## 配置要点

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct   # 可换 3B，显存更大用 7B+4bit
dataset:
  train_file: data/train.jsonl
lora:
  r: 16
train:
  num_epochs: 3
quantization:
  load_in_4bit: true    # QLoRA
```

## 流程说明

1. **训练**：`transformers` + `peft` + `trl` 做 4bit QLoRA，适配器保存到 `output/run1/adapter/`
2. **合并**：将 LoRA 权重合并进基座，输出 `output/merged/`
3. **GGUF**：通过 llama.cpp 转换，供 LM Studio 高效加载
4. **推理**：LM Studio Local Server 提供 `/v1/chat/completions`

## 许可

MIT
