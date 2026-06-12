# LM Studio 部署微调模型

## 1. 安装 LM Studio

从 https://lmstudio.ai/ 下载安装。

## 2. 训练完成后导入

### 方式 A：GGUF（推荐）

1. 安装 [llama.cpp](https://github.com/ggerganov/llama.cpp)，在 `config.yaml` 设置 `export.llama_cpp_dir`
2. 运行 `python export.py --all` 生成多档量化（如 `f16`、`q4_k_m`）
3. LM Studio → **Load Model** → 选择 `output/model.gguf` 或 `output/model-q4_k_m.gguf`

单档量化：

```powershell
python export.py --gguf --quant q4_k_m
```

### 方式 B：HuggingFace 合并目录

1. `python export.py` → `output/merged/`
2. 部分 LM Studio 版本支持文件夹加载，或使用 llama.cpp 转换

### 方式 C：Ollama

合并后目录含 `Modelfile`：

```bash
ollama create my-finetuned -f output/merged/Modelfile
```

## 3. 开启本地 API

1. 加载微调模型
2. **Local Server** → Start（默认 `http://127.0.0.1:1234`）
3. 测试：`python chat.py "你好"`

## 4. 评测对比

在 LM Studio 中分别记下基座与微调模型的 **model id**，然后：

```powershell
python eval.py --mode compare --base-model <基座id> --finetuned-model <微调id>
```

## 5. 用 LM Studio 生成训练数据

```powershell
python generate_data.py docs/ -o data/generated.jsonl
```

或在 Gradio「造数据」页粘贴文档。

## 6. 与其他项目联调

OpenAI 兼容地址：`http://127.0.0.1:1234/v1`，与 video-promo-pipeline 等项目的 `lm_studio.base_url` 一致。
