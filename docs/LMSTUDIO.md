# LM Studio 部署微调模型

## 1. 安装 LM Studio

从 https://lmstudio.ai/ 下载安装。

## 2. 训练完成后导入

### 方式 A：GGUF（推荐）

1. 安装 [llama.cpp](https://github.com/ggerganov/llama.cpp)，在 `config.yaml` 设置 `export.llama_cpp_dir`
2. 运行 `python export.py --gguf` 生成 `output/model.gguf`
3. LM Studio → **Load Model** → 选择 `.gguf` 文件

### 方式 B：HuggingFace 合并目录

1. 运行 `python export.py` 得到 `output/merged/`
2. LM Studio 部分版本支持从文件夹加载，或使用社区转换工具转 GGUF

## 3. 开启本地 API

1. 加载你的微调模型
2. 左侧 **Local Server** → Start Server（默认 `http://127.0.0.1:1234`）
3. 测试：`python chat.py "你好"`

## 4. 与其他项目联调

OpenAI 兼容地址：`http://127.0.0.1:1234/v1`，与 video-promo-pipeline 等项目的 `lm_studio.base_url` 一致。
