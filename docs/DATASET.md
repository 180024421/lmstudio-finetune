# 数据集格式

每行一条 JSON（JSONL），支持两种格式：

## 1. 多轮对话（推荐）

```json
{"messages": [
  {"role": "user", "content": "问题"},
  {"role": "assistant", "content": "回答"}
]}
```

## 2. Alpaca 指令格式

```json
{"instruction": "任务说明", "input": "可选输入", "output": "期望输出"}
```

将训练集放到 `data/train.jsonl`，验证集放到 `data/eval.jsonl`，并在 `config.yaml` 中修改路径。

建议至少 **100+** 条高质量样本；领域越窄，所需样本越少。
