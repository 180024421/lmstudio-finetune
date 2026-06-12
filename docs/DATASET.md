# 数据集格式

每行一条 JSON（JSONL）。

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

## 3. RAG 格式（`context` + 问答）

```json
{"context": "参考文档片段", "question": "用户问题", "answer": "基于文档的回答"}
```

## 4. KTO 格式（`kto_train.py`）

```json
{"prompt": "问题", "completion": "回答", "label": true}
```

也可用 DPO 风格自动展开为 chosen/rejected。

## 5. DPO 偏好格式（`dpo_train.py`）

```json
{"prompt": "用户问题", "chosen": "更好的回答", "rejected": "较差的回答"}
```

## 6. 评测可选字段

```json
{"messages": [...], "keywords": ["Redis", "布隆过滤器"]}
```

## 工具

| 命令 | 说明 |
|------|------|
| `python validate_data.py` | 格式校验、重复检测 |
| `python stats_data.py` | 长度/token 估计 |
| `python split_data.py all.jsonl -o data` | 8:1:1 切分 |
| `python convert_data.py faq.md -f faq_md -o data/train.jsonl` | 格式转换 |
| `python generate_data.py docs/ -o data/generated.jsonl` | LM Studio 造数据 |

`config.yaml` 中 `dataset.chat_template` 需与基座模型匹配。

建议至少 **100+** 条高质量样本；领域越窄，所需样本越少。
