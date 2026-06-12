from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def _messages_to_text(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}")
    return "\n".join(parts)


def _row_to_text(row: dict[str, Any]) -> dict[str, str]:
    if "messages" in row:
        return {"text": _messages_to_text(row["messages"])}
    if "instruction" in row:
        inst = row.get("instruction", "")
        inp = row.get("input", "")
        out = row.get("output", "")
        user = f"{inst}\n{inp}".strip() if inp else inst
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": out},
        ]
        return {"text": _messages_to_text(messages)}
    raise ValueError(f"无法解析样本，需要 messages 或 instruction 字段: {list(row.keys())}")


def load_jsonl(path: Path, max_samples: int = 0) -> Dataset:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples and len(rows) >= max_samples:
                break
    return Dataset.from_list([_row_to_text(r) for r in rows])
