from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track

from .config_loader import load_config
from .data_io import load_rows, write_jsonl
from .lmstudio_client import check_lm_studio, completion, openai_client

console = Console()

AUGMENT_PROMPT = """你是数据增强专家。对以下问答样本生成 {n} 个语义等价但表述不同的变体。
要求：
1. 保持事实准确，不改变原意
2. 问题与回答都要改写
3. 只输出 JSON 数组，无 markdown：
[{{"instruction":"新问题","input":"","output":"新回答"}}]

原问题：{question}
原回答：{answer}"""


def _parse_augment_json(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return []
    try:
        data = json.loads(m.group())
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict) and x.get("output")]
    except json.JSONDecodeError:
        pass
    return []


def _row_qa(row: dict[str, Any]) -> tuple[str, str]:
    if "messages" in row:
        msgs = row["messages"]
        q = next((m["content"] for m in msgs if m.get("role") == "user"), "")
        a = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
        return q, a
    if "instruction" in row:
        inst = row.get("instruction", "")
        inp = row.get("input", "")
        q = f"{inst}\n{inp}".strip() if inp else inst
        return q, row.get("output", "")
    raise ValueError("不支持的样本格式")


def augment_rows(
    rows: list[dict[str, Any]],
    *,
    cfg: dict[str, Any] | None = None,
    variants_per_row: int = 2,
    max_rows: int = 0,
) -> list[dict[str, Any]]:
    cfg = cfg or load_config()
    if not check_lm_studio():
        raise RuntimeError("LM Studio 未连接，请先启动 Local Server")

    _, model = openai_client(cfg)

    augmented: list[dict[str, Any]] = []
    source = rows[:max_rows] if max_rows else rows

    for row in track(source, description="数据增强"):
        augmented.append(row)
        try:
            q, a = _row_qa(row)
        except ValueError:
            continue
        if not q.strip() or not a.strip():
            continue
        prompt = AUGMENT_PROMPT.format(n=variants_per_row, question=q, answer=a)
        try:
            resp = completion(
                [{"role": "user", "content": prompt}],
                cfg,
                model=model,
                temperature=0.8,
                max_tokens=2048,
                operation="augment",
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            console.print(f"[yellow]增强失败: {e}[/yellow]")
            continue
        for item in _parse_augment_json(raw):
            if "messages" in row:
                augmented.append(
                    {
                        "messages": [
                            {"role": "user", "content": item.get("instruction", q)},
                            {"role": "assistant", "content": item.get("output", "")},
                        ]
                    }
                )
            else:
                augmented.append(
                    {
                        "instruction": item.get("instruction", q),
                        "input": item.get("input", ""),
                        "output": item.get("output", ""),
                    }
                )
    return augmented


def augment_file(
    path: Path,
    output: Path,
    *,
    cfg: dict[str, Any] | None = None,
    variants_per_row: int = 2,
    max_rows: int = 0,
) -> dict[str, Any]:
    rows = load_rows(path)
    out_rows = augment_rows(rows, cfg=cfg, variants_per_row=variants_per_row, max_rows=max_rows)
    write_jsonl(output, out_rows)
    return {
        "input": str(path),
        "output": str(output),
        "source_rows": len(rows),
        "output_rows": len(out_rows),
        "added": len(out_rows) - len(rows[:max_rows] if max_rows else rows),
    }
