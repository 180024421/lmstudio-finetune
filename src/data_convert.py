from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .data_io import write_jsonl


def sharegpt_to_messages(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    mapping = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
    out: list[dict[str, str]] = []
    for turn in conversations:
        frm = mapping.get(turn.get("from", ""), turn.get("from", "user"))
        out.append({"role": frm, "content": turn.get("value", "")})
    return out


def convert_sharegpt(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("ShareGPT JSON 应为数组")
    for item in data:
        conv = item.get("conversations") or item.get("conversation")
        if conv:
            rows.append({"messages": sharegpt_to_messages(conv)})
    return rows


def convert_csv(path: Path, *, instruction_col: str, input_col: str, output_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "instruction": r.get(instruction_col, ""),
                    "input": r.get(input_col, ""),
                    "output": r.get(output_col, ""),
                }
            )
    return rows


def convert_faq_md(path: Path) -> list[dict[str, Any]]:
    """解析 ## 问题 / 答案 或 Q: / A: 风格 Markdown。"""
    rows: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n## ") if b.strip()]
    for block in blocks:
        lines = block.splitlines()
        title = lines[0].strip("# ").strip()
        body = "\n".join(lines[1:]).strip()
        if not body:
            continue
        if body.lower().startswith("a:") or body.startswith("答："):
            ans = body.split("\n", 1)[-1].strip()
        else:
            ans = body
        rows.append({"instruction": title, "input": "", "output": ans})
    return rows


def convert_rag_csv(
    path: Path,
    *,
    context_col: str = "context",
    question_col: str = "question",
    answer_col: str = "answer",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "context": r.get(context_col, ""),
                "question": r.get(question_col, ""),
                "answer": r.get(answer_col, ""),
            })
    return rows


def convert_rag_md(path: Path, chunk_size: int = 1500) -> list[dict[str, Any]]:
    """将 Markdown 按 ## 标题切分为 context，配对 FAQ 式 question/answer 需另附 CSV；此处生成占位问答。"""
    text = path.read_text(encoding="utf-8")
    blocks = [b.strip() for b in text.split("\n## ") if b.strip()]
    rows: list[dict[str, Any]] = []
    for block in blocks:
        lines = block.splitlines()
        title = lines[0].strip("# ").strip()
        body = "\n".join(lines[1:]).strip()[:chunk_size]
        if not body:
            continue
        rows.append({
            "context": body,
            "question": f"请总结「{title}」的要点",
            "answer": body[:500],
        })
    return rows


def convert_file(
    input_path: Path,
    output_path: Path,
    fmt: str,
    **kwargs: Any,
) -> int:
    fmt = fmt.lower()
    if fmt == "sharegpt":
        rows = convert_sharegpt(input_path)
    elif fmt == "csv":
        rows = convert_csv(
            input_path,
            instruction_col=kwargs.get("instruction_col", "instruction"),
            input_col=kwargs.get("input_col", "input"),
            output_col=kwargs.get("output_col", "output"),
        )
    elif fmt == "faq_md":
        rows = convert_faq_md(input_path)
    elif fmt == "rag_csv":
        rows = convert_rag_csv(
            input_path,
            context_col=kwargs.get("context_col", "context"),
            question_col=kwargs.get("question_col", "question"),
            answer_col=kwargs.get("answer_col", "answer"),
        )
    elif fmt == "rag_md":
        rows = convert_rag_md(input_path, chunk_size=int(kwargs.get("chunk_size", 1500)))
    else:
        raise ValueError(f"不支持的格式: {fmt}")
    write_jsonl(output_path, rows)
    return len(rows)
