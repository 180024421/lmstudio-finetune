from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .data_io import load_rows, write_jsonl

MIN_OUTPUT_LEN = 4
MAX_OUTPUT_LEN = 8000
FILLER_PATTERNS = [
    re.compile(r"作为(一个)?AI", re.I),
    re.compile(r"我无法"),
    re.compile(r"^\.+$"),
]


def score_row(row: dict[str, Any]) -> tuple[float, list[str]]:
    issues: list[str] = []
    score = 1.0

    if "messages" in row:
        out = next((m.get("content", "") for m in row["messages"] if m.get("role") == "assistant"), "")
        inst = next((m.get("content", "") for m in row["messages"] if m.get("role") == "user"), "")
    elif "instruction" in row:
        out = row.get("output", "")
        inst = f"{row.get('instruction', '')} {row.get('input', '')}".strip()
    else:
        return 0.0, ["未知格式"]

    out = str(out).strip()
    inst = str(inst).strip()

    if len(out) < MIN_OUTPUT_LEN:
        score -= 0.5
        issues.append("回答过短")
    if len(out) > MAX_OUTPUT_LEN:
        score -= 0.2
        issues.append("回答过长")
    if not inst:
        score -= 0.4
        issues.append("问题为空")
    if out and inst and out.strip() == inst.strip():
        score -= 0.6
        issues.append("回答与问题相同")
    for pat in FILLER_PATTERNS:
        if pat.search(out):
            score -= 0.3
            issues.append("含敷衍/拒答用语")
            break
    # 回答应比问题稍长（经验规则）
    if inst and out and len(out) < len(inst) * 0.3:
        score -= 0.15
        issues.append("回答相对问题过短")

    return max(0.0, min(1.0, score)), issues


def filter_rows(rows: list[dict[str, Any]], min_score: float = 0.6) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for row in rows:
        s, issues = score_row(row)
        row_copy = dict(row)
        row_copy["_quality_score"] = round(s, 3)
        row_copy["_quality_issues"] = issues
        if s >= min_score:
            kept.append({k: v for k, v in row_copy.items() if not k.startswith("_quality")})
        else:
            rejected.append(row_copy)
    return kept, rejected


def filter_file(
    input_path: Path,
    output_path: Path,
    rejected_path: Path | None = None,
    *,
    min_score: float = 0.6,
) -> dict[str, int]:
    rows = load_rows(input_path)
    kept, rejected = filter_rows(rows, min_score=min_score)
    write_jsonl(output_path, kept)
    if rejected_path:
        write_jsonl(rejected_path, rejected)
    return {"input": len(rows), "kept": len(kept), "rejected": len(rejected)}
