"""数据血缘：记录训练样本来源。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_loader import ROOT
from .data_io import iter_jsonl, write_jsonl

LINEAGE_PATH = ROOT / "data" / "lineage.jsonl"


def record_lineage(
    *,
    source: str,
    output_file: str | Path,
    count: int,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "output_file": str(output_file),
        "count": count,
        "meta": meta or {},
    }
    LINEAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LINEAGE_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def tag_rows_with_lineage(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        tagged = dict(row)
        tagged["_lineage"] = source
        out.append(tagged)
    return out


def lineage_summary(limit: int = 30) -> list[dict[str, Any]]:
    if not LINEAGE_PATH.exists():
        return []
    rows: list[dict[str, Any]] = []
    for _, row in iter_jsonl(LINEAGE_PATH):
        rows.append(row)
    return rows[-limit:]


def lineage_markdown(limit: int = 20) -> str:
    rows = lineage_summary(limit)
    if not rows:
        return "暂无血缘记录。数据合并/造数/增强时会自动登记。"
    lines = ["| 时间 | 来源 | 输出 | 条数 |", "|------|------|------|------|"]
    for r in reversed(rows):
        lines.append(f"| {r['ts'][:19]} | {r['source']} | {r['output_file']} | {r['count']} |")
    return "\n".join(lines)
