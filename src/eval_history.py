"""评测历史对比。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config_loader import ROOT

HISTORY_DIR = ROOT / "output" / "eval_history"


def save_eval_snapshot(report: dict[str, Any], *, name: str | None = None) -> Path:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = f"{name or 'eval'}_{ts}.json"
    path = HISTORY_DIR / fname
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_eval_history(limit: int = 20) -> list[dict[str, Any]]:
    if not HISTORY_DIR.exists():
        return []
    items: list[dict[str, Any]] = []
    for p in sorted(HISTORY_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            items.append(
                {
                    "file": p.name,
                    "mode": data.get("mode", ""),
                    "avg_score": data.get("avg_score", 0),
                    "total": data.get("total", 0),
                }
            )
        except Exception:
            continue
    return items


def eval_history_markdown(limit: int = 15) -> str:
    rows = list_eval_history(limit)
    if not rows:
        return "暂无评测历史。运行评测后会自动归档到 output/eval_history/"
    lines = ["| 文件 | 模式 | 均分 | 样本数 |", "|------|------|------|--------|"]
    for r in rows:
        lines.append(f"| {r['file']} | {r['mode']} | {r['avg_score']:.3f} | {r['total']} |")
    return "\n".join(lines)
