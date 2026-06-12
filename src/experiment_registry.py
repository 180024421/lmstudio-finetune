from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_loader import ROOT

REGISTRY = ROOT / "output" / "experiments.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def register_experiment(
    *,
    name: str,
    cfg: dict[str, Any],
    metrics_path: Path | None = None,
    eval_score: float | None = None,
    eval_mode: str = "",
    notes: str = "",
) -> dict[str, Any]:
    entry = {
        "id": f"{name}_{_now()[:19].replace(':', '')}",
        "name": name,
        "created_at": _now(),
        "config_hash": _config_hash(cfg),
        "base_model": cfg.get("base_model"),
        "output_dir": cfg.get("output_dir"),
        "dataset": (cfg.get("dataset") or {}).get("train_file"),
        "lora": (cfg.get("lora") or {}).get("name"),
        "train_epochs": (cfg.get("train") or {}).get("num_epochs"),
        "metrics_path": str(metrics_path) if metrics_path else "",
        "eval_score": eval_score,
        "eval_mode": eval_mode,
        "notes": notes,
    }
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def list_experiments(limit: int = 50) -> list[dict[str, Any]]:
    if not REGISTRY.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows[-limit:]


def load_metrics_tail(metrics_path: str, n: int = 20) -> list[dict[str, Any]]:
    p = Path(metrics_path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()
    out: list[dict[str, Any]] = []
    for line in lines[-n:]:
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def compare_experiments(names: list[str] | None = None) -> list[dict[str, Any]]:
    exps = list_experiments(200)
    if names:
        exps = [e for e in exps if e.get("name") in names]
    # 每个 name 取最新一条
    latest: dict[str, dict[str, Any]] = {}
    for e in exps:
        latest[e.get("name", e.get("id", ""))] = e
    rows = list(latest.values())
    rows.sort(key=lambda x: x.get("eval_score") or 0, reverse=True)
    return rows


def experiments_markdown() -> str:
    rows = compare_experiments()
    if not rows:
        return "暂无实验记录。训练完成后自动写入。"
    lines = ["# 实验对比\n", "| 名称 | 基座 | 评测分 | Epoch | 时间 |", "|------|------|--------|-------|------|"]
    for e in rows:
        lines.append(
            f"| {e.get('name')} | {e.get('base_model', '')} | "
            f"{e.get('eval_score', '-')} | {e.get('train_epochs', '-')} | {e.get('created_at', '')[:19]} |"
        )
    return "\n".join(lines)
