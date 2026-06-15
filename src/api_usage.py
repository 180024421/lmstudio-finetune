from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config_loader import ROOT, load_config

DEFAULT_LOG = ROOT / "output" / "api_usage.json"


def _usage_cfg(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    return cfg.get("api_usage") or {}


def usage_log_path(cfg: dict[str, Any] | None = None) -> Path:
    ucfg = _usage_cfg(cfg)
    p = ucfg.get("log_path") or "output/api_usage.json"
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def record_usage(
    *,
    operation: str,
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    ucfg = _usage_cfg(cfg)
    if ucfg.get("enabled", True) is False:
        return {}

    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens

    cost_prompt = float(ucfg.get("cost_per_1k_prompt", 0) or 0)
    cost_completion = float(ucfg.get("cost_per_1k_completion", 0) or 0)
    estimated_cost = (prompt_tokens / 1000.0) * cost_prompt + (completion_tokens / 1000.0) * cost_completion

    entry = {
        "at": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": round(estimated_cost, 6),
    }

    path = usage_log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if path.exists():
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            records = []
    records.append(entry)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def record_from_response(response: Any, *, operation: str, model: str, cfg: dict[str, Any] | None = None) -> None:
    usage = getattr(response, "usage", None)
    if not usage:
        return
    record_usage(
        operation=operation,
        model=model,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(usage, "total_tokens", 0) or 0),
        cfg=cfg,
    )


def summarize_usage(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    path = usage_log_path(cfg)
    if not path.exists():
        return {"total_calls": 0, "total_tokens": 0, "estimated_cost": 0.0, "by_operation": {}}
    records: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    by_op: dict[str, dict[str, Any]] = {}
    total_tokens = 0
    total_cost = 0.0
    for r in records:
        op = r.get("operation", "unknown")
        bucket = by_op.setdefault(op, {"calls": 0, "tokens": 0, "cost": 0.0})
        bucket["calls"] += 1
        bucket["tokens"] += int(r.get("total_tokens", 0))
        bucket["cost"] += float(r.get("estimated_cost", 0))
        total_tokens += int(r.get("total_tokens", 0))
        total_cost += float(r.get("estimated_cost", 0))
    return {
        "log_path": str(path),
        "total_calls": len(records),
        "total_tokens": total_tokens,
        "estimated_cost": round(total_cost, 4),
        "by_operation": by_op,
    }


def usage_markdown(cfg: dict[str, Any] | None = None) -> str:
    s = summarize_usage(cfg)
    lines = [
        f"**API 调用统计**（{s.get('log_path', '')}）",
        f"- 总调用: {s['total_calls']}",
        f"- 总 Token: {s['total_tokens']}",
        f"- 估算费用: {s['estimated_cost']}",
        "",
        "| 操作 | 次数 | Token | 费用 |",
        "|------|------|-------|------|",
    ]
    for op, info in (s.get("by_operation") or {}).items():
        lines.append(f"| {op} | {info['calls']} | {info['tokens']} | {info['cost']:.4f} |")
    return "\n".join(lines)
