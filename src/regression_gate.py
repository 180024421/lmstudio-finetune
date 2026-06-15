from __future__ import annotations

import json
from typing import Any

from .config_loader import ROOT
from .experiment_registry import list_experiments
from .notify import notify

GATE_PATH = ROOT / "output" / "regression_baseline.json"


def load_baseline() -> dict[str, Any] | None:
    if not GATE_PATH.exists():
        return None
    return json.loads(GATE_PATH.read_text(encoding="utf-8"))


def save_baseline(score: float, *, name: str = "", mode: str = "") -> dict[str, Any]:
    data = {"score": score, "name": name, "mode": mode}
    GATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def check_regression(
    current_score: float,
    *,
    min_delta: float = -0.05,
    name: str = "",
) -> dict[str, Any]:
    """若当前分低于基线 + min_delta 则判定回归失败。min_delta 为负表示允许下降幅度。"""
    baseline = load_baseline()
    if baseline is None:
        return {"ok": True, "reason": "无基线，跳过回归检查", "current": current_score}

    base_score = float(baseline.get("score", 0))
    delta = current_score - base_score
    ok = delta >= min_delta
    result = {
        "ok": ok,
        "baseline": base_score,
        "current": current_score,
        "delta": round(delta, 4),
        "min_delta": min_delta,
        "name": name or baseline.get("name", ""),
    }
    if not ok:
        result["reason"] = f"评测回归：{current_score:.3f} < 基线 {base_score:.3f} + {min_delta}"
        notify("lmstudio-finetune 回归告警", result["reason"])
    else:
        result["reason"] = "通过回归检查"
    return result


def auto_baseline_from_experiments() -> float | None:
    exps = [e for e in list_experiments(100) if e.get("eval_score") is not None]
    if not exps:
        return None
    best = max(exps, key=lambda x: float(x["eval_score"]))
    return float(best["eval_score"])
