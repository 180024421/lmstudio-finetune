from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config_loader import load_config
from .lmstudio_client import chat_with_system, check_lm_studio

console = Console()


def run_prompt_ab(
    prompts: list[str],
    test_questions: list[str],
    *,
    system_prompts: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    cfg = cfg or load_config()
    if not check_lm_studio(cfg):
        raise RuntimeError("LM Studio 未连接")

    systems = system_prompts or [""]
    results: list[dict[str, Any]] = []

    for si, system in enumerate(systems):
        for qi, question in enumerate(test_questions):
            for pi, prefix in enumerate(prompts):
                user = f"{prefix}\n{question}".strip() if prefix else question
                try:
                    answer = chat_with_system(user, system, cfg)
                    score = min(1.0, len(answer.strip()) / max(len(question), 1) / 10)
                except Exception as e:
                    answer, score = "", 0.0
                    err = str(e)
                else:
                    err = ""
                results.append({
                    "system_idx": si,
                    "system": system[:60],
                    "prompt_idx": pi,
                    "prompt_prefix": prefix[:40],
                    "question": question[:80],
                    "answer": answer[:200],
                    "heuristic_score": round(score, 3),
                    "error": err,
                })
    return results


def summarize_ab(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, list[float]] = {}
    for r in results:
        key = f"s{r['system_idx']}_p{r['prompt_idx']}"
        by_key.setdefault(key, []).append(float(r.get("heuristic_score", 0)))
    summary = {k: round(sum(v) / len(v), 3) for k, v in by_key.items()}
    best = max(summary, key=lambda k: summary[k]) if summary else ""
    return {"summary": summary, "best": best}


def print_ab_summary(results: list[dict[str, Any]]) -> None:
    s = summarize_ab(results)
    table = Table(title="Prompt A/B 汇总")
    table.add_column("组合")
    table.add_column("启发分")
    for k, v in sorted(s["summary"].items(), key=lambda x: -x[1]):
        table.add_row(k, f"{v:.3f}")
    console.print(table)
    console.print(f"推荐: {s['best']}")


def save_ab_report(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"results": results, **summarize_ab(results)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
