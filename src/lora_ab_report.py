from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config_loader import ROOT, load_config
from .eval_runner import EvalReport, run_eval_judge, run_eval_lmstudio
from .lora_manager import list_adapters, resolve_adapter

console = Console()


def _adapter_meta(name: str) -> dict[str, Any]:
    for a in list_adapters():
        if a.get("name") == name:
            return a
    return {}


def compare_loras(
    lora_names: list[str],
    eval_path: Path,
    cfg: dict[str, Any] | None = None,
    *,
    max_samples: int = 20,
    mode: str = "judge",
    base_model_name: str | None = None,
) -> dict[str, Any]:
    """在固定评测集上对比多个 LoRA（通过 LM Studio 模型 ID）。"""
    cfg = cfg or load_config()
    eval_path = eval_path if eval_path.is_absolute() else ROOT / eval_path
    if not eval_path.exists():
        raise FileNotFoundError(f"评测集不存在: {eval_path}")

    reports: dict[str, EvalReport] = {}
    ranking: list[dict[str, Any]] = []

    if base_model_name:
        lora_names = ["__base__"] + list(lora_names)

    for name in lora_names:
        if name == "__base__":
            model_id = base_model_name or ""
            label = "base"
        else:
            meta = _adapter_meta(name)
            try:
                resolve_adapter(name)
            except KeyError as e:
                raise KeyError(f"LoRA 未注册: {name}") from e
            model_id = meta.get("lm_studio_model") or name
            label = name

        console.print(f"[cyan]评测 LoRA[/cyan] {label} → LM Studio model={model_id or '(当前加载)'}")
        if mode == "judge":
            report = run_eval_judge(eval_path, cfg, max_samples=max_samples, model_override=model_id or None)
        else:
            report = run_eval_lmstudio(eval_path, cfg, max_samples=max_samples, model_override=model_id or None)

        reports[label] = report
        ranking.append(
            {
                "name": label,
                "avg_score": round(report.avg_score, 4),
                "total": report.total,
                "mode": report.mode,
                "lm_studio_model": model_id,
                "adapter_path": _adapter_meta(name).get("path", "") if label != "base" else "",
            }
        )

    ranking.sort(key=lambda x: x["avg_score"], reverse=True)
    winner = ranking[0]["name"] if ranking else ""

    return {
        "eval_file": str(eval_path),
        "mode": mode,
        "max_samples": max_samples,
        "winner": winner,
        "ranking": ranking,
        "reports": {k: v.to_dict() for k, v in reports.items()},
        "delta_vs_runner_up": (
            round(ranking[0]["avg_score"] - ranking[1]["avg_score"], 4) if len(ranking) >= 2 else 0.0
        ),
    }


def save_lora_ab_html(result: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_html = []
    for item in result.get("ranking", []):
        medal = "★" if item["name"] == result.get("winner") else ""
        rows_html.append(
            f"<tr><td>{medal} {_esc(item['name'])}</td>"
            f"<td>{item['avg_score']:.3f}</td>"
            f"<td>{item['total']}</td>"
            f"<td>{_esc(item.get('lm_studio_model', ''))}</td></tr>"
        )

    samples_html = []
    winner = result.get("winner", "")
    winner_report = (result.get("reports") or {}).get(winner, {})
    for s in (winner_report.get("samples") or [])[:12]:
        samples_html.append(
            f"<tr><td>{s.get('score', 0):.2f}</td>"
            f"<td>{_esc(str(s.get('prompt', ''))[:100])}</td>"
            f"<td>{_esc(str(s.get('notes', ''))[:80])}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LoRA A/B 报告</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;line-height:1.5}}
table{{border-collapse:collapse;width:100%;margin:1em 0}}
th,td{{border:1px solid #ddd;padding:8px;text-align:left}}
th{{background:#f5f5f5}}
.winner{{color:#0a0;font-weight:bold}}
</style></head><body>
<h1>LoRA A/B 自动对比报告</h1>
<p>评测集: <code>{_esc(result.get("eval_file", ""))}</code></p>
<p>模式: {result.get("mode")} | 样本上限: {result.get("max_samples")}</p>
<p class="winner">推荐: {_esc(winner)}（均分 {ranking_score(result):.3f}，领先第二名 {result.get("delta_vs_runner_up", 0):+.3f}）</p>
<h2>排名</h2>
<table><tr><th>LoRA</th><th>均分</th><th>条数</th><th>LM Studio 模型</th></tr>
{"".join(rows_html)}
</table>
<h2>胜出模型样例</h2>
<table><tr><th>分</th><th>问题</th><th>说明</th></tr>
{"".join(samples_html)}
</table>
<p><small>生成于 lmstudio-finetune lora_ab_report</small></p>
</body></html>"""
    path.write_text(html, encoding="utf-8")
    return path


def ranking_score(result: dict[str, Any]) -> float:
    for item in result.get("ranking", []):
        if item.get("name") == result.get("winner"):
            return float(item.get("avg_score", 0))
    return 0.0


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def print_ab_table(result: dict[str, Any]) -> None:
    table = Table(title="LoRA A/B 对比")
    table.add_column("LoRA")
    table.add_column("均分", justify="right")
    table.add_column("LM Studio 模型")
    for item in result.get("ranking", []):
        mark = "*" if item["name"] == result.get("winner") else ""
        table.add_row(f"{mark}{item['name']}", f"{item['avg_score']:.3f}", item.get("lm_studio_model", ""))
    console.print(table)
    console.print(f"[bold]推荐: {result.get('winner')}[/bold]")
