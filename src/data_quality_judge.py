from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track

from .config_loader import load_config
from .data_io import load_rows, write_jsonl
from .data_quality import score_row
from .llm_judge import judge_response

console = Console()


def _extract_qa(row: dict[str, Any]) -> tuple[str, str]:
    if "messages" in row:
        user = next((m["content"] for m in row["messages"] if m.get("role") == "user"), "")
        out = next((m["content"] for m in row["messages"] if m.get("role") == "assistant"), "")
        return user, out
    if "instruction" in row:
        q = f"{row.get('instruction', '')}\n{row.get('input', '')}".strip()
        return q, str(row.get("output", ""))
    return str(row.get("prompt", "")), str(row.get("chosen") or row.get("output", ""))


def filter_with_judge(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
    *,
    min_rule: float = 0.5,
    min_judge: float = 0.6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = cfg or load_config()
    kept, rejected = [], []
    for row in track(rows, description="LLM 质检"):
        rs, _ = score_row(row)
        if rs < min_rule:
            rejected.append({**row, "_reject": "rule", "_rule_score": rs})
            continue
        q, a = _extract_qa(row)
        if not a.strip():
            rejected.append({**row, "_reject": "empty"})
            continue
        try:
            js, reason = judge_response(q, a, a, cfg)
        except Exception as e:
            rejected.append({**row, "_reject": str(e)})
            continue
        if js >= min_judge:
            kept.append(row)
        else:
            rejected.append({**row, "_reject": reason, "_judge_score": js})
    return kept, rejected


def filter_file_with_judge(
    input_path: Path,
    output_path: Path,
    rejected_path: Path | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, int]:
    qcfg = (cfg or load_config()).get("data_generation") or {}
    rows = load_rows(input_path)
    kept, rejected = filter_with_judge(
        rows,
        cfg,
        min_rule=float(qcfg.get("min_quality_score", 0.5)),
        min_judge=float(qcfg.get("min_judge_score", 0.6)),
    )
    write_jsonl(output_path, kept)
    if rejected_path:
        write_jsonl(rejected_path, rejected)
    return {"input": len(rows), "kept": len(kept), "rejected": len(rejected)}
