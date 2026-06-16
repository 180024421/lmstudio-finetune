"""从评测低分样本回流训练集（主动学习 / 难例采样）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data_io import iter_jsonl, load_rows, write_jsonl
from .data_lineage import record_lineage, tag_rows_with_lineage


def sample_hard_examples(
    eval_report_path: Path,
    *,
    threshold: float = 0.6,
    out_path: Path,
    max_n: int = 50,
) -> dict[str, Any]:
    if not eval_report_path.exists():
        raise FileNotFoundError(f"评测报告不存在: {eval_report_path}")

    data = json.loads(eval_report_path.read_text(encoding="utf-8"))
    samples = data.get("samples") or []
    hard: list[dict[str, Any]] = []
    for s in samples:
        score = float(s.get("score", 0))
        if score >= threshold:
            continue
        prompt = s.get("prompt", "")
        expected = s.get("expected", "")
        if not prompt:
            continue
        row: dict[str, Any] = {
            "instruction": prompt,
            "input": "",
            "output": expected or s.get("actual", ""),
            "_hard_score": score,
        }
        hard.append(row)
        if len(hard) >= max_n:
            break

    tagged = tag_rows_with_lineage(hard, f"hard_examples<{eval_report_path.name}>")
    write_jsonl(out_path, tagged)
    record_lineage(source="hard_example_sampler", output_file=out_path, count=len(tagged), meta={"threshold": threshold})
    return {"count": len(tagged), "threshold": threshold, "out": str(out_path)}
