from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import track

from .config_loader import ROOT, load_config
from .lmstudio_client import completion, openai_client
from .data_io import write_jsonl
from .llm_judge import judge_response

console = Console()

DISTILL_PROMPT = """你是教师模型。根据「知识点/文档」生成高质量训练样本（学生模型将学习这些样本）。

生成 {n} 条，覆盖不同问法。只输出 JSON 数组：
[{{"instruction":"问题","input":"","output":"详细回答"}}]

知识点：
{topic}
"""


def distill_from_topics(
    topics: list[str],
    cfg: dict[str, Any] | None = None,
    *,
    pairs_per_topic: int = 3,
    min_judge_score: float = 0.6,
) -> list[dict[str, Any]]:
    cfg = cfg or load_config()
    dcfg = cfg.get("distillation") or {}
    lcfg = cfg.get("lm_studio") or {}
    _, default_model = openai_client(cfg)
    teacher_model = dcfg.get("teacher_model") or lcfg.get("model") or default_model
    pairs_per_topic = int(dcfg.get("pairs_per_topic", pairs_per_topic))
    use_judge = bool(dcfg.get("judge_filter", True))
    min_judge = float(dcfg.get("min_judge_score", min_judge_score))

    kept: list[dict[str, Any]] = []
    for topic in track(topics, description="蒸馏造数"):
        prompt = DISTILL_PROMPT.format(n=pairs_per_topic, topic=topic[:2000])
        resp = completion(
            [{"role": "user", "content": prompt}],
            cfg,
            model=teacher_model,
            temperature=0.8,
            max_tokens=2048,
            operation="distill",
        )
        raw = resp.choices[0].message.content or ""
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            continue
        try:
            items = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        for item in items:
            if not isinstance(item, dict) or not item.get("output"):
                continue
            row = {
                "instruction": item.get("instruction", ""),
                "input": item.get("input", ""),
                "output": item.get("output", ""),
            }
            if use_judge:
                q = f"{row['instruction']}\n{row['input']}".strip()
                score, _ = judge_response(q, row["output"], row["output"], cfg)
                if score < min_judge:
                    continue
            kept.append(row)
    return kept


def distill_from_file(path: Path, output: Path, cfg: dict[str, Any] | None = None) -> int:
    text = path.read_text(encoding="utf-8", errors="ignore")
    topics = [t.strip() for t in re.split(r"\n#{1,3}\s+|\n\n+", text) if t.strip()]
    rows = distill_from_topics(topics[:50], cfg)
    write_jsonl(output, rows)
    console.print(f"[green]蒸馏完成[/green] {len(rows)} 条 → {output}")
    return len(rows)
