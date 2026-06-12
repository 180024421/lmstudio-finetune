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
from .data_quality import filter_rows
from .review_store import queue_for_review

console = Console()

GEN_PROMPT = """你是训练数据标注专家。根据以下文档片段，生成 {n} 条高质量的问答训练样本。
要求：
1. 问题应覆盖片段中的关键信息
2. 回答准确、简洁、可直接用作 assistant 回复
3. 只输出 JSON 数组，无 markdown，格式：
[{{"instruction":"问题","input":"","output":"回答"}}]
文档片段：
---
{chunk}
---"""


def _chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _parse_qa_json(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        return []
    try:
        data = json.loads(m.group())
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict) and x.get("output")]
    except json.JSONDecodeError:
        pass
    return []


def generate_from_text(
    text: str,
    cfg: dict[str, Any] | None = None,
    *,
    pairs_per_chunk: int = 3,
    chunk_size: int = 2000,
) -> list[dict[str, Any]]:
    cfg = cfg or load_config()
    _, model = openai_client(cfg)
    gen_cfg = cfg.get("data_generation") or {}
    pairs_per_chunk = int(gen_cfg.get("pairs_per_chunk", pairs_per_chunk))

    all_rows: list[dict[str, Any]] = []
    chunks = _chunk_text(text, chunk_size)
    for chunk in track(chunks, description="LM Studio 生成样本"):
        prompt = GEN_PROMPT.format(n=pairs_per_chunk, chunk=chunk)
        resp = completion(
            [{"role": "user", "content": prompt}],
            cfg,
            model=model,
            temperature=0.7,
            max_tokens=2048,
            operation="generate_data",
        )
        content = resp.choices[0].message.content or ""
        all_rows.extend(_parse_qa_json(content))
    return all_rows


def generate_from_files(
    input_paths: list[Path],
    output_path: Path,
    cfg: dict[str, Any] | None = None,
) -> int:
    combined = ""
    for p in input_paths:
        combined += f"\n\n# {p.name}\n" + p.read_text(encoding="utf-8", errors="ignore")
    rows = generate_from_text(combined, cfg)
    gcfg = (cfg or load_config()).get("data_generation") or {}
    min_score = float(gcfg.get("min_quality_score", 0.5))
    kept, rejected = filter_rows(rows, min_score=min_score)
    if gcfg.get("queue_for_review", False):
        queue_for_review(kept)
        console.print(f"[cyan]已加入校对队列[/cyan] {len(kept)} 条")
    write_jsonl(output_path, kept)
    if rejected and gcfg.get("save_rejected", True):
        rej_path = output_path.parent / f"{output_path.stem}_rejected.jsonl"
        write_jsonl(rej_path, rejected)
    console.print(f"[green]已生成 {len(kept)} 条[/green]（过滤 {len(rejected)}）→ {output_path}")
    return len(kept)
