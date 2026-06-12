from __future__ import annotations

from pathlib import Path
from typing import Any

from .chat_templates import row_to_text
from .data_io import load_rows, write_jsonl


def _shingles(text: str, n: int = 3) -> set[str]:
    t = "".join(text.split()).lower()
    if len(t) <= n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate_rows(
    rows: list[dict[str, Any]],
    *,
    template: str = "qwen",
    threshold: float = 0.85,
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    shingles_list: list[set[str]] = []
    removed = 0
    for row in rows:
        text = row_to_text(row, template)["text"]
        sh = _shingles(text)
        dup = False
        for prev in shingles_list:
            if jaccard(sh, prev) >= threshold:
                dup = True
                break
        if dup:
            removed += 1
            continue
        kept.append(row)
        shingles_list.append(sh)
    return kept, removed


def deduplicate_file(
    input_path: Path,
    output_path: Path,
    *,
    template: str = "qwen",
    threshold: float = 0.85,
) -> dict[str, int]:
    rows = load_rows(input_path)
    kept, removed = deduplicate_rows(rows, template=template, threshold=threshold)
    write_jsonl(output_path, kept)
    return {"input": len(rows), "kept": len(kept), "removed": removed}
