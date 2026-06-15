from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_dedup import deduplicate_rows
from .data_io import iter_jsonl, row_fingerprint, write_jsonl


def merge_jsonl_files(
    inputs: list[Path],
    output: Path,
    *,
    dedup: bool = True,
    template: str = "qwen",
    threshold: float = 0.85,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_file: dict[str, int] = {}
    skipped_dup = 0

    for inp in inputs:
        if not inp.exists():
            raise FileNotFoundError(f"文件不存在: {inp}")
        count = 0
        for _, row in iter_jsonl(inp):
            fp = row_fingerprint(row)
            if fp in seen:
                skipped_dup += 1
                continue
            seen.add(fp)
            rows.append(row)
            count += 1
        per_file[str(inp)] = count

    removed_fuzzy = 0
    if dedup and rows:
        rows, removed_fuzzy = deduplicate_rows(rows, template=template, threshold=threshold)

    write_jsonl(output, rows)
    return {
        "output": str(output),
        "total": len(rows),
        "per_file": per_file,
        "exact_duplicates_skipped": skipped_dup,
        "fuzzy_duplicates_removed": removed_fuzzy,
    }
