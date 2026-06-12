from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield lineno, json.loads(line)


def load_rows(path: Path, max_samples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in iter_jsonl(path):
        rows.append(row)
        if max_samples and len(rows) >= max_samples:
            break
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_fingerprint(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True)
