from __future__ import annotations

from pathlib import Path
from typing import Any

from .data_io import iter_jsonl, row_fingerprint


def diff_datasets(a: Path, b: Path, *, sample_limit: int = 20) -> dict[str, Any]:
    fps_a: dict[str, dict[str, Any]] = {}
    for lineno, row in iter_jsonl(a):
        fps_a[row_fingerprint(row)] = {"line": lineno, "row": row}

    fps_b: dict[str, dict[str, Any]] = {}
    for lineno, row in iter_jsonl(b):
        fps_b[row_fingerprint(row)] = {"line": lineno, "row": row}

    only_a = [fps_a[k] for k in fps_a if k not in fps_b]
    only_b = [fps_b[k] for k in fps_b if k not in fps_a]
    common = len(fps_a) - len(only_a)

    def _preview(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for item in items[:sample_limit]:
            row = item["row"]
            if "messages" in row:
                user = next((m.get("content", "") for m in row["messages"] if m.get("role") == "user"), "")
                preview = user[:80]
            elif "instruction" in row:
                preview = str(row.get("instruction", ""))[:80]
            else:
                preview = str(row)[:80]
            out.append({"line": item["line"], "preview": preview})
        return out

    return {
        "a": str(a),
        "b": str(b),
        "count_a": len(fps_a),
        "count_b": len(fps_b),
        "common": common,
        "only_in_a": len(only_a),
        "only_in_b": len(only_b),
        "samples_only_a": _preview(only_a),
        "samples_only_b": _preview(only_b),
    }
