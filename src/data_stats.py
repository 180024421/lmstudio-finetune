from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .chat_templates import row_to_text
from .data_io import iter_jsonl


def estimate_tokens(text: str) -> int:
    # 中英文混合粗略估计：约 3~4 字符/token
    return max(1, len(text) // 3)


def compute_stats(path: Path, *, template: str = "qwen") -> dict[str, Any]:
    char_lens: list[int] = []
    token_lens: list[int] = []
    roles = Counter()
    formats = Counter()

    for _, row in iter_jsonl(path):
        if "messages" in row:
            formats["messages"] += 1
            for m in row.get("messages") or []:
                if isinstance(m, dict):
                    roles[m.get("role", "?")] += 1
        elif "instruction" in row:
            formats["instruction"] += 1
        text = row_to_text(row, template)["text"]
        char_lens.append(len(text))
        token_lens.append(estimate_tokens(text))

    n = len(char_lens)
    if n == 0:
        return {"path": str(path), "count": 0}

    def _pct(vals: list[int], p: float) -> int:
        s = sorted(vals)
        idx = min(len(s) - 1, int(len(s) * p))
        return s[idx]

    return {
        "path": str(path),
        "count": n,
        "formats": dict(formats),
        "roles": dict(roles),
        "chars": {
            "min": min(char_lens),
            "max": max(char_lens),
            "avg": round(sum(char_lens) / n, 1),
            "p50": _pct(char_lens, 0.5),
            "p95": _pct(char_lens, 0.95),
        },
        "tokens_est": {
            "min": min(token_lens),
            "max": max(token_lens),
            "avg": round(sum(token_lens) / n, 1),
            "p50": _pct(token_lens, 0.5),
            "p95": _pct(token_lens, 0.95),
        },
    }
