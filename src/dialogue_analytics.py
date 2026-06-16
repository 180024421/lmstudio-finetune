"""多轮对话数据质量分析。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .data_io import iter_jsonl


def analyze_dialogue_dataset(path: Path) -> dict[str, Any]:
    turn_counts: list[int] = []
    empty_replies = 0
    duplicate_user = 0
    total = 0
    seen_user: set[str] = set()

    for _, row in iter_jsonl(path):
        total += 1
        if "messages" in row:
            msgs = row["messages"]
            turn_counts.append(len(msgs))
            for m in msgs:
                if m.get("role") == "assistant" and not str(m.get("content", "")).strip():
                    empty_replies += 1
            users = [str(m.get("content", "")) for m in msgs if m.get("role") == "user"]
            for u in users:
                if u in seen_user:
                    duplicate_user += 1
                seen_user.add(u)
        else:
            turn_counts.append(2)
            out = str(row.get("output", ""))
            if not out.strip():
                empty_replies += 1
            inst = str(row.get("instruction", ""))
            if inst in seen_user:
                duplicate_user += 1
            seen_user.add(inst)

    dist = Counter(turn_counts)
    avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0
    return {
        "total": total,
        "avg_turns": round(avg_turns, 2),
        "turn_distribution": dict(sorted(dist.items())),
        "empty_replies": empty_replies,
        "duplicate_user_prompts": duplicate_user,
        "empty_reply_rate": round(empty_replies / max(total, 1), 4),
    }


def format_analytics_markdown(stats: dict[str, Any]) -> str:
    lines = [
        "## 对话数据分析",
        "",
        f"- 总条数: {stats['total']}",
        f"- 平均轮数: {stats['avg_turns']}",
        f"- 空回复数: {stats['empty_replies']}（率 {stats['empty_reply_rate']:.1%}）",
        f"- 重复用户问法: {stats['duplicate_user_prompts']}",
        f"- 轮数分布: {stats['turn_distribution']}",
    ]
    return "\n".join(lines)
