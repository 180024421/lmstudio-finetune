from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config_loader import ROOT
from .data_io import load_rows, write_jsonl

PENDING = ROOT / "data" / "review" / "pending.jsonl"
APPROVED = ROOT / "data" / "review" / "approved.jsonl"


def queue_for_review(rows: list[dict[str, Any]]) -> int:
    PENDING.parent.mkdir(parents=True, exist_ok=True)
    existing = load_rows(PENDING) if PENDING.exists() else []
    merged = existing + rows
    write_jsonl(PENDING, merged)
    return len(rows)


def list_pending() -> list[dict[str, Any]]:
    if not PENDING.exists():
        return []
    return load_rows(PENDING)


def approve_index(index: int, *, edited_output: str | None = None) -> dict[str, Any] | None:
    rows = list_pending()
    if index < 0 or index >= len(rows):
        return None
    row = rows.pop(index)
    if edited_output is not None:
        if "messages" in row:
            for m in row["messages"]:
                if m.get("role") == "assistant":
                    m["content"] = edited_output
        elif "output" in row:
            row["output"] = edited_output
    write_jsonl(PENDING, rows)
    approved = load_rows(APPROVED) if APPROVED.exists() else []
    approved.append(row)
    write_jsonl(APPROVED, approved)
    return row


def reject_index(index: int) -> bool:
    rows = list_pending()
    if index < 0 or index >= len(rows):
        return False
    rows.pop(index)
    write_jsonl(PENDING, rows)
    return True


def merge_approved_to(target: Path) -> int:
    if not APPROVED.exists():
        return 0
    approved = load_rows(APPROVED)
    existing = load_rows(target) if target.exists() else []
    write_jsonl(target, existing + approved)
    APPROVED.unlink(missing_ok=True)
    return len(approved)
