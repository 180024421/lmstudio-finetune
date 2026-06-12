from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from .chat_templates import row_to_text
from .data_io import load_rows
from .rag_dataset import is_rag_row, row_to_rag_text


def row_to_training_text(row: dict, template: str = "qwen") -> dict[str, str]:
    if is_rag_row(row):
        return row_to_rag_text(row, template)
    return row_to_text(row, template)


def load_jsonl(path: Path, max_samples: int = 0, template: str = "qwen") -> Dataset:
    rows = load_rows(path, max_samples)
    return Dataset.from_list([row_to_training_text(r, template) for r in rows])
