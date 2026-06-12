from __future__ import annotations

from src.rag_dataset import is_rag_row, row_to_rag_text


def test_rag_row():
    row = {"context": "doc", "question": "q", "answer": "a"}
    assert is_rag_row(row)
    text = row_to_rag_text(row)["text"]
    assert "doc" in text and "a" in text
