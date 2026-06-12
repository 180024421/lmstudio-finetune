from __future__ import annotations

from typing import Any

from .chat_templates import messages_to_text


def row_to_rag_text(row: dict[str, Any], template: str = "qwen") -> dict[str, str]:
    """RAG 格式: context + question → answer"""
    context = str(row.get("context", "")).strip()
    question = str(row.get("question") or row.get("input") or "").strip()
    answer = str(row.get("answer") or row.get("output", "")).strip()
    if not question and "instruction" in row:
        question = str(row.get("instruction", "")).strip()
    user = f"参考文档：\n{context}\n\n问题：{question}" if context else question
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    return {"text": messages_to_text(messages, template)}


def is_rag_row(row: dict[str, Any]) -> bool:
    return bool(row.get("context")) and bool(row.get("question") or row.get("instruction"))
