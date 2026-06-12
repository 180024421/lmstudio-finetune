from __future__ import annotations

from src.chat_templates import messages_to_text, row_to_text


def test_qwen_template():
    text = messages_to_text([{"role": "user", "content": "hi"}], "qwen")
    assert "<|im_start|>user" in text
    assert "assistant" in text


def test_row_instruction():
    row = {"instruction": "A", "input": "B", "output": "C"}
    text = row_to_text(row, "alpaca")["text"]
    assert "Instruction" in text
    assert "C" in text


def test_llama3_template():
    text = messages_to_text([{"role": "user", "content": "hello"}], "llama3")
    assert "hello" in text
