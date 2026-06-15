from __future__ import annotations

from src.pii_filter import redact_row, redact_text


def test_redact_phone():
    assert "[手机号]" in redact_text("联系我 13812345678")


def test_redact_row():
    row = redact_row({"instruction": "x", "output": "email test@example.com"})
    assert "[邮箱]" in row["output"]
