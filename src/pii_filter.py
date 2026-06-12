from __future__ import annotations

import re
from typing import Any

PHONE = re.compile(r"1[3-9]\d{9}")
EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
ID_CARD = re.compile(r"\d{17}[\dXx]|\d{15}")


def redact_text(text: str) -> str:
    text = PHONE.sub("[手机号]", text)
    text = EMAIL.sub("[邮箱]", text)
    text = ID_CARD.sub("[证件号]", text)
    return text


def redact_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "messages" in out:
        out["messages"] = [
            {**m, "content": redact_text(str(m.get("content", "")))} if isinstance(m, dict) else m
            for m in out["messages"]
        ]
    for key in ("instruction", "input", "output", "prompt", "chosen", "rejected"):
        if key in out:
            out[key] = redact_text(str(out[key]))
    return out
