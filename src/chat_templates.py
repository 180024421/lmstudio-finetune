from __future__ import annotations

from typing import Any

SUPPORTED_TEMPLATES = ("qwen", "llama3", "chatml", "mistral", "alpaca")


def messages_to_text(messages: list[dict[str, str]], template: str = "qwen") -> str:
    t = (template or "qwen").lower()
    if t not in SUPPORTED_TEMPLATES:
        raise ValueError(f"不支持的 chat_template: {template}，可选: {SUPPORTED_TEMPLATES}")

    if t == "qwen":
        return _qwen(messages)
    if t == "llama3":
        return _llama3(messages)
    if t == "chatml":
        return _chatml(messages)
    if t == "mistral":
        return _mistral(messages)
    return _alpaca(messages)


def _qwen(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _llama3(messages: list[dict[str, str]]) -> str:
    parts: list[str] = ["<|begin_of_text|>"]
    for m in messages:
        role = m.get("role", "user")
        tag = "user" if role == "user" else "assistant" if role == "assistant" else "system"
        content = m.get("content", "")
        parts.append(
            f"<|start_header_id|>{tag}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _chatml(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


def _mistral(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    system = ""
    for m in messages:
        if m.get("role") == "system":
            system = m.get("content", "")
            break
    if system:
        chunks.append(f"[INST] {system}\n\n")
    for m in messages:
        role = m.get("role", "user")
        if role == "system":
            continue
        content = m.get("content", "")
        if role == "user":
            chunks.append(f"[INST] {content} [/INST]")
        elif role == "assistant":
            chunks.append(f" {content}</s>")
    return "".join(chunks)


def _alpaca(messages: list[dict[str, str]]) -> str:
    user_parts: list[str] = []
    assistant = ""
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant":
            assistant = content
        else:
            user_parts.append(content)
    user = "\n".join(user_parts).strip()
    return (
        f"### Instruction:\n{user}\n\n### Response:\n{assistant}"
        if assistant
        else f"### Instruction:\n{user}\n\n### Response:\n"
    )


def row_to_text(row: dict[str, Any], template: str = "qwen") -> dict[str, str]:
    if "messages" in row:
        return {"text": messages_to_text(row["messages"], template)}
    if "instruction" in row:
        inst = row.get("instruction", "")
        inp = row.get("input", "")
        out = row.get("output", "")
        user = f"{inst}\n{inp}".strip() if inp else inst
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": out},
        ]
        return {"text": messages_to_text(messages, template)}
    raise ValueError(f"无法解析样本，需要 messages 或 instruction 字段: {list(row.keys())}")
