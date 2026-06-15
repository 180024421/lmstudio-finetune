from __future__ import annotations

from typing import Any

REQUIRED_TOP = ("base_model",)
CHAT_TEMPLATES = ("qwen", "llama3", "chatml", "mistral", "alpaca")


def validate_config(cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_TOP:
        if not cfg.get(key):
            errors.append(f"缺少必填项: {key}")

    template = (cfg.get("dataset") or {}).get("chat_template", "qwen")
    if template not in CHAT_TEMPLATES:
        errors.append(f"不支持的 chat_template: {template}")

    train = cfg.get("train") or {}
    if float(train.get("num_epochs", 1)) <= 0:
        errors.append("train.num_epochs 必须 > 0")
    if int(train.get("per_device_train_batch_size", 1)) < 1:
        errors.append("train.per_device_train_batch_size 必须 >= 1")

    lora = cfg.get("lora") or {}
    if int(lora.get("r", 16)) < 1:
        errors.append("lora.r 必须 >= 1")

    api_mode = (cfg.get("api") or {}).get("mode", "proxy")
    if api_mode not in ("proxy", "local"):
        errors.append("api.mode 必须是 proxy 或 local")

    return errors
