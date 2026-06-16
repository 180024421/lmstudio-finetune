from __future__ import annotations

from typing import Any

REQUIRED_TOP = ("base_model",)
CHAT_TEMPLATES = ("qwen", "llama3", "chatml", "mistral", "alpaca")
GGUF_QUANTS = ("f16", "q8_0", "q4_k_m", "q4_0")


def _positive_int(val: Any, name: str, errors: list[str], *, min_val: int = 1) -> None:
    try:
        if int(val) < min_val:
            errors.append(f"{name} 必须 >= {min_val}")
    except (TypeError, ValueError):
        errors.append(f"{name} 必须是整数")


def validate_config(cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_TOP:
        if not cfg.get(key):
            errors.append(f"缺少必填项: {key}")

    template = (cfg.get("dataset") or {}).get("chat_template", "qwen")
    if template not in CHAT_TEMPLATES:
        errors.append(f"不支持的 chat_template: {template}")

    dcfg = cfg.get("dataset") or {}
    if not dcfg.get("train_file"):
        errors.append("dataset.train_file 不能为空")
    _positive_int(dcfg.get("max_seq_length", 2048), "dataset.max_seq_length", errors)

    train = cfg.get("train") or {}
    if float(train.get("num_epochs", 1)) <= 0:
        errors.append("train.num_epochs 必须 > 0")
    _positive_int(train.get("per_device_train_batch_size", 1), "train.per_device_train_batch_size", errors)

    lora = cfg.get("lora") or {}
    if int(lora.get("r", 16)) < 1:
        errors.append("lora.r 必须 >= 1")

    api_mode = (cfg.get("api") or {}).get("mode", "proxy")
    if api_mode not in ("proxy", "local"):
        errors.append("api.mode 必须是 proxy 或 local")

    ecfg = cfg.get("export") or {}
    for q in ecfg.get("gguf_quants") or []:
        if q not in GGUF_QUANTS:
            errors.append(f"export.gguf_quants 含不支持量化: {q}")
    if ecfg.get("gguf_quant") and ecfg.get("gguf_quant") not in GGUF_QUANTS:
        errors.append(f"export.gguf_quant 不支持: {ecfg.get('gguf_quant')}")

    dpo = cfg.get("dpo") or {}
    if dpo.get("train_file") and not str(dpo["train_file"]).endswith(".jsonl"):
        errors.append("dpo.train_file 应为 .jsonl 文件")

    inbox = cfg.get("inbox") or {}
    if inbox.get("poll_seconds") is not None:
        try:
            if int(inbox["poll_seconds"]) < 5:
                errors.append("inbox.poll_seconds 建议 >= 5")
        except (TypeError, ValueError):
            errors.append("inbox.poll_seconds 必须是整数")

    alignment = cfg.get("alignment") or {}
    for key in ("skip_sft", "skip_dpo"):
        if key in alignment and not isinstance(alignment[key], bool):
            errors.append(f"alignment.{key} 必须是布尔值")

    promo = cfg.get("promo") or {}
    if promo.get("eval_job_ratio") is not None:
        try:
            r = float(promo["eval_job_ratio"])
            if not 0 <= r <= 1:
                errors.append("promo.eval_job_ratio 应在 0~1")
        except (TypeError, ValueError):
            errors.append("promo.eval_job_ratio 必须是数字")

    return errors
