from __future__ import annotations

import re
from typing import Any


def _parse_params_b(model_id: str) -> float | None:
    """从模型名解析参数量（十亿），如 Qwen2.5-1.5B-Instruct → 1.5。"""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]", model_id)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*[Mm]", model_id)
    if m:
        return float(m.group(1)) / 1000.0
    return None


def estimate_vram_gb(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    粗略估算 QLoRA 训练显存（GB）。
    基于参数量、4bit 量化、序列长度与 batch 设置的启发式公式。
    """
    base_model = str(cfg.get("base_model", ""))
    params_b = _parse_params_b(base_model)

    qcfg = cfg.get("quantization") or {}
    load_4bit = bool(qcfg.get("load_in_4bit", True))
    dcfg = cfg.get("dataset") or {}
    tcfg = cfg.get("train") or {}
    lcfg = cfg.get("lora") or {}

    max_seq = int(dcfg.get("max_seq_length", 2048))
    batch = int(tcfg.get("per_device_train_batch_size", 1))
    grad_accum = int(tcfg.get("gradient_accumulation_steps", 8))
    lora_r = int(lcfg.get("r", 16))

    if params_b is None:
        params_b = 3.0
        note = f"无法从模型名解析参数量，按 {params_b}B 估算"
    else:
        note = f"解析参数量约 {params_b}B"

    if load_4bit:
        weight_gb = params_b * 0.7
    else:
        weight_gb = params_b * 2.0

    lora_gb = (lora_r / 16) * 0.3 * (params_b / 1.5)
    activation_gb = (max_seq / 2048) * batch * 0.8 * (params_b / 1.5)
    optimizer_gb = params_b * 0.15 if load_4bit else params_b * 0.5
    overhead_gb = 1.0

    estimated = weight_gb + lora_gb + activation_gb + optimizer_gb + overhead_gb
    recommended = estimated * 1.2

    return {
        "base_model": base_model,
        "params_b": params_b,
        "load_4bit": load_4bit,
        "max_seq_length": max_seq,
        "batch_size": batch,
        "gradient_accumulation_steps": grad_accum,
        "estimated_gb": round(estimated, 1),
        "recommended_gb": round(recommended, 1),
        "breakdown": {
            "weights_gb": round(weight_gb, 1),
            "lora_gb": round(lora_gb, 1),
            "activation_gb": round(activation_gb, 1),
            "optimizer_gb": round(optimizer_gb, 1),
            "overhead_gb": overhead_gb,
        },
        "note": note,
    }


def format_vram_markdown(info: dict[str, Any]) -> str:
    b = info["breakdown"]
    lines = [
        "## 显存估算（QLoRA 训练）",
        "",
        f"- 模型: `{info['base_model']}`（约 {info['params_b']}B）",
        f"- 4bit 量化: {'是' if info['load_4bit'] else '否'}",
        f"- max_seq_length: {info['max_seq_length']} | batch: {info['batch_size']}",
        f"- **预估显存**: ~{info['estimated_gb']} GB",
        f"- **建议显存**: ≥ {info['recommended_gb']} GB（含 20% 余量）",
        "",
        "| 组成 | GB |",
        "|------|-----|",
        f"| 权重 | {b['weights_gb']} |",
        f"| LoRA | {b['lora_gb']} |",
        f"| 激活 | {b['activation_gb']} |",
        f"| 优化器 | {b['optimizer_gb']} |",
        f"| 开销 | {b['overhead_gb']} |",
        "",
        f"_{info['note']}_",
    ]
    return "\n".join(lines)
