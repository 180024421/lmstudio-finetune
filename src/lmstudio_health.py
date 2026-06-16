"""LM Studio 健康检查与加载建议。"""

from __future__ import annotations

from typing import Any

from .lmstudio_client import check_lm_studio, list_models
from .vram_estimate import _parse_params_b


def _gpu_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return round(props.total_memory / (1024**3), 1)
    except Exception:
        pass
    return None


def recommend_quantization(vram_gb: float | None) -> str:
    if vram_gb is None:
        return "q4_k_m（未检测到 GPU，建议 CPU 或小模型 Q4）"
    if vram_gb < 4:
        return "q4_0 / q4_k_m，模型 ≤1.5B，GPU Offload=0 试 CPU"
    if vram_gb < 8:
        return "q4_k_m，模型 ≤3B"
    if vram_gb < 16:
        return "q4_k_m 或 q8_0，模型 ≤7B"
    return "q8_0 或 f16，可试 7B+"


def recommend_base_model(vram_gb: float | None) -> str:
    if vram_gb is None:
        return "Qwen/Qwen2.5-0.5B-Instruct 或 Qwen/Qwen2.5-1.5B-Instruct"
    if vram_gb < 4:
        return "Qwen/Qwen2.5-0.5B-Instruct"
    if vram_gb < 8:
        return "Qwen/Qwen2.5-1.5B-Instruct"
    if vram_gb < 16:
        return "Qwen/Qwen2.5-3B-Instruct 或 Qwen2.5-7B-Instruct（Q4）"
    return "Qwen/Qwen2.5-7B-Instruct 或更大"


def run_lmstudio_health(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    from .config_loader import load_config

    cfg = cfg or load_config()
    connected = check_lm_studio(cfg)
    models = list_models(cfg) if connected else []
    vram = _gpu_vram_gb()
    base = cfg.get("base_model", "")
    params = _parse_params_b(str(base))

    tips: list[str] = []
    if not connected:
        tips.extend(
            [
                "1. 打开 LM Studio → 加载模型（建议 Qwen2.5-1.5B Q4_K_M）",
                "2. Local Server → Start（默认 http://127.0.0.1:1234）",
                "3. 若加载崩溃：GPU Offload 设为 0，或换更小量化",
            ]
        )
    elif not models:
        tips.append("API 已连通但未检测到已加载模型，请在 LM Studio 中 Load Model")
    if vram is not None and params and params * 2 > vram:
        tips.append(f"当前配置 base_model 约 {params}B，显存 {vram}GB 可能不足，建议换更小模型或 Q4")

    return {
        "connected": connected,
        "models": models,
        "gpu_vram_gb": vram,
        "recommend_quant": recommend_quantization(vram),
        "recommend_base_model": recommend_base_model(vram),
        "troubleshooting": tips,
    }


def format_health_markdown(report: dict[str, Any]) -> str:
    lines = [
        "## LM Studio 健康检查",
        "",
        f"- API: {'已连接' if report['connected'] else '未连接'}",
        f"- 已加载模型: {', '.join(report['models']) or '(无)'}",
        f"- GPU 显存: {report['gpu_vram_gb'] or '未知'} GB",
        f"- 推荐量化: {report['recommend_quant']}",
        f"- 推荐基座: {report['recommend_base_model']}",
    ]
    if report.get("troubleshooting"):
        lines.extend(["", "### 排查建议", ""])
        lines.extend(f"- {t}" for t in report["troubleshooting"])
    return "\n".join(lines)
