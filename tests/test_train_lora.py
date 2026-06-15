from __future__ import annotations

from src.vram_estimate import estimate_vram_gb, format_vram_markdown


def test_vram_estimate_qwen_1_5b():
    cfg = {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "quantization": {"load_in_4bit": True},
        "dataset": {"max_seq_length": 2048},
        "train": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 8},
        "lora": {"r": 16},
    }
    info = estimate_vram_gb(cfg)
    assert info["params_b"] == 1.5
    assert info["estimated_gb"] > 0
    assert info["recommended_gb"] >= info["estimated_gb"]
    md = format_vram_markdown(info)
    assert "显存估算" in md
    assert "1.5" in md


def test_vram_estimate_unknown_model():
    cfg = {"base_model": "custom/unknown-model"}
    info = estimate_vram_gb(cfg)
    assert info["params_b"] == 3.0
    assert "无法从模型名解析" in info["note"]
