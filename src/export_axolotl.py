from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config_loader import ROOT, load_config


def to_axolotl_yaml(cfg: dict[str, Any] | None = None, output: Path | None = None) -> Path:
    cfg = cfg or load_config()
    dcfg = cfg.get("dataset") or {}
    lcfg = cfg.get("lora") or {}
    tcfg = cfg.get("train") or {}

    train_path = ROOT / dcfg.get("train_file", "data/examples/train.jsonl")
    ax = {
        "base_model": cfg["base_model"],
        "model_type": "AutoModelForCausalLM",
        "tokenizer_type": "AutoTokenizer",
        "datasets": [{"path": str(train_path), "type": "json", "field": "text"}],
        "dataset_prepared_path": str(ROOT / "output" / "axolotl_prepared"),
        "val_set_size": 0.1,
        "output_dir": str(ROOT / cfg.get("output_dir", "output/axolotl_run")),
        "sequence_len": int(dcfg.get("max_seq_length", 2048)),
        "sample_packing": False,
        "adapter": "lora",
        "lora_r": int(lcfg.get("r", 16)),
        "lora_alpha": int(lcfg.get("lora_alpha", 32)),
        "lora_dropout": float(lcfg.get("lora_dropout", 0.05)),
        "micro_batch_size": int(tcfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(tcfg.get("gradient_accumulation_steps", 8)),
        "num_epochs": float(tcfg.get("num_epochs", 3)),
        "learning_rate": float(tcfg.get("learning_rate", 2e-4)),
        "optimizer": "adamw_bnb_8bit",
        "bf16": bool(tcfg.get("bf16", True)),
        "load_in_4bit": bool((cfg.get("quantization") or {}).get("load_in_4bit", True)),
    }
    out = output or ROOT / "output" / "axolotl_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.dump(ax, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return out
