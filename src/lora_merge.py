from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config_loader import ROOT, load_config
from .lora_manager import resolve_adapter

console = Console()


def merge_multiple_loras(
    adapter_names: list[str],
    cfg: dict[str, Any] | None = None,
    *,
    output_dir: Path | None = None,
) -> Path:
    """将多个 LoRA 依次合并进基座（顺序敏感）。"""
    cfg = cfg or load_config()
    base_model = cfg["base_model"]
    ecfg = cfg.get("export") or {}
    out = output_dir or ROOT / ecfg.get("merged_multi_dir", "output/merged_multi")
    out.mkdir(parents=True, exist_ok=True)

    if not adapter_names:
        raise ValueError("至少指定一个 LoRA 名称")

    console.print(f"[cyan]合并 {len(adapter_names)} 个 LoRA[/cyan] → {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    for name in adapter_names:
        adapter_path = resolve_adapter(name)
        console.print(f"  + {name} ({adapter_path})")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        model = model.merge_and_unload()

    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    console.print(f"[green]多 LoRA 合并完成[/green] {out}")
    return out
