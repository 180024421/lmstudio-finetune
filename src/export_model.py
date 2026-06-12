from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config_loader import ROOT, load_config

console = Console()


def merge_lora(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    base_model = cfg["base_model"]
    out_dir = ROOT / cfg.get("output_dir", "output/run1")
    adapter_dir = out_dir / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"未找到 adapter: {adapter_dir}，请先训练")

    ecfg = cfg.get("export") or {}
    merged_dir = ROOT / ecfg.get("merged_dir", "output/merged")
    merged_dir.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]合并 LoRA 到基座模型[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    console.print(f"[green]合并模型已保存[/green] {merged_dir}")
    return merged_dir


def convert_to_gguf(cfg: dict[str, Any] | None = None) -> Path | None:
    cfg = cfg or load_config()
    ecfg = cfg.get("export") or {}
    merged_dir = ROOT / ecfg.get("merged_dir", "output/merged")
    gguf_out = ROOT / ecfg.get("gguf_out", "output/model.gguf")
    llama_dir = ecfg.get("llama_cpp_dir", "")

    if not merged_dir.exists():
        merge_lora(cfg)

    convert_script = None
    if llama_dir:
        p = Path(llama_dir) / "convert_hf_to_gguf.py"
        if p.exists():
            convert_script = p

    if convert_script is None:
        console.print(
            "[yellow]未配置 llama.cpp，跳过 GGUF 转换。[/yellow]\n"
            "请安装 https://github.com/ggerganov/llama.cpp 并设置 export.llama_cpp_dir，\n"
            "或在 LM Studio 中直接加载 HuggingFace 合并目录（部分版本支持）。"
        )
        return None

    gguf_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(convert_script), str(merged_dir), "--outfile", str(gguf_out), "--outtype", "f16"]
    console.print(f"[cyan]转换 GGUF[/cyan] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    console.print(f"[green]GGUF 已生成[/green] {gguf_out}")
    console.print("LM Studio → 加载模型 → 选择该 .gguf 文件")
    return gguf_out
