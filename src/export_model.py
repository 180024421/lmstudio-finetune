from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config_loader import ROOT, load_config
from .lora_manager import resolve_adapter

console = Console()

GGUF_QUANT_TYPES = ("f16", "q8_0", "q4_k_m", "q4_0")


def merge_lora(cfg: dict[str, Any] | None = None, *, adapter_path: Path | None = None) -> Path:
    cfg = cfg or load_config()
    base_model = cfg["base_model"]
    out_dir = ROOT / cfg.get("output_dir", "output/run1")

    if adapter_path is None:
        lora_name = (cfg.get("lora") or {}).get("name")
        if lora_name:
            try:
                adapter_path = resolve_adapter(lora_name)
            except KeyError:
                adapter_path = out_dir / "adapter"
        else:
            adapter_path = out_dir / "adapter"

    if not adapter_path.exists():
        raise FileNotFoundError(f"未找到 adapter: {adapter_path}，请先训练")

    ecfg = cfg.get("export") or {}
    merged_dir = ROOT / ecfg.get("merged_dir", "output/merged")
    merged_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]合并 LoRA[/cyan] {adapter_path} → {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    write_model_card(merged_dir, cfg, adapter_path=adapter_path)
    console.print(f"[green]合并模型已保存[/green] {merged_dir}")
    return merged_dir


from .model_card import write_enhanced_model_card as write_model_card
def write_ollama_modelfile(merged_dir: Path, cfg: dict[str, Any], out_path: Path | None = None) -> Path:
    out_path = out_path or merged_dir / "Modelfile"
    ecfg = cfg.get("export") or {}
    template = (cfg.get("dataset") or {}).get("chat_template", "qwen")
    system = ecfg.get("ollama_system", "你是一个有帮助的 AI 助手。")
    gguf = ecfg.get("gguf_out", "output/model.gguf")
    gguf_path = ROOT / gguf if not Path(str(gguf)).is_absolute() else Path(gguf)

    lines = [
        f"FROM {gguf_path.as_posix() if gguf_path.exists() else './model.gguf'}",
        f'SYSTEM """{system}"""',
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.9",
        f"# chat_template: {template}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Ollama Modelfile[/green] {out_path}")
    return out_path


def convert_to_gguf(cfg: dict[str, Any] | None = None, *, quant: str | None = None) -> Path | None:
    cfg = cfg or load_config()
    ecfg = cfg.get("export") or {}
    merged_dir = ROOT / ecfg.get("merged_dir", "output/merged")
    quant = quant or ecfg.get("gguf_quant", "f16")
    if quant not in GGUF_QUANT_TYPES:
        raise ValueError(f"不支持的量化: {quant}，可选 {GGUF_QUANT_TYPES}")

    gguf_name = ecfg.get("gguf_out", "output/model.gguf")
    if quant != "f16":
        p = Path(gguf_name)
        gguf_name = str(p.parent / f"{p.stem}-{quant}{p.suffix}")
    gguf_out = ROOT / gguf_name
    llama_dir = ecfg.get("llama_cpp_dir", "")

    if not merged_dir.exists():
        merge_lora(cfg)

    convert_script = None
    if llama_dir:
        for name in ("convert_hf_to_gguf.py", "convert.py"):
            p = Path(llama_dir) / name
            if p.exists():
                convert_script = p
                break

    if convert_script is None:
        console.print("[yellow]未配置 llama.cpp，跳过 GGUF。[/yellow]\n设置 export.llama_cpp_dir 后重试。")
        write_ollama_modelfile(merged_dir, cfg)
        return None

    gguf_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(convert_script), str(merged_dir), "--outfile", str(gguf_out), "--outtype", quant]
    console.print(f"[cyan]转换 GGUF ({quant})[/cyan] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    console.print(f"[green]GGUF 已生成[/green] {gguf_out}")
    write_ollama_modelfile(merged_dir, cfg)
    return gguf_out


def export_all(cfg: dict[str, Any] | None = None) -> dict[str, Path | None]:
    cfg = cfg or load_config()
    ecfg = cfg.get("export") or {}
    merged = merge_lora(cfg)
    results: dict[str, Path | None] = {"merged": merged}
    quants = ecfg.get("gguf_quants") or [ecfg.get("gguf_quant", "f16")]
    for q in quants:
        key = f"gguf_{q}"
        results[key] = convert_to_gguf(cfg, quant=q)
    results["modelfile"] = write_ollama_modelfile(merged, cfg)
    return results
