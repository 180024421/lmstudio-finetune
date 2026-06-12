from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

from .config_loader import ROOT, load_config
from .data_io import load_rows
from .lora_manager import resolve_adapter
from .train_lora import _bnb_config, _report_to

console = Console()


def _load_dpo_dataset(path: Path, max_samples: int = 0) -> Dataset:
    rows = load_rows(path, max_samples)
    for i, row in enumerate(rows):
        for key in ("prompt", "chosen", "rejected"):
            if key not in row or not str(row[key]).strip():
                raise ValueError(f"第 {i + 1} 行缺少 DPO 字段: {key}")
    return Dataset.from_list(rows)


def run_dpo(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    dpo_cfg = cfg.get("dpo") or {}
    base_model = cfg["base_model"]
    out_dir = ROOT / dpo_cfg.get("output_dir", "output/dpo_run1")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = ROOT / dpo_cfg.get("train_file", "data/examples/dpo.jsonl")
    max_samples = int(dpo_cfg.get("max_samples", 0))
    console.print(f"[cyan]加载 DPO 数据[/cyan] {data_path}")
    train_ds = _load_dpo_dataset(data_path, max_samples)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = _bnb_config(cfg)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not bnb else None,
    )
    if bnb:
        model = prepare_model_for_kbit_training(model)

    ref_model = None
    if not dpo_cfg.get("reference_free", True):
        ref_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not bnb else None,
        )

    lcfg = cfg.get("lora") or {}
    sft_adapter = dpo_cfg.get("continue_from_adapter") or ""
    if sft_adapter:
        ap = resolve_adapter(str(sft_adapter)) or (ROOT / sft_adapter)
        if not ap.exists():
            raise FileNotFoundError(f"SFT adapter 不存在: {sft_adapter}")
        console.print(f"[cyan]从 SFT adapter 继续 DPO[/cyan] {ap}")
        model = PeftModel.from_pretrained(model, ap, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=int(lcfg.get("r", 16)),
            lora_alpha=int(lcfg.get("lora_alpha", 32)),
            lora_dropout=float(lcfg.get("lora_dropout", 0.05)),
            target_modules=lcfg.get("target_modules"),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    tcfg = cfg.get("train") or {}
    use_bf16 = bool(tcfg.get("bf16", True)) and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(dpo_cfg.get("num_epochs", tcfg.get("num_epochs", 1))),
        per_device_train_batch_size=int(dpo_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(dpo_cfg.get("learning_rate", 5e-5)),
        bf16=use_bf16,
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 100)),
        report_to=_report_to(cfg),
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        beta=float(dpo_cfg.get("beta", 0.1)),
        max_length=int(dpo_cfg.get("max_length", 1024)),
        max_prompt_length=int(dpo_cfg.get("max_prompt_length", 512)),
    )
    console.print("[bold green]开始 DPO 训练[/bold green]")
    trainer.train()
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    console.print(f"[green]DPO LoRA 已保存[/green] {adapter_dir}")
    return adapter_dir
