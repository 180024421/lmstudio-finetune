from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from .config_loader import ROOT, load_config
from .dataset import load_jsonl

console = Console()


def _bnb_config(cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    qcfg = cfg.get("quantization") or {}
    if not qcfg.get("load_in_4bit", True):
        return None
    dtype_name = qcfg.get("bnb_4bit_compute_dtype", "bfloat16")
    dtype = torch.bfloat16 if "bf" in str(dtype_name) else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )


def run_train(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    base_model = cfg["base_model"]
    out_dir = ROOT / cfg.get("output_dir", "output/run1")
    out_dir.mkdir(parents=True, exist_ok=True)

    dcfg = cfg.get("dataset") or {}
    train_path = ROOT / dcfg.get("train_file", "data/examples/train.jsonl")
    eval_path = dcfg.get("eval_file")
    max_samples = int(dcfg.get("max_samples", 0))
    max_seq = int(dcfg.get("max_seq_length", 2048))

    console.print(f"[cyan]加载数据集[/cyan] {train_path}")
    train_ds = load_jsonl(train_path, max_samples)
    eval_ds = None
    if eval_path and Path(ROOT / eval_path).exists():
        eval_ds = load_jsonl(ROOT / eval_path, max_samples)

    console.print(f"[cyan]加载基座模型[/cyan] {base_model}")
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

    lcfg = cfg.get("lora") or {}
    peft_config = LoraConfig(
        r=int(lcfg.get("r", 16)),
        lora_alpha=int(lcfg.get("lora_alpha", 32)),
        lora_dropout=float(lcfg.get("lora_dropout", 0.05)),
        target_modules=lcfg.get("target_modules"),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tcfg = cfg.get("train") or {}
    use_bf16 = bool(tcfg.get("bf16", True)) and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(tcfg.get("num_epochs", 3)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 100)),
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=int(tcfg.get("eval_steps", 100)) if eval_ds else None,
        bf16=use_bf16,
        fp16=bool(tcfg.get("fp16", False)) and not use_bf16,
        optim="paged_adamw_8bit" if bnb else "adamw_torch",
        report_to="none",
        seed=int(tcfg.get("seed", 42)),
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        max_seq_length=max_seq,
    )
    console.print("[bold green]开始训练[/bold green]")
    trainer.train()
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    console.print(f"[green]LoRA 已保存[/green] {adapter_dir}")
    return adapter_dir
