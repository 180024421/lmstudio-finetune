from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from .config_loader import ROOT, load_config
from .dataset import load_jsonl
from .lora_manager import register_adapter

console = Console()


def run_unsloth(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    try:
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError as e:
        raise ImportError("请安装 Unsloth: pip install unsloth") from e

    base_model = cfg["base_model"]
    out_dir = ROOT / cfg.get("output_dir", "output/unsloth_run1")
    out_dir.mkdir(parents=True, exist_ok=True)
    ucfg = cfg.get("unsloth") or {}
    dcfg = cfg.get("dataset") or {}
    lcfg = cfg.get("lora") or {}
    tcfg = cfg.get("train") or {}

    max_seq = int(dcfg.get("max_seq_length", 2048))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq,
        load_in_4bit=bool((cfg.get("quantization") or {}).get("load_in_4bit", True)),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lcfg.get("r", 16)),
        lora_alpha=int(lcfg.get("lora_alpha", 32)),
        lora_dropout=float(lcfg.get("lora_dropout", 0.05)),
        target_modules=lcfg.get("target_modules"),
        use_gradient_checkpointing=bool(ucfg.get("gradient_checkpointing", True)),
    )

    train_path = ROOT / dcfg.get("train_file", "data/examples/train.jsonl")
    train_ds = load_jsonl(train_path, int(dcfg.get("max_samples", 0)), dcfg.get("chat_template", "qwen"))

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(tcfg.get("num_epochs", 3)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 100)),
        fp16=not bool(tcfg.get("bf16", True)),
        report_to="none",
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=train_ds, args=args, max_seq_length=max_seq)
    console.print("[bold green]Unsloth 训练开始[/bold green]")
    trainer.train()

    adapter_dir = out_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    register_adapter(
        lcfg.get("name") or "unsloth",
        adapter_dir,
        base_model=base_model,
        description="Unsloth 训练",
        tags=["unsloth"],
    )
    return adapter_dir
