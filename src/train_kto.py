from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import KTOTrainer

from .config_loader import ROOT, load_config
from .data_io import load_rows
from .train_lora import _bnb_config, _report_to

console = Console()


def _load_kto_dataset(path: Path, max_samples: int = 0) -> Dataset:
    rows: list[dict[str, Any]] = []
    for row in load_rows(path, max_samples):
        if "prompt" in row and "completion" in row and "label" in row:
            rows.append({
                "prompt": row["prompt"],
                "completion": row["completion"],
                "label": bool(row["label"]),
            })
        elif "prompt" in row and "chosen" in row:
            rows.append({"prompt": row["prompt"], "completion": row["chosen"], "label": True})
            if row.get("rejected"):
                rows.append({"prompt": row["prompt"], "completion": row["rejected"], "label": False})
        else:
            raise ValueError("KTO 需要 prompt+completion+label 或 prompt+chosen/rejected")
    return Dataset.from_list(rows)


def run_kto(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    kcfg = cfg.get("kto") or {}
    base_model = cfg["base_model"]
    out_dir = ROOT / kcfg.get("output_dir", "output/kto_run1")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = ROOT / kcfg.get("train_file", "data/examples/kto.jsonl")
    train_ds = _load_kto_dataset(data_path, int(kcfg.get("max_samples", 0)))

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
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lcfg.get("r", 16)),
            lora_alpha=int(lcfg.get("lora_alpha", 32)),
            lora_dropout=float(lcfg.get("lora_dropout", 0.05)),
            target_modules=lcfg.get("target_modules"),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    tcfg = cfg.get("train") or {}
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=float(kcfg.get("num_epochs", 1)),
        per_device_train_batch_size=int(kcfg.get("batch_size", 1)),
        learning_rate=float(kcfg.get("learning_rate", 5e-5)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 100)),
        report_to=_report_to(cfg),
        remove_unused_columns=False,
    )

    trainer = KTOTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        beta=float(kcfg.get("beta", 0.1)),
    )
    console.print("[bold green]开始 KTO 训练[/bold green]")
    trainer.train()
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir
