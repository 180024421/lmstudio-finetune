from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import ORPOTrainer

from .config_loader import ROOT, load_config
from .data_io import load_rows
from .train_lora import _bnb_config, _report_to

console = Console()


def _load_orpo_dataset(path: Path, max_samples: int = 0) -> Dataset:
    rows: list[dict[str, Any]] = []
    for row in load_rows(path, max_samples):
        if not row.get("prompt") or not row.get("chosen"):
            raise ValueError("ORPO 需要 prompt + chosen；rejected 可选")
        rejected = row.get("rejected", "")
        if not str(rejected).strip():
            continue
        rows.append(
            {
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": rejected,
            }
        )
    return Dataset.from_list(rows)


def run_orpo(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    ocfg = cfg.get("orpo") or {}
    base_model = cfg["base_model"]
    out_dir = ROOT / ocfg.get("output_dir", "output/orpo_run1")
    out_dir.mkdir(parents=True, exist_ok=True)

    data_path = ROOT / ocfg.get("train_file", "data/examples/dpo.jsonl")
    train_ds = _load_orpo_dataset(data_path, int(ocfg.get("max_samples", 0)))

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
        num_train_epochs=float(ocfg.get("num_epochs", 1)),
        per_device_train_batch_size=int(ocfg.get("batch_size", 1)),
        learning_rate=float(ocfg.get("learning_rate", 5e-5)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 100)),
        report_to=_report_to(cfg),
        remove_unused_columns=False,
    )

    trainer = ORPOTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        beta=float(ocfg.get("beta", 0.1)),
    )
    console.print("[bold green]开始 ORPO 训练[/bold green]")
    trainer.train()
    adapter_dir = out_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir
