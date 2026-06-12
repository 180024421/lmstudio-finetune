from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, TrainingArguments
from trl import SFTTrainer

from .config_loader import ROOT, load_config
from .dataset import load_jsonl
from .experiment_registry import register_experiment
from .lora_manager import register_adapter, resolve_adapter
from .train_callbacks import CopyBestFromCheckpointCallback, MetricsJsonlCallback, TrainCompleteNotifyCallback

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


def _report_to(cfg: dict[str, Any]) -> str | list[str]:
    mon = cfg.get("monitoring") or {}
    targets: list[str] = []
    if mon.get("tensorboard", True):
        targets.append("tensorboard")
    if mon.get("wandb", False):
        targets.append("wandb")
    return targets if targets else "none"


def run_train(cfg: dict[str, Any] | None = None, *, resume: bool = False) -> Path:
    cfg = cfg or load_config()
    base_model = cfg["base_model"]
    out_dir = ROOT / cfg.get("output_dir", "output/run1")
    out_dir.mkdir(parents=True, exist_ok=True)

    dcfg = cfg.get("dataset") or {}
    template = dcfg.get("chat_template", "qwen")
    train_path = ROOT / dcfg.get("train_file", "data/examples/train.jsonl")
    eval_path = dcfg.get("eval_file")
    max_samples = int(dcfg.get("max_samples", 0))
    max_seq = int(dcfg.get("max_seq_length", 2048))

    console.print(f"[cyan]加载数据集[/cyan] {train_path} (template={template})")
    train_ds = load_jsonl(train_path, max_samples, template)
    eval_ds = None
    if eval_path and Path(ROOT / eval_path).exists():
        eval_ds = load_jsonl(ROOT / eval_path, max_samples, template)

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
    tcfg = cfg.get("train") or {}
    continue_from = tcfg.get("continue_from_adapter") or lcfg.get("continue_from")
    adapter_loaded = False
    if continue_from:
        adapter_path = Path(continue_from)
        if not adapter_path.is_absolute():
            try:
                adapter_path = resolve_adapter(str(continue_from))
            except KeyError:
                adapter_path = ROOT / continue_from
        if adapter_path.exists():
            console.print(f"[cyan]增量续训[/cyan] 加载已有 LoRA: {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
            adapter_loaded = True
    if not adapter_loaded:
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

    use_bf16 = bool(tcfg.get("bf16", True)) and torch.cuda.is_bf16_supported()
    has_eval = eval_ds is not None
    metric_name = tcfg.get("metric_for_best_model", "eval_loss")
    load_best = bool(tcfg.get("load_best_model_at_end", has_eval))

    ds_path = tcfg.get("deepspeed") or ""
    deepspeed_cfg = None
    if ds_path:
        p = Path(ds_path)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            deepspeed_cfg = str(p)

    args = TrainingArguments(
        output_dir=str(out_dir),
        deepspeed=deepspeed_cfg,
        num_train_epochs=float(tcfg.get("num_epochs", 3)),
        per_device_train_batch_size=int(tcfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(tcfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 100)),
        eval_strategy="steps" if has_eval else "no",
        eval_steps=int(tcfg.get("eval_steps", 100)) if has_eval else None,
        save_total_limit=int(tcfg.get("save_total_limit", 3)),
        load_best_model_at_end=load_best and has_eval,
        metric_for_best_model=metric_name if has_eval else None,
        greater_is_better=False if "loss" in str(metric_name) else True,
        bf16=use_bf16,
        fp16=bool(tcfg.get("fp16", False)) and not use_bf16,
        optim="paged_adamw_8bit" if bnb else "adamw_torch",
        report_to=_report_to(cfg),
        seed=int(tcfg.get("seed", 42)),
        logging_dir=str(out_dir / "logs"),
    )

    callbacks: list[Any] = [
        MetricsJsonlCallback(out_dir / "metrics.jsonl"),
    ]
    if bool(tcfg.get("early_stopping", has_eval)) and has_eval:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(tcfg.get("early_stopping_patience", 3))))
    if bool((cfg.get("monitoring") or {}).get("notify_on_complete", True)):
        callbacks.append(TrainCompleteNotifyCallback())
    if load_best and has_eval:
        callbacks.append(CopyBestFromCheckpointCallback(out_dir, out_dir / "best_adapter"))

    resume_ckpt = None
    if resume or tcfg.get("resume_from_checkpoint"):
        val = tcfg.get("resume_from_checkpoint")
        if val is True or val == "true":
            checkpoints = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            resume_ckpt = str(checkpoints[-1]) if checkpoints else None
        elif val:
            resume_ckpt = str(ROOT / val) if not str(val).startswith("checkpoint") else str(out_dir / val)

    sft_kw: dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        max_seq_length=max_seq,
        callbacks=callbacks,
    )
    try:
        trainer = SFTTrainer(**sft_kw, tokenizer=tokenizer)
    except TypeError:
        trainer = SFTTrainer(**sft_kw, processing_class=tokenizer)
    console.print("[bold green]开始训练[/bold green]")
    trainer.train(resume_from_checkpoint=resume_ckpt)

    adapter_dir = out_dir / "adapter"
    if (out_dir / "best_adapter").exists():
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        shutil.copytree(out_dir / "best_adapter", adapter_dir)
        console.print(f"[green]使用最佳 checkpoint[/green] {out_dir / 'best_adapter'}")

    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    console.print(f"[green]LoRA 已保存[/green] {adapter_dir}")

    lora_name = (lcfg.get("name") or out_dir.name)
    register_adapter(
        lora_name,
        adapter_dir,
        base_model=base_model,
        description=lcfg.get("description", ""),
        tags=lcfg.get("tags") or [],
    )

    eval_score = None
    if bool(tcfg.get("auto_eval_after_train", False)):
        try:
            from .eval_runner import run_eval_lmstudio
            eval_file = dcfg.get("eval_file")
            if eval_file and (ROOT / eval_file).exists():
                er = run_eval_lmstudio(ROOT / eval_file, cfg, max_samples=int(tcfg.get("auto_eval_samples", 5)))
                eval_score = er.avg_score
                console.print(f"[green]自动评测[/green] avg={eval_score:.3f}")
        except Exception as ex:
            console.print(f"[yellow]自动评测跳过[/yellow] {ex}")

    register_experiment(
        name=lora_name,
        cfg=cfg,
        metrics_path=out_dir / "metrics.jsonl",
        eval_score=eval_score,
        eval_mode="lmstudio" if eval_score is not None else "",
    )

    if eval_score is not None and bool(tcfg.get("regression_check", False)):
        from .regression_gate import check_regression

        gate = check_regression(
            eval_score,
            min_delta=float(tcfg.get("regression_min_delta", -0.05)),
            name=lora_name,
        )
        if not gate["ok"]:
            console.print(f"[red]回归门禁未通过[/red] {gate['reason']}")
        else:
            console.print(f"[green]回归检查通过[/green]")

    return adapter_dir
