from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console

from .config_loader import ROOT, load_config
from .config_schema import validate_config
from .data_validate import validate_jsonl
from .experiment_registry import register_experiment
from .export_model import export_all

console = Console()


def run_sft_dpo_pipeline(
    cfg: dict[str, Any] | None = None,
    *,
    skip_sft: bool = False,
    skip_dpo: bool = False,
    skip_export: bool = False,
    skip_eval: bool = False,
) -> dict[str, Any]:
    """SFT (QLoRA) → DPO 串联对齐流水线。"""
    cfg = cfg or load_config()
    acfg = cfg.get("alignment") or {}
    skip_sft = skip_sft or bool(acfg.get("skip_sft", False))
    skip_dpo = skip_dpo or bool(acfg.get("skip_dpo", False))

    errs = validate_config(cfg)
    if errs:
        raise ValueError("配置错误: " + "; ".join(errs))

    dcfg = cfg.get("dataset") or {}
    train_file = ROOT / dcfg.get("train_file", "data/examples/train.jsonl")
    report = validate_jsonl(train_file, template=dcfg.get("chat_template", "qwen"))
    if not report.ok:
        raise ValueError(f"SFT 数据校验失败: {report.errors[0].message}")

    result: dict[str, Any] = {"validate_sft": report.to_dict()}
    adapter_path: Path | None = None

    if not skip_sft:
        from .train_lora import run_train

        adapter_path = run_train(cfg)
        result["sft_adapter"] = str(adapter_path)

    dpo_cfg = cfg.get("dpo") or {}
    dpo_file = ROOT / dpo_cfg.get("train_file", "data/examples/dpo.jsonl")
    if not skip_dpo:
        dpo_report = validate_jsonl(dpo_file, template=dcfg.get("chat_template", "qwen"))
        if not dpo_report.ok:
            raise ValueError(f"DPO 数据校验失败: {dpo_report.errors[0].message}")
        result["validate_dpo"] = dpo_report.to_dict()

        if bool(acfg.get("use_sft_adapter_for_dpo", True)) and adapter_path:
            cfg = dict(cfg)
            dpo_section = dict(cfg.get("dpo") or {})
            dpo_section["continue_from_adapter"] = str(adapter_path)
            cfg["dpo"] = dpo_section

        from .train_dpo import run_dpo

        dpo_out = run_dpo(cfg)
        result["dpo_adapter"] = str(dpo_out)

    if not skip_export:
        result["export"] = {k: str(v) for k, v in (export_all(cfg) or {}).items() if v}

    if not skip_eval:
        eval_file = dcfg.get("eval_file")
        if eval_file and (ROOT / eval_file).exists():
            from .eval_runner import run_eval_judge, run_eval_lmstudio

            pcfg = cfg.get("pipeline") or {}
            if pcfg.get("use_judge", True):
                er = run_eval_judge(ROOT / eval_file, cfg, max_samples=int(pcfg.get("eval_samples", 10)))
            else:
                er = run_eval_lmstudio(ROOT / eval_file, cfg, max_samples=int(pcfg.get("eval_samples", 10)))
            result["eval_score"] = er.avg_score

    register_experiment(
        name=f"{(cfg.get('lora') or {}).get('name', 'run')}-sft-dpo",
        cfg=cfg,
        eval_score=result.get("eval_score"),
    )
    console.print("[bold green]SFT→DPO 流水线完成[/bold green]")
    return result
