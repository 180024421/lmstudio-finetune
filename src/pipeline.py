from __future__ import annotations

from typing import Any

from rich.console import Console

from .config_loader import ROOT, load_config
from .config_schema import validate_config
from .data_validate import validate_jsonl
from .experiment_registry import register_experiment
from .export_model import export_all
from .train_lora import run_train

console = Console()


def run_full_pipeline(
    cfg: dict[str, Any] | None = None,
    *,
    skip_train: bool = False,
    skip_export: bool = False,
    skip_eval: bool = False,
) -> dict[str, Any]:
    cfg = cfg or load_config()
    errs = validate_config(cfg)
    if errs:
        raise ValueError("配置错误: " + "; ".join(errs))

    dcfg = cfg.get("dataset") or {}
    train_file = ROOT / dcfg.get("train_file", "data/examples/train.jsonl")
    report = validate_jsonl(train_file, template=dcfg.get("chat_template", "qwen"))
    if not report.ok:
        raise ValueError(f"训练数据校验失败: {report.errors[0].message}")

    result: dict[str, Any] = {"validate": report.to_dict()}

    if not skip_train:
        adapter = run_train(cfg)
        result["adapter"] = str(adapter)

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
            if bool(pcfg.get("regression", False)):
                from .regression_gate import check_regression

                result["regression"] = check_regression(er.avg_score)

    register_experiment(
        name=(cfg.get("lora") or {}).get("name", "pipeline"),
        cfg=cfg,
        eval_score=result.get("eval_score"),
    )
    console.print("[bold green]Pipeline 完成[/bold green]")
    return result
