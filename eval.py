#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import ROOT, load_config
from src.eval_runner import (
    compare_models,
    print_report,
    run_eval_judge,
    run_eval_lmstudio,
    run_eval_local,
    save_report,
    save_report_html,
)
from src.regression_gate import check_regression, save_baseline


def main() -> None:
    p = argparse.ArgumentParser(description="评测微调效果")
    p.add_argument("--config", default=None)
    p.add_argument("--file", default=None, help="eval.jsonl 路径")
    p.add_argument("--mode", choices=["lmstudio", "local", "compare", "judge"], default="lmstudio")
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("--output", default="output/eval_report.json")
    p.add_argument("--base-model", default="", help="compare 模式：基座模型名")
    p.add_argument("--finetuned-model", default="", help="compare 模式：微调模型名")
    p.add_argument("--set-baseline", action="store_true", help="将本次均分设为回归基线")
    p.add_argument("--regression", action="store_true", help="与基线对比，不达标 exit 1")
    p.add_argument("--min-delta", type=float, default=-0.05)
    p.add_argument("--html", action="store_true", help="同时输出 HTML 报告")
    args = p.parse_args()
    cfg = load_config(Path(args.config)) if args.config else load_config()
    eval_path = Path(args.file) if args.file else ROOT / cfg.get("dataset", {}).get("eval_file", "data/examples/eval.jsonl")

    if args.mode == "compare":
        if not args.base_model or not args.finetuned_model:
            raise SystemExit("compare 模式需要 --base-model 和 --finetuned-model")
        result = compare_models(
            eval_path, cfg,
            base_model_name=args.base_model,
            finetuned_model_name=args.finetuned_model,
            max_samples=args.max_samples,
        )
        save_report(result, Path(args.output))
        print(f"基座: {result['base']['avg_score']:.3f}  微调: {result['finetuned']['avg_score']:.3f}  Δ={result['delta']:+.3f}")
        return

    if args.mode == "local":
        report = run_eval_local(eval_path, cfg, max_samples=args.max_samples)
    elif args.mode == "judge":
        report = run_eval_judge(eval_path, cfg, max_samples=args.max_samples)
    else:
        report = run_eval_lmstudio(eval_path, cfg, max_samples=args.max_samples)
    print_report(report)
    save_report(report, Path(args.output))
    if args.html:
        html_path = Path(args.output).with_suffix(".html")
        save_report_html(report, html_path)
        print(f"HTML 报告: {html_path}")
    if args.set_baseline:
        save_baseline(report.avg_score, mode=args.mode)
        print(f"已设置回归基线: {report.avg_score:.3f}")
    if args.regression:
        gate = check_regression(report.avg_score, min_delta=args.min_delta)
        print(gate["reason"])
        if not gate["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
