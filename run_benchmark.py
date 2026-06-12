#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmark_runner import print_benchmark, run_and_save
from src.config_loader import load_config
from src.regression_gate import check_regression, save_baseline


def main() -> None:
    p = argparse.ArgumentParser(description="标准 Benchmark 评测")
    p.add_argument("--file", default=None)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--set-baseline", action="store_true")
    p.add_argument("--regression", action="store_true")
    p.add_argument("--min-delta", type=float, default=-0.05)
    args = p.parse_args()
    cfg = load_config()
    bench = Path(args.file) if args.file else None
    result = run_and_save(bench, cfg, use_judge=args.judge)
    print_benchmark(result)
    score = result.judge_score if args.judge and result.judge_score is not None else result.rule_score
    if args.set_baseline:
        save_baseline(score, name=result.name, mode="judge" if args.judge else "rule")
        print(f"已设置基线: {score:.3f}")
    if args.regression:
        gate = check_regression(score, min_delta=args.min_delta, name=result.name)
        print(gate)
        if not gate["ok"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
