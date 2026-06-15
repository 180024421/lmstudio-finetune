#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from src.regression_gate import auto_baseline_from_experiments, check_regression, load_baseline, save_baseline


def main() -> None:
    p = argparse.ArgumentParser(description="评测回归门禁")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show")
    setb = sub.add_parser("set")
    setb.add_argument("score", type=float)
    chk = sub.add_parser("check")
    chk.add_argument("score", type=float)
    chk.add_argument("--min-delta", type=float, default=-0.05)
    sub.add_parser("auto-baseline")

    args = p.parse_args()
    if args.cmd == "show":
        print(json.dumps(load_baseline(), ensure_ascii=False, indent=2))
    elif args.cmd == "set":
        print(json.dumps(save_baseline(args.score), ensure_ascii=False))
    elif args.cmd == "check":
        print(json.dumps(check_regression(args.score, min_delta=args.min_delta), ensure_ascii=False))
    elif args.cmd == "auto-baseline":
        s = auto_baseline_from_experiments()
        if s is None:
            raise SystemExit("无历史实验评测分")
        print(json.dumps(save_baseline(s), ensure_ascii=False))


if __name__ == "__main__":
    main()
