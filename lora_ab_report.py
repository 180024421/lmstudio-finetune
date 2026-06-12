#!/usr/bin/env python3
"""多 LoRA A/B 自动对比 + HTML 报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config_loader import ROOT, load_config
from src.lora_ab_report import compare_loras, print_ab_table, save_lora_ab_html
from src.lora_manager import list_by_tag, update_adapter_metadata


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA A/B 自动对比报告")
    p.add_argument("--config", default=None)
    p.add_argument("--profile", default=None)
    p.add_argument("--file", default=None, help="评测 JSONL")
    p.add_argument("--loras", nargs="*", default=None, help="LoRA 名称列表")
    p.add_argument("--tag", default="promo", help="未指定 --loras 时按标签筛选")
    p.add_argument("--base-model", default="", help="LM Studio 基座模型名，用于对比")
    p.add_argument("--mode", choices=["judge", "lmstudio"], default="judge")
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("-o", "--output", default="output/lora_ab_report.html")
    p.add_argument("--update-registry", action="store_true", help="将均分写回 LoRA 注册表")
    args = p.parse_args()

    cfg = load_config(Path(args.config) if args.config else None, profile=args.profile)
    dcfg = cfg.get("dataset") or {}
    eval_path = Path(args.file) if args.file else ROOT / dcfg.get("eval_file", "data/examples/eval.jsonl")

    names = args.loras or [a["name"] for a in list_by_tag(args.tag)]
    if len(names) < 2 and not args.base_model:
        raise SystemExit("需要至少 2 个 LoRA，或指定 --base-model + 1 个 LoRA")

    result = compare_loras(
        names,
        eval_path,
        cfg,
        max_samples=args.max_samples,
        mode=args.mode,
        base_model_name=args.base_model or None,
    )
    out = save_lora_ab_html(result, Path(args.output))
    print_ab_table(result)
    print(f"HTML 报告: {out}")

    if args.update_registry:
        for item in result.get("ranking", []):
            if item.get("name") == "base":
                continue
            update_adapter_metadata(item["name"], eval_score=item.get("avg_score"))

    json_path = Path(args.output).with_suffix(".json")
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
