#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.lora_manager import export_adapter_copy, list_adapters, registry_to_markdown, update_adapter_metadata


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA 适配器管理")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出已注册 LoRA")

    exp = sub.add_parser("export", help="导出 LoRA 副本")
    exp.add_argument("name")
    exp.add_argument("-o", "--output", required=True)

    md = sub.add_parser("markdown", help="输出注册表 Markdown")

    meta = sub.add_parser("set-meta", help="更新 LoRA 元数据（LM Studio 模型 ID 等）")
    meta.add_argument("name")
    meta.add_argument("--lm-studio-model", default="")
    meta.add_argument("--eval-score", type=float, default=None)
    meta.add_argument("--gguf-path", default="")

    args = p.parse_args()

    if args.cmd == "list":
        for a in list_adapters():
            print(f"{a.get('name')}\t{a.get('path')}\t{a.get('base_model')}")
    elif args.cmd == "export":
        dest = export_adapter_copy(args.name, Path(args.output))
        print(f"已导出 → {dest}")
    elif args.cmd == "markdown":
        print(registry_to_markdown())
    elif args.cmd == "set-meta":
        fields = {}
        if args.lm_studio_model:
            fields["lm_studio_model"] = args.lm_studio_model
        if args.eval_score is not None:
            fields["eval_score"] = args.eval_score
        if args.gguf_path:
            fields["gguf_path"] = args.gguf_path
        print(update_adapter_metadata(args.name, **fields))


if __name__ == "__main__":
    main()
