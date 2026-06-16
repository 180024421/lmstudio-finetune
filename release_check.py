#!/usr/bin/env python3
"""发布检查清单。"""

from __future__ import annotations

import argparse

from src.release_check import format_release_markdown, run_release_check


def main() -> None:
    p = argparse.ArgumentParser(description="训练→导出→GGUF→LM Studio 发布检查")
    p.add_argument("--skip-lmstudio", action="store_true")
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()
    report = run_release_check(skip_lmstudio=args.skip_lmstudio)
    if args.markdown:
        print(format_release_markdown(report))
    else:
        import json

        print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
