"""从 auto-script-studio docs/LUA.md 导出 lmstudio-finetune 训练 JSONL。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .data_io import write_jsonl


def _parse_lua_api_table(md_text: str) -> list[dict[str, str]]:
    """解析 Markdown API 表格为 instruction/input/output 行。"""
    rows: list[dict[str, str]] = []
    in_table = False
    for line in md_text.splitlines():
        if line.strip().startswith("| `bot."):
            in_table = True
        if not in_table:
            continue
        if not line.strip().startswith("|"):
            if in_table and rows:
                break
            continue
        if "---" in line or "函数" in line:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 2:
            continue
        fn_cell, desc = parts[0], parts[1]
        m = re.search(r"`(bot\.[^`]+)`", fn_cell)
        if not m:
            continue
        fn = m.group(1)
        rows.append(
            {
                "instruction": f"说明 auto-script-studio Lua API：{fn}",
                "input": fn,
                "output": desc,
            }
        )
    return rows


def _lua_snippet_rows(md_text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    blocks = re.findall(r"```lua\n(.*?)```", md_text, re.S)
    for block in blocks:
        code = block.strip()
        if len(code) < 20:
            continue
        rows.append(
            {
                "instruction": "根据 auto-script-studio bot API 编写 Lua 示例",
                "input": "写一个使用 bot.yoloDetect 或 bot.findImage 的示例",
                "output": code,
            }
        )
    return rows


def export_lua_api_jsonl(lua_md: Path, out_path: Path, *, eval_ratio: float = 0.1) -> dict:
    text = lua_md.read_text(encoding="utf-8")
    rows = _parse_lua_api_table(text) + _lua_snippet_rows(text)
    if not rows:
        raise ValueError(f"未从 {lua_md} 解析到训练行")
    n_eval = max(1, int(len(rows) * eval_ratio)) if len(rows) > 5 else 1
    if len(rows) <= n_eval:
        train = rows
        eval_rows: list[dict[str, str]] = []
    else:
        train = rows[:-n_eval]
        eval_rows = rows[-n_eval:]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path.with_name("lua_api_train.jsonl"), train)
    write_jsonl(out_path.with_name("lua_api_eval.jsonl"), eval_rows)
    return {"train": len(train), "eval": len(eval_rows), "source": str(lua_md)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 auto-script-studio Lua API 微调数据")
    parser.add_argument(
        "--lua-md",
        type=Path,
        default=Path("../auto-script-studio/docs/LUA.md"),
        help="LUA.md 路径",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("data/lua_api_train.jsonl"))
    args = parser.parse_args(argv)
    stats = export_lua_api_jsonl(args.lua_md.resolve(), args.output)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
