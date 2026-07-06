"""Tests for import_lua_api."""

from pathlib import Path

from src.import_lua_api import _parse_lua_api_table, export_lua_api_jsonl


def test_parse_lua_api_table_minimal():
    md = """
| 函数 | 说明 |
|------|------|
| `bot.tap(x, y)` | 点击 |
| `bot.delay(seconds)` | 等待 |
"""
    rows = _parse_lua_api_table(md)
    assert len(rows) == 2
    assert rows[0]["input"] == "bot.tap(x, y)"


def test_export_lua_api_jsonl(tmp_path: Path):
    md = tmp_path / "LUA.md"
    md.write_text(
        "| 函数 | 说明 |\n|------|------|\n| `bot.log(msg)` | 写日志 |\n",
        encoding="utf-8",
    )
    stats = export_lua_api_jsonl(md, tmp_path / "out.jsonl")
    assert stats["train"] >= 1
    assert (tmp_path / "lua_api_train.jsonl").is_file()
