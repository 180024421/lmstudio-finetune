from __future__ import annotations

from pathlib import Path

from src.data_convert import convert_faq_md, convert_file


def test_faq_md(tmp_path: Path):
    md = tmp_path / "f.md"
    md.write_text("## 什么是 Redis\n\n答：内存数据库\n", encoding="utf-8")
    rows = convert_faq_md(md)
    assert len(rows) == 1
    assert "Redis" in rows[0]["instruction"]


def test_convert_file_csv(tmp_path: Path):
    csv = tmp_path / "d.csv"
    csv.write_text("instruction,input,output\nQ,I,A\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    n = convert_file(csv, out, "csv")
    assert n == 1
    assert out.exists()
