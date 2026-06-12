from __future__ import annotations

from pathlib import Path

from src.lora_ab_report import ranking_score, save_lora_ab_html


def test_save_ab_html(tmp_path: Path):
    result = {
        "eval_file": "data/promo_eval.jsonl",
        "mode": "judge",
        "max_samples": 5,
        "winner": "promo-v2",
        "delta_vs_runner_up": 0.05,
        "ranking": [
            {"name": "promo-v2", "avg_score": 0.85, "total": 5, "lm_studio_model": "promo-v2"},
            {"name": "promo-v1", "avg_score": 0.80, "total": 5, "lm_studio_model": "promo-v1"},
        ],
        "reports": {
            "promo-v2": {
                "samples": [{"score": 0.9, "prompt": "写标题", "notes": "好"}],
            },
        },
    }
    out = save_lora_ab_html(result, tmp_path / "ab.html")
    html = out.read_text(encoding="utf-8")
    assert "promo-v2" in html
    assert ranking_score(result) == 0.85
