from __future__ import annotations

from src.config_schema import validate_config
from src.dialogue_analytics import analyze_dialogue_dataset
from src.export_wizard import check_export_readiness
from src.hard_example_sampler import sample_hard_examples
from src.lmstudio_health import recommend_base_model, recommend_quantization
from src.release_check import run_release_check


def test_extended_config_invalid_quant():
    errs = validate_config({"base_model": "x", "export": {"gguf_quant": "bad"}})
    assert any("gguf_quant" in e for e in errs)


def test_dialogue_analytics(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":""}]}\n',
        encoding="utf-8",
    )
    stats = analyze_dialogue_dataset(p)
    assert stats["total"] == 1
    assert stats["empty_replies"] >= 1


def test_export_wizard():
    info = check_export_readiness({"base_model": "Qwen/Qwen2.5-1.5B-Instruct", "output_dir": "output/run1"})
    assert "adapter_exists" in info


def test_lmstudio_recommendations():
    assert "q4" in recommend_quantization(4.0).lower()
    assert "1.5B" in recommend_base_model(8.0) or "Qwen" in recommend_base_model(8.0)


def test_release_check():
    report = run_release_check({"base_model": "x", "dataset": {"train_file": "data/examples/train.jsonl"}}, skip_lmstudio=True)
    assert "steps" in report


def test_hard_examples(tmp_path):
    eval_p = tmp_path / "eval.json"
    eval_p.write_text(
        '{"mode":"rule","total":1,"avg_score":0.3,"samples":[{"prompt":"q","expected":"a","score":0.2}]}',
        encoding="utf-8",
    )
    out = tmp_path / "hard.jsonl"
    r = sample_hard_examples(eval_p, out_path=out, threshold=0.6)
    assert r["count"] == 1
