from __future__ import annotations

from src.config_loader import _deep_merge, load_config


def test_deep_merge():
    base = {"train": {"epochs": 3, "lr": 1e-4}, "dataset": {"max_samples": 0}}
    override = {"train": {"epochs": 1}, "output_dir": "out"}
    merged = _deep_merge(base, override)
    assert merged["train"]["epochs"] == 1
    assert merged["train"]["lr"] == 1e-4
    assert merged["output_dir"] == "out"


def test_load_config_with_profile():
    cfg = load_config(profile="dev")
    assert cfg.get("output_dir") == "output/dev_run"
    assert (cfg.get("dataset") or {}).get("max_samples") == 50
