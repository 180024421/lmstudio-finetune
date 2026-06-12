from __future__ import annotations

from src.sweep import run_sweep


def test_sweep_dry_run():
    results = run_sweep({"lora.r": [8]}, dry_run=True)
    assert len(results) == 1
    assert results[0]["params"]["lora.r"] == 8
