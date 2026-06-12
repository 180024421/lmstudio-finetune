from __future__ import annotations

from src.train_multi import build_torchrun_command


def test_build_torchrun_command():
    cmd = build_torchrun_command({"multi_gpu": {"nproc": 2}}, nproc=2, extra_args=["--dry-run"])
    assert any("train.py" in part for part in cmd)
    assert any("2" in part for part in cmd)
