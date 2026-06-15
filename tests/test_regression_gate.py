from __future__ import annotations

from src.regression_gate import check_regression, save_baseline


def test_regression_pass(tmp_path, monkeypatch):
    from src import regression_gate

    monkeypatch.setattr(regression_gate, "GATE_PATH", tmp_path / "b.json")
    save_baseline(0.8)
    r = check_regression(0.82, min_delta=-0.05)
    assert r["ok"]


def test_regression_fail(tmp_path, monkeypatch):
    from src import regression_gate

    monkeypatch.setattr(regression_gate, "GATE_PATH", tmp_path / "b.json")
    save_baseline(0.8)
    r = check_regression(0.5, min_delta=-0.05)
    assert not r["ok"]
