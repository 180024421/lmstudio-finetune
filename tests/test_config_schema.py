from __future__ import annotations

from src.config_loader import load_config
from src.config_schema import validate_config


def test_valid_config():
    cfg = load_config()
    # use example if no config.yaml fields
    from src.config_loader import ROOT

    ex = ROOT / "config.example.yaml"
    if ex.exists():
        import yaml

        cfg = yaml.safe_load(ex.read_text(encoding="utf-8"))
    assert validate_config(cfg) == []


def test_invalid_template():
    errs = validate_config({"base_model": "x", "dataset": {"chat_template": "bad"}})
    assert any("chat_template" in e for e in errs)
