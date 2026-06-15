from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_torch_deps(monkeypatch):
    """仅在本 fixture 作用域内 mock 重型 ML 依赖，避免污染其他测试。"""
    names = ("torch", "peft", "transformers", "trl", "bitsandbytes")
    for name in names:
        monkeypatch.setitem(sys.modules, name, MagicMock())
    for name in list(sys.modules):
        if name in ("src.pipeline", "src.export_model", "src.train_lora"):
            if name in sys.modules:
                monkeypatch.delitem(sys.modules, name, raising=False)
    yield


@pytest.fixture
def run_full_pipeline(mock_torch_deps):
    from src.pipeline import run_full_pipeline as fn

    return fn
