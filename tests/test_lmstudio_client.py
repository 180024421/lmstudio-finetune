from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.lmstudio_client import chat, check_lm_studio, list_models, openai_client


def test_openai_client_from_config():
    cfg = {
        "lm_studio": {
            "base_url": "http://test-host:9999/v1",
            "api_key": "test-key",
            "model": "my-model",
        }
    }
    client, model = openai_client(cfg)
    assert str(client.base_url).rstrip("/") == "http://test-host:9999/v1"
    assert model == "my-model"


def test_openai_client_defaults():
    cfg = {"lm_studio": {}}
    _, model = openai_client(cfg)
    assert model == "local-model"


@patch("src.lmstudio_client.completion")
def test_chat_returns_content(mock_completion):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="你好"))]
    mock_completion.return_value = mock_resp
    assert chat("hi", cfg={"lm_studio": {}}) == "你好"


@patch("src.lmstudio_client.openai_client")
def test_list_models_success(mock_openai_client):
    mock_client = MagicMock()
    mock_model = MagicMock(id="model-a")
    mock_client.models.list.return_value.data = [mock_model]
    mock_openai_client.return_value = (mock_client, "local")
    assert list_models(cfg={"lm_studio": {}}) == ["model-a"]


@patch("src.lmstudio_client.openai_client")
def test_list_models_on_error(mock_openai_client):
    mock_client = MagicMock()
    mock_client.models.list.side_effect = RuntimeError("down")
    mock_openai_client.return_value = (mock_client, "local")
    assert list_models(cfg={"lm_studio": {}}) == []


@patch("urllib.request.urlopen")
def test_check_lm_studio_ok(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_resp
    assert check_lm_studio(cfg={"lm_studio": {"base_url": "http://127.0.0.1:1234/v1"}}) is True


@patch("urllib.request.urlopen", side_effect=OSError("connection refused"))
def test_check_lm_studio_fail(_mock_urlopen):
    assert check_lm_studio(cfg={"lm_studio": {}}) is False
