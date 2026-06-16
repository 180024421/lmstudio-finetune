from __future__ import annotations

from unittest.mock import patch

from src.package_check import check_package
from src.webhook_notify import send_webhook


def test_package_check_missing():
    r = check_package("nonexistent_pkg_xyz_12345")
    assert r["status"] == "missing"


def test_webhook_fallback_no_url():
    with patch("src.webhook_notify.notify") as m:
        r = send_webhook("", "t", "msg")
        assert r["channel"] == "desktop"
        m.assert_called_once()
