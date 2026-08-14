from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.observability.dashboard.api_client import DashboardAPIClient, DashboardAPIError


def test_client_can_decode_explicitly_accepted_non_success_status(monkeypatch):
    response = MagicMock(status_code=503)
    response.json.return_value = {"status": "not_ready", "bootstrap_required": True}
    monkeypatch.setattr(
        "src.observability.dashboard.api_client.httpx.request",
        lambda *args, **kwargs: response,
    )

    result = DashboardAPIClient().get("/health/ready", accepted_statuses={503})

    assert result == {"status": "not_ready", "bootstrap_required": True}


def test_client_still_rejects_unaccepted_error_status(monkeypatch):
    response = MagicMock(status_code=503, text="not ready")
    response.json.return_value = {"detail": "not ready"}
    monkeypatch.setattr(
        "src.observability.dashboard.api_client.httpx.request",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(DashboardAPIError, match="503: not ready"):
        DashboardAPIClient().get("/health/ready")
