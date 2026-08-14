"""Typed-enough HTTP client used by the Streamlit production UI."""

from __future__ import annotations

import os
from typing import Any

import httpx


class DashboardAPIError(RuntimeError):
    pass


class DashboardAPIClient:
    def __init__(self, base_url: str | None = None, access_token: str | None = None, timeout: float = 60.0) -> None:
        self.base_url = (base_url or os.environ.get("RAG_API_URL") or "http://127.0.0.1:8766").rstrip("/")
        self.access_token = access_token
        self.timeout = timeout

    def request(
        self, method: str, path: str, *, accepted_statuses: set[int] | None = None,
        **kwargs: Any,
    ) -> Any:
        headers = dict(kwargs.pop("headers", {}))
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        try:
            response = httpx.request(method, f"{self.base_url}{path}", headers=headers, timeout=self.timeout, **kwargs)
        except httpx.HTTPError as exc:
            raise DashboardAPIError(f"无法连接后端：{exc}") from exc
        if response.status_code >= 400 and response.status_code not in (accepted_statuses or set()):
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise DashboardAPIError(f"{response.status_code}: {detail}")
        if response.status_code == 204:
            return None
        return response.json()

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> Any:
        return self.request("POST", path, **kwargs)

    def patch(self, path: str, **kwargs: Any) -> Any:
        return self.request("PATCH", path, **kwargs)

    def put(self, path: str, **kwargs: Any) -> Any:
        return self.request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> Any:
        return self.request("DELETE", path, **kwargs)
