from __future__ import annotations

from typing import Any

import httpx


class ChallengeLikeHttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        path_prefix: str = "/challenge/api",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.path_prefix = path_prefix.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout_seconds)

    def close(self) -> None:
        self._client.close()

    def start_game(self, *, mode: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{self.path_prefix}/start",
            json={"api_key": self.api_key, "mode": mode},
        )

    def get_status(self, game_id: str) -> dict[str, Any]:
        return self._request("GET", f"{self.path_prefix}/status/{game_id}")

    def place_box(self, game_id: str, action: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{self.path_prefix}/place",
            json={"game_id": game_id, **action},
        )

    def stop_game(self, game_id: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"{self.path_prefix}/stop",
            json={"game_id": game_id, "api_key": self.api_key},
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._client.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()
