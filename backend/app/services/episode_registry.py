from __future__ import annotations

from threading import RLock

from app.core.exceptions import NotFoundError
from app.models.entities import GameState


class EpisodeRegistry:
    def __init__(self) -> None:
        self._games: dict[str, GameState] = {}
        self._lock = RLock()

    def add(self, state: GameState) -> GameState:
        with self._lock:
            self._games[state.game_id] = state
            return state

    def get(self, game_id: str) -> GameState:
        with self._lock:
            if game_id not in self._games:
                raise NotFoundError(
                    f"Unknown game_id: {game_id}",
                    category="game_not_found",
                    details={"game_id": game_id},
                )
            return self._games[game_id]

    def update(self, state: GameState) -> GameState:
        with self._lock:
            self._games[state.game_id] = state
            return state

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self._games.keys())

