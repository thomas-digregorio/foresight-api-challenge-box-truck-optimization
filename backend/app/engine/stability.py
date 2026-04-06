from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.models.entities import GameState


class StabilityBackend(ABC):
    @abstractmethod
    def validate_candidate(self, state: GameState, metadata: dict[str, Any]) -> tuple[bool, str | None, dict[str, Any]]:
        raise NotImplementedError


class NoOpDeterministicStabilityBackend(StabilityBackend):
    def validate_candidate(self, state: GameState, metadata: dict[str, Any]) -> tuple[bool, str | None, dict[str, Any]]:
        return True, None, {}


class PhysicsStabilityBackend(StabilityBackend):
    def validate_candidate(self, state: GameState, metadata: dict[str, Any]) -> tuple[bool, str | None, dict[str, Any]]:
        raise NotImplementedError("Physics-based settling/stability is reserved for a future backend.")

