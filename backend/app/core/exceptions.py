from __future__ import annotations

from typing import Any


class EngineError(Exception):
    def __init__(self, message: str, *, category: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.category = category
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "error": "validation_error",
            "message": self.message,
            "details": {
                "category": self.category,
                **self.details,
            },
        }


class ValidationError(EngineError):
    pass


class NotFoundError(EngineError):
    pass


class StateConflictError(EngineError):
    pass

