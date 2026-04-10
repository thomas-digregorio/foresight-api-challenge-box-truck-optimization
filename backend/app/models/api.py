from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TruckResponse(BaseModel):
    depth: float
    width: float
    height: float


class CurrentBoxResponse(BaseModel):
    id: str
    dimensions: tuple[float, float, float]
    weight: float


class PlacedBoxResponse(BaseModel):
    id: str
    dimensions: tuple[float, float, float]
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]


class ChallengeStartRequest(BaseModel):
    api_key: str = Field(min_length=1)
    mode: Literal["dev", "compete"] = "compete"


class LocalStartRequest(BaseModel):
    api_key: str = Field(min_length=1)
    mode: Literal["dev", "compete"] = "dev"
    seed: int | None = None


class ChallengeStartResponse(BaseModel):
    game_id: str
    truck: TruckResponse
    current_box: CurrentBoxResponse | None
    boxes_remaining: int
    mode: Literal["dev", "compete"]


class LocalStartResponse(ChallengeStartResponse):
    pass


class PlaceRequest(BaseModel):
    game_id: str
    box_id: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]


class StopRequest(BaseModel):
    game_id: str
    api_key: str = Field(min_length=1)


class ChallengePlaceResponse(BaseModel):
    status: Literal["ok", "terminated"]
    placed_boxes: list[PlacedBoxResponse]
    current_box: CurrentBoxResponse | None
    boxes_remaining: int
    density: float
    game_status: Literal["in_progress", "completed"]
    termination_reason: str | None


class ChallengeStatusResponse(BaseModel):
    game_id: str
    game_status: Literal["in_progress", "completed"]
    mode: Literal["dev", "compete"]
    boxes_placed: int
    boxes_remaining: int
    density: float
    placed_boxes: list[PlacedBoxResponse]
    current_box: CurrentBoxResponse | None


class LocalGameStateResponse(BaseModel):
    game_id: str
    status: Literal["ok", "terminated"]
    truck: TruckResponse
    placed_boxes: list[PlacedBoxResponse]
    current_box: CurrentBoxResponse | None
    boxes_remaining: int
    density: float
    game_status: Literal["in_progress", "completed", "timed_out", "no_feasible_placement"]
    termination_reason: str | None
    mode: Literal["dev", "compete"]
    created_at: str
    current_box_started_at: str | None = None
    current_box_deadline: str | None = None
    timeout_seconds: float
    loading_guide_x: float | None = None


class PreviewRequest(BaseModel):
    game_id: str
    box_id: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]


class PreviewResponse(BaseModel):
    is_valid: bool
    message: str
    category: str | None
    details: dict[str, Any] = Field(default_factory=dict)
    support_ratio: float | None = None
    normalized_position: tuple[float, float, float] | None = None
    normalized_orientation_wxyz: tuple[float, float, float, float] | None = None
    latest_valid_preview_action: dict[str, Any] | None = None
    snap_suggestions: dict[str, Any] = Field(default_factory=dict)
    repair_suggestions: dict[str, Any] = Field(default_factory=dict)
    game_status: Literal["in_progress", "completed", "timed_out", "no_feasible_placement"]
    termination_reason: str | None = None
    current_box_deadline: str | None = None
    density: float


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
