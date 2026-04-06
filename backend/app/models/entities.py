from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

Mode = Literal["dev", "compete_stub"]
GameStatus = Literal["in_progress", "completed", "timed_out", "no_feasible_placement"]


@dataclass(slots=True)
class Truck:
    depth: float = 2.0
    width: float = 2.6
    height: float = 2.75


@dataclass(slots=True)
class BoxSpec:
    id: str
    dimensions: tuple[float, float, float]
    weight: float

    @property
    def volume(self) -> float:
        return float(self.dimensions[0] * self.dimensions[1] * self.dimensions[2])


@dataclass(slots=True)
class CurrentBox(BoxSpec):
    pass


@dataclass(slots=True)
class PlacementAction:
    box_id: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]


@dataclass(slots=True)
class PreviewAction(PlacementAction):
    pass


@dataclass(slots=True)
class PlacedBox(BoxSpec):
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    source: Literal["manual", "timeout_auto_place", "rl_repair"] = "manual"


@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    message: str
    category: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    normalized_action: PlacementAction | None = None
    support_ratio: float | None = None


@dataclass(slots=True)
class EpisodeTimerState:
    current_box_started_at: datetime
    current_box_deadline: datetime
    timeout_seconds: float


@dataclass(slots=True)
class TerminationInfo:
    game_status: GameStatus
    termination_reason: str | None = None


@dataclass(slots=True)
class EngineConfig:
    truck: Truck = field(default_factory=Truck)
    loading_line_x: float = 0.78
    default_queue_length: int = 120
    dimension_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.26, 0.75),
        (0.26, 0.72),
        (0.22, 0.68),
    )
    weight_range: tuple[float, float] = (4.0, 40.0)
    timeout_seconds: float = 10.0
    support_threshold: float = 0.9
    geometry_epsilon: float = 1e-6
    support_plane_epsilon: float = 5e-3
    collision_epsilon: float = 1e-5
    vertical_axis_cos_tolerance: float = 0.9961946980917455
    boundary_snap_tolerance: float = 0.02
    repair_translation_step: float = 0.04
    repair_attempt_budget: int = 160
    repair_yaw_step_degrees: float = 7.5
    feasibility_xy_samples: int = 7
    feasibility_yaw_samples: int = 12


@dataclass(slots=True)
class GameState:
    game_id: str
    mode: Mode
    truck: Truck
    placed_boxes: list[PlacedBox]
    remaining_boxes: list[CurrentBox]
    current_box: CurrentBox | None
    density: float
    game_status: GameStatus
    created_at: datetime
    timer_state: EpisodeTimerState | None
    latest_preview_action: PreviewAction | None
    latest_valid_preview_action: PreviewAction | None
    termination_reason: str | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def current_box_deadline(self) -> datetime | None:
        return None if self.timer_state is None else self.timer_state.current_box_deadline

    @property
    def current_box_started_at(self) -> datetime | None:
        return None if self.timer_state is None else self.timer_state.current_box_started_at

    @property
    def timeout_seconds(self) -> float:
        return 0.0 if self.timer_state is None else self.timer_state.timeout_seconds

    @property
    def boxes_remaining(self) -> int:
        return len(self.remaining_boxes)
