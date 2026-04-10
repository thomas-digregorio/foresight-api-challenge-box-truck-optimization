from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.engine.geometry import OrientedBoxGeometry
from app.models.entities import PlacementAction

RectBounds = tuple[float, float, float, float]


@dataclass(slots=True, frozen=True)
class OrientationOption:
    index: int
    permutation: tuple[int, int, int]
    orientation_wxyz: tuple[float, float, float, float]
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    bottom_z: float
    top_z: float
    footprint_depth: float
    footprint_width: float
    height: float


@dataclass(slots=True)
class PlacedBoxView:
    id: str
    geometry: OrientedBoxGeometry
    footprint_bounds: RectBounds
    top_bounds: RectBounds | None
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    bottom_z: float
    top_z: float


@dataclass(slots=True, frozen=True)
class SupportPlane:
    index: int
    plane_id: str
    z_support: float
    supporting_box_ids: tuple[str, ...]
    support_rectangles: tuple[RectBounds, ...]
    support_area: float
    component_count: int
    max_component_area: float
    frontier_gap: float
    quality_score: float


@dataclass(slots=True, frozen=True)
class CandidatePlacement:
    group_index: int
    support_plane_id: str
    support_plane_index: int
    orientation_index: int
    anchor_style: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    support_component_bounds: RectBounds | None
    dedup_key: tuple[Any, ...]
    sort_key: tuple[Any, ...]

    def as_action(self, box_id: str) -> PlacementAction:
        return PlacementAction(
            box_id=box_id,
            position=self.position,
            orientation_wxyz=self.orientation_wxyz,
        )


@dataclass(slots=True)
class CandidateGroup:
    index: int
    support_plane: SupportPlane
    orientation: OrientationOption
    candidates: list[CandidatePlacement] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class ScoreWeights:
    alpha: float = 10.0
    beta: float = 6.0
    gamma: float = 2.0
    eta: float = 1.5
    front_gap_weight: float = 1.0
    left_gap_weight: float = 0.5
    right_gap_weight: float = 0.5
    frontier_slack_weight: float = 0.5
    future_sliver_weight: float = 0.75
    frontier_reach_weight: float = 0.5
    future_usable_area_weight: float = 1.5
    fragmentation_penalty_weight: float = 2.0
    shelf_completion_weight: float = 0.75
    flush_contact_tolerance: float = 1e-3
    floor_contact_bonus: float = 0.25


@dataclass(slots=True, frozen=True)
class ScoreBreakdown:
    total_score: float
    delta_x: float
    gap_penalty: float
    front_gap: float
    frontier_slack: float
    future_sliver_penalty: float
    fragmentation_penalty: float
    left_gap: float
    right_gap: float
    support_reward: float
    contact_reward: float
    frontier_reach_reward: float
    future_usable_area_reward: float
    shelf_completion_reward: float
    front_contact_fraction: float
    left_contact_fraction: float
    right_contact_fraction: float


@dataclass(slots=True, frozen=True)
class ProxyCandidate:
    candidate: CandidatePlacement
    score: ScoreBreakdown

    def ranking_key(self) -> tuple[Any, ...]:
        position = tuple(round(value, 12) for value in self.candidate.position)
        return (
            -round(self.score.total_score, 12),
            round(self.score.delta_x, 12),
            round(self.score.gap_penalty, 12),
            round(self.candidate.position[2], 12),
            -round(self.score.support_reward, 12),
            -round(self.score.contact_reward, 12),
            round(self.candidate.position[1], 12),
            self.candidate.orientation_index,
            position,
        )


@dataclass(slots=True)
class RankedCandidate:
    candidate: CandidatePlacement
    action: PlacementAction
    score: ScoreBreakdown
    support_ratio: float
    validation_message: str
    validation_category: str | None

    def ranking_key(self) -> tuple[Any, ...]:
        position = tuple(round(value, 12) for value in self.action.position)
        return (
            -round(self.score.total_score, 12),
            round(self.score.delta_x, 12),
            round(self.score.gap_penalty, 12),
            round(self.action.position[2], 12),
            -round(self.score.support_reward, 12),
            -round(self.score.contact_reward, 12),
            round(self.action.position[1], 12),
            self.candidate.orientation_index,
            position,
        )


@dataclass(slots=True)
class EvaluationSummary:
    ranked_candidates: list[RankedCandidate]
    generated_count: int
    validated_count: int
    valid_count: int
    parallel_used: bool
    group_count: int
    pruned_count: int = 0
    skipped_by_bound_count: int = 0
    deadline_hit: bool = False
    validation_budget_hit: bool = False
    best_proxy_candidate: CandidatePlacement | None = None
    best_proxy_score: ScoreBreakdown | None = None
    top_proxy_candidates: list[ProxyCandidate] = field(default_factory=list)
    preparation_time_ms: float = 0.0
    validation_time_ms: float = 0.0
    invalid_reason_counts: dict[str, int] = field(default_factory=dict)
    valid_counts_by_group: dict[str, int] = field(default_factory=dict)
