from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np
from scipy.spatial.transform import Rotation
from shapely.ops import unary_union

from app.core.exceptions import StateConflictError, ValidationError
from app.engine.geometry import (
    compute_box_geometry,
    corners_within_truck,
    normalize_quaternion,
    obb_intersects,
    rotation_from_wxyz,
    stable_orientation_quaternions,
)
from app.engine.stability import NoOpDeterministicStabilityBackend, StabilityBackend
from app.models.entities import (
    BoxSpec,
    CurrentBox,
    EngineConfig,
    EpisodeTimerState,
    GameState,
    PlacementAction,
    PlacedBox,
    PreviewAction,
    ValidationResult,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class TruckPackingEngine:
    def __init__(
        self,
        config: EngineConfig | None = None,
        *,
        stability_backend: StabilityBackend | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.stability_backend = stability_backend or NoOpDeterministicStabilityBackend()
        self._clock = clock or utc_now

    def start_episode(
        self,
        *,
        mode: str,
        seed: int | None,
        queue_length: int | None = None,
    ) -> GameState:
        boxes = self._generate_box_queue(seed=seed, queue_length=queue_length)
        current_box = boxes.pop(0) if boxes else None
        now = self._clock()
        state = GameState(
            game_id=str(uuid4()),
            mode=mode,  # type: ignore[arg-type]
            truck=self.config.truck,
            placed_boxes=[],
            remaining_boxes=boxes,
            current_box=current_box,
            density=0.0,
            game_status="in_progress",
            created_at=now,
            timer_state=self._build_timer(now) if current_box is not None else None,
            latest_preview_action=None,
            latest_valid_preview_action=None,
            termination_reason=None,
            seed=seed,
            metadata={},
        )
        if state.current_box is not None and not self.has_feasible_placement(state):
            state.game_status = "no_feasible_placement"
            state.termination_reason = "no_feasible_placement"
            state.timer_state = None
        return state

    def stop_episode(self, state: GameState) -> GameState:
        self.handle_timeout_if_needed(state)
        state.game_status = "completed"
        state.termination_reason = "player_stop"
        state.timer_state = None
        return state

    def handle_timeout_if_needed(self, state: GameState) -> bool:
        if state.game_status != "in_progress" or state.current_box is None or state.timer_state is None:
            return False
        now = self._clock()
        if now < state.timer_state.current_box_deadline:
            return False
        latest_valid = state.latest_valid_preview_action
        if latest_valid is not None and latest_valid.box_id == state.current_box.id:
            self.commit_place_action(state, latest_valid, source="timeout_auto_place")
            return True
        state.game_status = "timed_out"
        state.termination_reason = "timeout_no_valid_preview"
        state.timer_state = None
        return True

    def update_preview(self, state: GameState, action: PreviewAction) -> ValidationResult:
        self.handle_timeout_if_needed(state)
        if state.game_status != "in_progress":
            raise StateConflictError(
                "Preview rejected because the game is no longer active.",
                category="game_completed",
                details={"game_status": state.game_status},
            )
        validation = self._validate_action(state, action)
        normalized_preview = PreviewAction(
            box_id=validation.normalized_action.box_id,
            position=validation.normalized_action.position,
            orientation_wxyz=validation.normalized_action.orientation_wxyz,
        )
        state.latest_preview_action = normalized_preview
        if validation.is_valid:
            state.latest_valid_preview_action = normalized_preview
        return validation

    def validate_place_action(self, state: GameState, action: PlacementAction) -> ValidationResult:
        self.handle_timeout_if_needed(state)
        return self._validate_action(state, action)

    def commit_place_action(
        self,
        state: GameState,
        action: PlacementAction,
        *,
        source: str = "manual",
    ) -> GameState:
        validation = self._validate_action(state, action)
        if not validation.is_valid or validation.normalized_action is None:
            raise ValidationError(
                validation.message,
                category=validation.category or "validation_error",
                details=validation.details,
            )
        current_box = state.current_box
        if current_box is None:
            raise StateConflictError(
                "No current box is available to place.",
                category="game_completed",
                details={"game_status": state.game_status},
            )
        state.placed_boxes.append(
            PlacedBox(
                id=current_box.id,
                dimensions=current_box.dimensions,
                weight=current_box.weight,
                position=validation.normalized_action.position,
                orientation_wxyz=validation.normalized_action.orientation_wxyz,
                source=source,  # type: ignore[arg-type]
            )
        )
        state.density = self.compute_density(state)
        state.latest_preview_action = None
        state.latest_valid_preview_action = None
        self.advance_to_next_box(state)
        if state.current_box is None:
            state.game_status = "completed"
            state.termination_reason = None
            state.timer_state = None
            return state
        if not self.has_feasible_placement(state):
            state.game_status = "no_feasible_placement"
            state.termination_reason = "no_feasible_placement"
            state.timer_state = None
        return state

    def advance_to_next_box(self, state: GameState) -> None:
        if state.remaining_boxes:
            state.current_box = state.remaining_boxes.pop(0)
            state.timer_state = self._build_timer(self._clock())
        else:
            state.current_box = None
            state.timer_state = None

    def compute_density(self, state: GameState) -> float:
        if not state.placed_boxes:
            return 0.0
        total_volume = sum(box.volume for box in state.placed_boxes)
        max_x_reached = max(
            compute_box_geometry(
                box.position,
                box.dimensions,
                box.orientation_wxyz,
                vertical_axis_cos_tolerance=self.config.vertical_axis_cos_tolerance,
            ).aabb_max[0]
            for box in state.placed_boxes
        )
        if max_x_reached <= self.config.geometry_epsilon:
            return 0.0
        return float(total_volume / (max_x_reached * state.truck.width * state.truck.height))

    def has_feasible_placement(self, state: GameState) -> bool:
        return self.find_any_valid_action(state) is not None

    def find_any_valid_action(self, state: GameState) -> PlacementAction | None:
        current_box = state.current_box
        if current_box is None:
            return None
        support_planes = self._support_plane_heights(state)
        orientation_quats = stable_orientation_quaternions(self.config.feasibility_yaw_samples)
        for orientation in orientation_quats:
            reference = compute_box_geometry(
                (0.0, 0.0, 0.0),
                current_box.dimensions,
                orientation,
                vertical_axis_cos_tolerance=self.config.vertical_axis_cos_tolerance,
            )
            if reference.footprint_polygon is None:
                continue
            min_x, min_y, max_x, max_y = reference.footprint_polygon.bounds
            x_min = -min_x + self.config.geometry_epsilon
            x_max = min(
                state.truck.depth - max_x - self.config.geometry_epsilon,
                self.config.loading_line_x - max_x - self.config.geometry_epsilon,
            )
            y_min = -min_y + self.config.geometry_epsilon
            y_max = state.truck.width - max_y - self.config.geometry_epsilon
            if x_min > x_max or y_min > y_max:
                continue
            x_values = np.linspace(x_min, x_max, num=self.config.feasibility_xy_samples)
            y_values = np.linspace(y_min, y_max, num=self.config.feasibility_xy_samples)
            bottom_relative = reference.bottom_z
            for plane_height in support_planes:
                z = float(plane_height - bottom_relative)
                for x in x_values:
                    for y in y_values:
                        action = PlacementAction(
                            box_id=current_box.id,
                            position=(float(x), float(y), z),
                            orientation_wxyz=orientation,
                        )
                        if self._validate_action(state, action).is_valid:
                            return action
        return None

    def find_valid_action_near(self, state: GameState, action: PlacementAction) -> PlacementAction | None:
        current_box = state.current_box
        if current_box is None:
            return None
        exact_result = self._validate_action(state, action)
        if exact_result.is_valid and exact_result.normalized_action is not None:
            return exact_result.normalized_action

        normalized_orientation = exact_result.normalized_action.orientation_wxyz if exact_result.normalized_action else normalize_quaternion(action.orientation_wxyz)
        base_position = np.asarray(action.position, dtype=float)
        support_planes = self._support_plane_heights(state)
        candidate_orientations = self._nearby_orientations(normalized_orientation)
        translation_values = range(-2, 3)
        attempts = 0

        for orientation in candidate_orientations:
            reference = compute_box_geometry(
                (0.0, 0.0, 0.0),
                current_box.dimensions,
                orientation,
                vertical_axis_cos_tolerance=self.config.vertical_axis_cos_tolerance,
            )
            candidate_planes = [reference.bottom_z + base_position[2]]
            candidate_planes.extend(support_planes)
            ordered_planes = sorted(set(float(value) for value in candidate_planes), key=lambda plane: abs(plane - base_position[2]))
            for plane_height in ordered_planes:
                center_z = float(plane_height - reference.bottom_z)
                for dx in translation_values:
                    for dy in translation_values:
                        attempts += 1
                        if attempts > self.config.repair_attempt_budget:
                            return None
                        position = (
                            float(np.clip(base_position[0] + dx * self.config.repair_translation_step, 0.0, state.truck.depth)),
                            float(np.clip(base_position[1] + dy * self.config.repair_translation_step, 0.0, state.truck.width)),
                            center_z,
                        )
                        candidate = PlacementAction(
                            box_id=current_box.id,
                            position=position,
                            orientation_wxyz=orientation,
                        )
                        result = self._validate_action(state, candidate)
                        if result.is_valid and result.normalized_action is not None:
                            return result.normalized_action
        return None

    def find_valid_action_at_current_xy(self, state: GameState, action: PlacementAction) -> PlacementAction | None:
        current_box = state.current_box
        if current_box is None:
            return None
        normalized_orientation = normalize_quaternion(action.orientation_wxyz)
        support_planes = sorted(self._support_plane_heights(state), reverse=True)

        reference = compute_box_geometry(
            (0.0, 0.0, 0.0),
            current_box.dimensions,
            normalized_orientation,
            vertical_axis_cos_tolerance=self.config.vertical_axis_cos_tolerance,
        )
        for plane_height in support_planes:
            position = (
                float(action.position[0]),
                float(action.position[1]),
                float(plane_height - reference.bottom_z),
            )
            candidate = PlacementAction(
                box_id=current_box.id,
                position=position,
                orientation_wxyz=normalized_orientation,
            )
            result = self._validate_action(state, candidate)
            if result.is_valid and result.normalized_action is not None:
                return result.normalized_action
        return None

    def snap_to_floor(self, box: BoxSpec, action: PlacementAction) -> PlacementAction:
        geometry = compute_box_geometry(
            action.position,
            box.dimensions,
            action.orientation_wxyz,
            vertical_axis_cos_tolerance=self.config.vertical_axis_cos_tolerance,
        )
        dz = -geometry.bottom_z
        position = (
            float(action.position[0]),
            float(action.position[1]),
            float(action.position[2] + dz),
        )
        return PlacementAction(box_id=action.box_id, position=position, orientation_wxyz=normalize_quaternion(action.orientation_wxyz))

    def snap_to_support_plane(self, state: GameState, box: BoxSpec, action: PlacementAction) -> PlacementAction:
        geometry = compute_box_geometry(
            action.position,
            box.dimensions,
            action.orientation_wxyz,
            vertical_axis_cos_tolerance=self.config.vertical_axis_cos_tolerance,
        )
        plane_heights = self._support_plane_heights(state)
        current_bottom = geometry.bottom_z
        target_plane = min(plane_heights, key=lambda height: abs(height - current_bottom))
        center_z = float(action.position[2] + (target_plane - current_bottom))
        return PlacementAction(
            box_id=action.box_id,
            position=(float(action.position[0]), float(action.position[1]), center_z),
            orientation_wxyz=normalize_quaternion(action.orientation_wxyz),
        )

    def _validate_action(self, state: GameState, action: PlacementAction) -> ValidationResult:
        if state.game_status != "in_progress":
            raise StateConflictError(
                "Game is no longer active.",
                category="game_completed",
                details={"game_status": state.game_status},
            )
        current_box = state.current_box
        if current_box is None:
            raise StateConflictError(
                "No current box is available.",
                category="game_completed",
                details={"game_status": state.game_status},
            )
        if action.box_id != current_box.id:
            raise ValidationError(
                "The action box_id does not match the current box.",
                category="invalid_box_id",
                details={"expected_box_id": current_box.id, "received_box_id": action.box_id},
            )
        try:
            normalized_quaternion = normalize_quaternion(action.orientation_wxyz)
        except ValueError as exc:
            raise ValidationError(
                "Quaternion must be non-zero.",
                category="invalid_quaternion",
                details={"orientation_wxyz": list(action.orientation_wxyz)},
            ) from exc

        normalized_action = PlacementAction(
            box_id=action.box_id,
            position=tuple(float(value) for value in action.position),
            orientation_wxyz=normalized_quaternion,
        )
        geometry = compute_box_geometry(
            normalized_action.position,
            current_box.dimensions,
            normalized_quaternion,
            vertical_axis_cos_tolerance=self.config.vertical_axis_cos_tolerance,
        )
        in_bounds, bounds_details = corners_within_truck(state.truck, geometry, epsilon=self.config.geometry_epsilon)
        if not in_bounds:
            return ValidationResult(
                is_valid=False,
                message="Placement would leave the truck bounds.",
                category="out_of_bounds",
                details=bounds_details,
                normalized_action=normalized_action,
            )

        if geometry.aabb_max[0] > self.config.loading_line_x + self.config.geometry_epsilon:
            return ValidationResult(
                is_valid=False,
                message="Placement crosses the fixed loading line.",
                category="loading_line_crossed",
                details={
                    "loading_line_x": self.config.loading_line_x,
                    "candidate_max_x": float(geometry.aabb_max[0]),
                },
                normalized_action=normalized_action,
            )

        if not geometry.is_gravity_compatible or geometry.footprint_polygon is None:
            return ValidationResult(
                is_valid=False,
                message="Deterministic v1 only accepts gravity-compatible resting poses with a horizontal support face.",
                category="insufficient_support",
                details={"vertical_alignment": geometry.vertical_alignment},
                normalized_action=normalized_action,
            )

        for placed_box in state.placed_boxes:
            placed_geometry = compute_box_geometry(
                placed_box.position,
                placed_box.dimensions,
                placed_box.orientation_wxyz,
                vertical_axis_cos_tolerance=self.config.vertical_axis_cos_tolerance,
            )
            if obb_intersects(geometry, placed_geometry, epsilon=self.config.collision_epsilon):
                return ValidationResult(
                    is_valid=False,
                    message="Placement overlaps an existing box.",
                    category="overlap",
                    details={"overlap_with_box_id": placed_box.id},
                    normalized_action=normalized_action,
                )

        support_ratio, support_details = self._support_ratio(state, geometry)
        if support_ratio < self.config.support_threshold:
            return ValidationResult(
                is_valid=False,
                message="Placement does not have enough support area.",
                category="insufficient_support",
                details=support_details,
                normalized_action=normalized_action,
                support_ratio=support_ratio,
            )

        backend_ok, backend_message, backend_details = self.stability_backend.validate_candidate(
            state,
            {"support_ratio": support_ratio, "geometry": geometry},
        )
        if not backend_ok:
            return ValidationResult(
                is_valid=False,
                message=backend_message or "Placement rejected by the stability backend.",
                category="insufficient_support",
                details=backend_details,
                normalized_action=normalized_action,
                support_ratio=support_ratio,
            )

        return ValidationResult(
            is_valid=True,
            message="Placement is valid.",
            category=None,
            details=support_details,
            normalized_action=normalized_action,
            support_ratio=support_ratio,
        )

    def _support_ratio(self, state: GameState, geometry: Any) -> tuple[float, dict[str, Any]]:
        footprint_area = float(geometry.footprint_polygon.area) if geometry.footprint_polygon is not None else 0.0
        if footprint_area <= self.config.geometry_epsilon:
            return 0.0, {"candidate_footprint_area": footprint_area}
        if abs(geometry.bottom_z) <= self.config.support_plane_epsilon:
            return 1.0, {
                "support_type": "floor",
                "candidate_footprint_area": footprint_area,
                "support_height": 0.0,
                "support_overlap_area": footprint_area,
            }
        support_polygons = []
        support_ids: list[str] = []
        for placed_box in state.placed_boxes:
            placed_geometry = compute_box_geometry(
                placed_box.position,
                placed_box.dimensions,
                placed_box.orientation_wxyz,
                vertical_axis_cos_tolerance=self.config.vertical_axis_cos_tolerance,
            )
            if (
                placed_geometry.is_gravity_compatible
                and placed_geometry.top_polygon is not None
                and abs(placed_geometry.top_z - geometry.bottom_z) <= self.config.support_plane_epsilon
            ):
                support_polygons.append(placed_geometry.top_polygon)
                support_ids.append(placed_box.id)
        if not support_polygons:
            return 0.0, {
                "support_type": "none",
                "candidate_footprint_area": footprint_area,
                "support_height": geometry.bottom_z,
                "support_overlap_area": 0.0,
            }
        support_union = unary_union(support_polygons)
        overlap_area = float(geometry.footprint_polygon.intersection(support_union).area)
        ratio = overlap_area / footprint_area if footprint_area > 0.0 else 0.0
        return ratio, {
            "support_type": "box_top_faces",
            "candidate_footprint_area": footprint_area,
            "support_overlap_area": overlap_area,
            "support_height": geometry.bottom_z,
            "supporting_box_ids": support_ids,
        }

    def _support_plane_heights(self, state: GameState) -> list[float]:
        heights = {0.0}
        for placed_box in state.placed_boxes:
            geometry = compute_box_geometry(
                placed_box.position,
                placed_box.dimensions,
                placed_box.orientation_wxyz,
                vertical_axis_cos_tolerance=self.config.vertical_axis_cos_tolerance,
            )
            if geometry.is_gravity_compatible:
                heights.add(round(float(geometry.top_z), 6))
        return sorted(heights)

    def _nearby_orientations(self, normalized_orientation: tuple[float, float, float, float]) -> list[tuple[float, float, float, float]]:
        candidates = [normalized_orientation]
        base_rotation = rotation_from_wxyz(normalized_orientation)
        yaw_step = np.deg2rad(self.config.repair_yaw_step_degrees)
        for step in (-2, -1, 1, 2):
            rotated = Rotation.from_rotvec(np.array([0.0, 0.0, yaw_step * step], dtype=float)) * base_rotation
            quat = normalize_quaternion((rotated.as_quat()[3], rotated.as_quat()[0], rotated.as_quat()[1], rotated.as_quat()[2]))
            candidates.append(quat)
        for quat in stable_orientation_quaternions(4):
            if not any(np.allclose(quat, existing, atol=1e-6) or np.allclose(np.negative(quat), existing, atol=1e-6) for existing in candidates):
                candidates.append(quat)
        return candidates

    def _build_timer(self, now: datetime) -> EpisodeTimerState:
        return EpisodeTimerState(
            current_box_started_at=now,
            current_box_deadline=now + timedelta(seconds=self.config.timeout_seconds),
            timeout_seconds=self.config.timeout_seconds,
        )

    def _generate_box_queue(self, *, seed: int | None, queue_length: int | None) -> list[CurrentBox]:
        rng = np.random.default_rng(seed)
        count = queue_length or self.config.default_queue_length
        dims_low = np.asarray([range_pair[0] for range_pair in self.config.dimension_ranges], dtype=float)
        dims_high = np.asarray([range_pair[1] for range_pair in self.config.dimension_ranges], dtype=float)
        weight_low, weight_high = self.config.weight_range
        boxes: list[CurrentBox] = []
        for index in range(count):
            dims = tuple(float(value) for value in rng.uniform(dims_low, dims_high))
            weight = float(rng.uniform(weight_low, weight_high))
            boxes.append(CurrentBox(id=f"box-{index + 1}", dimensions=dims, weight=weight))
        rng.shuffle(boxes)
        return boxes
