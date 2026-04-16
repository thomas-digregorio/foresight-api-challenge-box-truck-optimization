from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from shapely.geometry import box
from shapely.ops import unary_union

from app.engine.geometry import compute_box_geometry
from app.models.entities import CurrentBox, EngineConfig, GameState, PlacedBox, Truck

from app.agents.extreme_point.types import PlacedBoxView, RectBounds


def _coerce_truck(raw_state: dict[str, Any], fallback_truck: Truck | None) -> Truck:
    payload = raw_state.get("truck")
    if payload is not None:
        return Truck(
            depth=float(payload["depth"]),
            width=float(payload["width"]),
            height=float(payload["height"]),
        )
    if fallback_truck is not None:
        return Truck(depth=fallback_truck.depth, width=fallback_truck.width, height=fallback_truck.height)
    return Truck()


def _box_bounds(geometry) -> RectBounds:
    if geometry.footprint_polygon is not None:
        min_x, min_y, max_x, max_y = geometry.footprint_polygon.bounds
        return (float(min_x), float(min_y), float(max_x), float(max_y))
    return (
        float(geometry.aabb_min[0]),
        float(geometry.aabb_min[1]),
        float(geometry.aabb_max[0]),
        float(geometry.aabb_max[1]),
    )


@dataclass(slots=True)
class DecisionStateView:
    raw_state: dict[str, Any]
    config: EngineConfig
    truck: Truck
    game_state: GameState
    current_box: CurrentBox | None
    placed_boxes: tuple[PlacedBoxView, ...]
    placed_volume: float
    current_max_x: float
    obstacle_rectangles: tuple[RectBounds, ...]
    boxes_remaining: int
    _overlap_rectangles_cache: dict[tuple[float, float], tuple[RectBounds, ...]] = field(default_factory=dict)
    _support_surface_rectangles_cache: dict[float, tuple[RectBounds, ...]] = field(default_factory=dict)
    _resting_rectangles_cache: dict[float, tuple[RectBounds, ...]] = field(default_factory=dict)
    _free_rectangle_cache: dict[tuple[str, float, float], tuple[RectBounds, ...]] = field(default_factory=dict)

    @classmethod
    def from_raw_state(
        cls,
        raw_state: dict[str, Any],
        *,
        config: EngineConfig,
        fallback_truck: Truck | None = None,
    ) -> "DecisionStateView":
        truck = _coerce_truck(raw_state, fallback_truck)
        current_box_payload = raw_state.get("current_box")
        current_box = None
        if current_box_payload is not None:
            current_box = CurrentBox(
                id=str(current_box_payload["id"]),
                dimensions=tuple(float(value) for value in current_box_payload["dimensions"]),
                weight=float(current_box_payload.get("weight", 0.0)),
            )

        placed_boxes: list[PlacedBox] = []
        placed_views: list[PlacedBoxView] = []
        for payload in raw_state.get("placed_boxes", []):
            placed_box = PlacedBox(
                id=str(payload["id"]),
                dimensions=tuple(float(value) for value in payload["dimensions"]),
                weight=float(payload.get("weight", 0.0)),
                position=tuple(float(value) for value in payload["position"]),
                orientation_wxyz=tuple(float(value) for value in payload["orientation_wxyz"]),
            )
            geometry = compute_box_geometry(
                placed_box.position,
                placed_box.dimensions,
                placed_box.orientation_wxyz,
                vertical_axis_cos_tolerance=config.vertical_axis_cos_tolerance,
            )
            top_bounds = None
            if geometry.top_polygon is not None:
                top_bounds = tuple(float(value) for value in geometry.top_polygon.bounds)
            footprint_bounds = _box_bounds(geometry)
            placed_boxes.append(placed_box)
            placed_views.append(
                PlacedBoxView(
                    id=placed_box.id,
                    geometry=geometry,
                    footprint_bounds=footprint_bounds,
                    top_bounds=top_bounds,
                    min_x=float(geometry.aabb_min[0]),
                    max_x=float(geometry.aabb_max[0]),
                    min_y=float(geometry.aabb_min[1]),
                    max_y=float(geometry.aabb_max[1]),
                    bottom_z=float(geometry.bottom_z),
                    top_z=float(geometry.top_z),
                )
            )

        game_state = GameState(
            game_id=str(raw_state.get("game_id", "agent-state")),
            mode=str(raw_state.get("mode", "dev")),  # type: ignore[arg-type]
            truck=truck,
            placed_boxes=placed_boxes,
            remaining_boxes=[],
            current_box=current_box,
            density=float(raw_state.get("density", 0.0)),
            game_status=str(raw_state.get("game_status", "in_progress")),  # type: ignore[arg-type]
            created_at=datetime.now(UTC),
            timer_state=None,
            latest_preview_action=None,
            latest_valid_preview_action=None,
            termination_reason=raw_state.get("termination_reason"),
            metadata={},
        )
        obstacle_rectangles = tuple(box.footprint_bounds for box in placed_views)
        current_max_x = max((box.max_x for box in placed_views), default=0.0)
        placed_volume = sum(box.volume for box in placed_boxes)
        return cls(
            raw_state=raw_state,
            config=config,
            truck=truck,
            game_state=game_state,
            current_box=current_box,
            placed_boxes=tuple(placed_views),
            placed_volume=float(placed_volume),
            current_max_x=float(current_max_x),
            obstacle_rectangles=obstacle_rectangles,
            boxes_remaining=int(raw_state.get("boxes_remaining", 0)),
        )

    def obstacle_rectangles_for_z_range(self, bottom_z: float, top_z: float) -> tuple[RectBounds, ...]:
        key = (round(bottom_z, 6), round(top_z, 6))
        cached = self._overlap_rectangles_cache.get(key)
        if cached is not None:
            return cached
        epsilon = self.config.geometry_epsilon
        overlapping = tuple(
            box.footprint_bounds
            for box in self.placed_boxes
            if box.top_z > bottom_z + epsilon and box.bottom_z < top_z - epsilon
        )
        self._overlap_rectangles_cache[key] = overlapping
        return overlapping

    def support_surface_rectangles(self, z_support: float) -> tuple[RectBounds, ...]:
        key = round(z_support, 6)
        cached = self._support_surface_rectangles_cache.get(key)
        if cached is not None:
            return cached
        surfaces = tuple(
            box.top_bounds
            for box in self.placed_boxes
            if box.top_bounds is not None and abs(box.top_z - z_support) <= self.config.support_plane_epsilon
        )
        self._support_surface_rectangles_cache[key] = surfaces
        return surfaces

    def resting_rectangles_on_plane(self, z_support: float) -> tuple[RectBounds, ...]:
        key = round(z_support, 6)
        cached = self._resting_rectangles_cache.get(key)
        if cached is not None:
            return cached
        rectangles = tuple(
            box.footprint_bounds
            for box in self.placed_boxes
            if abs(box.bottom_z - z_support) <= self.config.support_plane_epsilon
        )
        self._resting_rectangles_cache[key] = rectangles
        return rectangles

    def free_rectangles(
        self,
        *,
        cache_key: str,
        base_rectangles: tuple[RectBounds, ...],
        fallback_bounds: RectBounds,
        bottom_z: float,
        top_z: float,
    ) -> tuple[RectBounds, ...]:
        key = (cache_key, round(bottom_z, 6), round(top_z, 6))
        cached = self._free_rectangle_cache.get(key)
        if cached is not None:
            return cached

        rectangles = base_rectangles or (fallback_bounds,)
        base_polygons = [box(*rectangle) for rectangle in rectangles]
        base_union = unary_union(base_polygons)
        obstacle_rectangles = self.obstacle_rectangles_for_z_range(bottom_z, top_z)
        if obstacle_rectangles:
            obstacle_union = unary_union([box(*rectangle) for rectangle in obstacle_rectangles])
            free_geometry = base_union.difference(obstacle_union)
        else:
            free_geometry = base_union

        if free_geometry.is_empty:
            self._free_rectangle_cache[key] = ()
            return ()

        x_values = {fallback_bounds[0], fallback_bounds[2]}
        y_values = {fallback_bounds[1], fallback_bounds[3]}
        for min_x, min_y, max_x, max_y in rectangles:
            x_values.update((min_x, max_x))
            y_values.update((min_y, max_y))
        for min_x, min_y, max_x, max_y in obstacle_rectangles:
            x_values.update((min_x, max_x))
            y_values.update((min_y, max_y))
        x_coords = sorted(round(value, 6) for value in x_values)
        y_coords = sorted(round(value, 6) for value in y_values)
        if len(x_coords) < 2 or len(y_coords) < 2:
            self._free_rectangle_cache[key] = ()
            return ()

        epsilon = self.config.geometry_epsilon
        occupied: dict[tuple[int, int], bool] = {}
        for x_index, (x0, x1) in enumerate(zip(x_coords, x_coords[1:])):
            if x1 - x0 <= epsilon:
                continue
            for y_index, (y0, y1) in enumerate(zip(y_coords, y_coords[1:])):
                if y1 - y0 <= epsilon:
                    continue
                cell = box(x0, y0, x1, y1)
                intersection_area = float(free_geometry.intersection(cell).area)
                if intersection_area + epsilon >= float(cell.area):
                    occupied[(x_index, y_index)] = True

        row_runs: dict[int, list[tuple[int, int]]] = {}
        max_y_index = max(0, len(y_coords) - 1)
        for x_index in range(max(0, len(x_coords) - 1)):
            start_y: int | None = None
            runs: list[tuple[int, int]] = []
            for y_index in range(max_y_index):
                filled = occupied.get((x_index, y_index), False)
                if filled and start_y is None:
                    start_y = y_index
                if not filled and start_y is not None:
                    runs.append((start_y, y_index))
                    start_y = None
            if start_y is not None:
                runs.append((start_y, max_y_index))
            if runs:
                row_runs[x_index] = runs

        active: dict[tuple[int, int], tuple[int, int]] = {}
        free_rectangles: list[RectBounds] = []
        max_x_index = max(0, len(x_coords) - 1)
        for x_index in range(max_x_index):
            current_runs = set(row_runs.get(x_index, []))
            next_active: dict[tuple[int, int], tuple[int, int]] = {}
            for run in current_runs:
                if run in active:
                    next_active[run] = (active[run][0], x_index + 1)
                else:
                    next_active[run] = (x_index, x_index + 1)
            for run, (start_x, end_x) in active.items():
                if run in current_runs:
                    continue
                free_rectangles.append(
                    (
                        float(x_coords[start_x]),
                        float(y_coords[run[0]]),
                        float(x_coords[end_x]),
                        float(y_coords[run[1]]),
                    )
                )
            active = next_active

        for run, (start_x, end_x) in active.items():
            free_rectangles.append(
                (
                    float(x_coords[start_x]),
                    float(y_coords[run[0]]),
                    float(x_coords[end_x]),
                    float(y_coords[run[1]]),
                )
            )

        deduped = tuple(
            sorted(
                {
                    tuple(round(value, 6) for value in rectangle): rectangle
                    for rectangle in free_rectangles
                    if rectangle[2] - rectangle[0] > epsilon and rectangle[3] - rectangle[1] > epsilon
                }.values()
            )
        )
        self._free_rectangle_cache[key] = deduped
        return deduped
