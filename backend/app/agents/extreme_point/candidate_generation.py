from __future__ import annotations

from time import perf_counter

from app.agents.extreme_point.anchors import ANCHOR_STYLES, candidate_dedup_key, generate_edge_anchors
from app.agents.extreme_point.state_view import DecisionStateView
from app.agents.extreme_point.types import CandidateGroup, CandidatePlacement, HeuristicProfile, OrientationOption, RectBounds, SupportPlane


def _rectangles_connected(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
    *,
    epsilon: float,
) -> bool:
    return bool(
        lhs[2] >= rhs[0] - epsilon
        and rhs[2] >= lhs[0] - epsilon
        and lhs[3] >= rhs[1] - epsilon
        and rhs[3] >= lhs[1] - epsilon
    )


def _rectangles_overlap(
    lhs: RectBounds,
    rhs: RectBounds,
    *,
    epsilon: float,
) -> bool:
    return bool(
        lhs[2] > rhs[0] + epsilon
        and rhs[2] > lhs[0] + epsilon
        and lhs[3] > rhs[1] + epsilon
        and rhs[3] > lhs[1] + epsilon
    )


def _support_anchor_components(
    support_plane: SupportPlane,
    *,
    epsilon: float,
) -> tuple[tuple[tuple[float, float, float, float], ...], ...]:
    if not support_plane.support_rectangles:
        return ((),)
    remaining = list(support_plane.support_rectangles)
    components: list[tuple[tuple[float, float, float, float], ...]] = []
    while remaining:
        seed = remaining.pop(0)
        component = [seed]
        changed = True
        while changed:
            changed = False
            next_remaining: list[tuple[float, float, float, float]] = []
            for rectangle in remaining:
                if any(_rectangles_connected(existing, rectangle, epsilon=epsilon) for existing in component):
                    component.append(rectangle)
                    changed = True
                else:
                    next_remaining.append(rectangle)
            remaining = next_remaining
        components.append(tuple(sorted(component)))
    components.sort()
    return tuple(components)


def _merged_intervals(
    intervals: tuple[tuple[float, float], ...],
    *,
    epsilon: float,
) -> tuple[tuple[float, float], ...]:
    if not intervals:
        return ()
    ordered = sorted(intervals)
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + epsilon:
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([start, end])
    return tuple((float(start), float(end)) for start, end in merged)


def _center_x_from_anchor(anchor_value: float, orientation: OrientationOption, style: str) -> float:
    if style.startswith("center_x"):
        return float(anchor_value)
    if style.startswith("min_x"):
        return float(anchor_value - orientation.min_x)
    return float(anchor_value - orientation.max_x)


def _center_y_from_anchor(anchor_value: float, orientation: OrientationOption, style: str) -> float:
    if style.endswith("center_y"):
        return float(anchor_value)
    if style.endswith("min_y"):
        return float(anchor_value - orientation.min_y)
    return float(anchor_value - orientation.max_y)


def _candidate_within_truck(view: DecisionStateView, orientation: OrientationOption, center_x: float, center_y: float) -> bool:
    min_x = center_x + orientation.min_x
    max_x = center_x + orientation.max_x
    min_y = center_y + orientation.min_y
    max_y = center_y + orientation.max_y
    epsilon = view.config.geometry_epsilon
    return bool(
        min_x >= -epsilon
        and max_x <= view.truck.depth + epsilon
        and min_y >= -epsilon
        and max_y <= view.truck.width + epsilon
    )


def _candidate_within_bounds(
    view: DecisionStateView,
    orientation: OrientationOption,
    center_x: float,
    center_y: float,
    bounds: tuple[float, float, float, float],
) -> bool:
    min_x = center_x + orientation.min_x
    max_x = center_x + orientation.max_x
    min_y = center_y + orientation.min_y
    max_y = center_y + orientation.max_y
    epsilon = view.config.geometry_epsilon
    return bool(
        min_x >= bounds[0] - epsilon
        and max_x <= bounds[2] + epsilon
        and min_y >= bounds[1] - epsilon
        and max_y <= bounds[3] + epsilon
    )


def _support_area_sufficient(
    view: DecisionStateView,
    support_plane: SupportPlane,
    orientation: OrientationOption,
    center_x: float,
    center_y: float,
) -> bool:
    if not support_plane.support_rectangles:
        return True
    min_x = center_x + orientation.min_x
    max_x = center_x + orientation.max_x
    min_y = center_y + orientation.min_y
    max_y = center_y + orientation.max_y
    overlap_area = 0.0
    for rect_min_x, rect_min_y, rect_max_x, rect_max_y in support_plane.support_rectangles:
        overlap_x = max(0.0, min(max_x, rect_max_x) - max(min_x, rect_min_x))
        overlap_y = max(0.0, min(max_y, rect_max_y) - max(min_y, rect_min_y))
        overlap_area += overlap_x * overlap_y
    footprint_area = max(view.config.geometry_epsilon, orientation.footprint_depth * orientation.footprint_width)
    required_support_area = footprint_area * view.config.support_threshold
    return overlap_area + view.config.geometry_epsilon >= required_support_area


def _component_bounds(
    component_rectangles: tuple[tuple[float, float, float, float], ...],
    *,
    truck_depth: float,
    truck_width: float,
) -> tuple[float, float, float, float]:
    if not component_rectangles:
        return (0.0, 0.0, float(truck_depth), float(truck_width))
    min_x = min(rectangle[0] for rectangle in component_rectangles)
    min_y = min(rectangle[1] for rectangle in component_rectangles)
    max_x = max(rectangle[2] for rectangle in component_rectangles)
    max_y = max(rectangle[3] for rectangle in component_rectangles)
    return (float(min_x), float(min_y), float(max_x), float(max_y))


def _support_components(
    view: DecisionStateView,
    support_plane: SupportPlane,
) -> tuple[tuple[tuple[float, float, float, float], ...], ...]:
    if not support_plane.support_rectangles:
        return (((0.0, 0.0, float(view.truck.depth), float(view.truck.width)),),)
    return _support_anchor_components(
        support_plane,
        epsilon=view.config.geometry_epsilon,
    )


def _component_boxes_on_plane(
    view: DecisionStateView,
    support_plane: SupportPlane,
    component_bounds: RectBounds,
) -> tuple[RectBounds, ...]:
    epsilon = view.config.geometry_epsilon
    return tuple(
        rectangle
        for rectangle in view.resting_rectangles_on_plane(support_plane.z_support)
        if _rectangles_overlap(rectangle, component_bounds, epsilon=epsilon)
    )


def _component_frontier_x(
    component_boxes: tuple[RectBounds, ...],
    *,
    default_frontier_x: float,
) -> float:
    return max((rectangle[2] for rectangle in component_boxes), default=float(default_frontier_x))


def _should_enable_center_y(
    view: DecisionStateView,
    orientation: OrientationOption,
    component_bounds: tuple[float, float, float, float],
) -> bool:
    component_width = component_bounds[3] - component_bounds[1]
    width_slack = component_width - orientation.footprint_width
    return bool(width_slack <= float(view.config.dimension_ranges[1][0]) + view.config.geometry_epsilon)


def _should_enable_center_x(
    view: DecisionStateView,
    orientation: OrientationOption,
    component_bounds: tuple[float, float, float, float],
) -> bool:
    component_depth = component_bounds[2] - component_bounds[0]
    depth_slack = component_depth - orientation.footprint_depth
    frontier_depth = max(0.0, component_bounds[2] - max(component_bounds[0], view.current_max_x))
    return bool(
        depth_slack <= float(view.config.dimension_ranges[0][0]) + view.config.geometry_epsilon
        or frontier_depth <= orientation.footprint_depth + float(view.config.dimension_ranges[0][0]) + view.config.geometry_epsilon
    )


def _allowed_anchor_styles(
    view: DecisionStateView,
    orientation: OrientationOption,
    component_bounds: tuple[float, float, float, float],
    *,
    heuristic_profile: HeuristicProfile,
    allow_secondary_widening: bool,
) -> tuple[str, ...]:
    primary_styles: list[str] = [
        "min_x_min_y",
        "min_x_max_y",
        "max_x_min_y",
        "max_x_max_y",
    ]
    if heuristic_profile == "future_aware":
        return ANCHOR_STYLES
    if not allow_secondary_widening:
        return tuple(primary_styles)
    if _should_enable_center_y(view, orientation, component_bounds):
        primary_styles.extend(("min_x_center_y", "max_x_center_y"))
    if _should_enable_center_x(view, orientation, component_bounds):
        primary_styles.extend(("center_x_min_y", "center_x_max_y"))
    return tuple(primary_styles)


def _bucket_name_for_style(style_name: str) -> str:
    if style_name.startswith("bucket_back_wall_"):
        return "rear_floor_anchored"
    if style_name.startswith("bucket_lock_"):
        return "wall_locked"
    if style_name.startswith("bucket_stack_"):
        return "stable_stack_on_top"
    if style_name.startswith("row_gap_"):
        return "row_gap_fill"
    return "generic_anchor"


def _append_candidate(
    candidates: list[CandidatePlacement],
    *,
    view: DecisionStateView,
    support_plane: SupportPlane,
    orientation: OrientationOption,
    group_index: int,
    component_index: int,
    component_bounds: RectBounds,
    style_name: str,
    style_rank: int,
    anchor_x: float,
    anchor_y: float,
    center_x: float,
    center_y: float,
    center_z: float,
) -> None:
    if not _candidate_within_truck(view, orientation, center_x, center_y):
        return
    if not _candidate_within_bounds(view, orientation, center_x, center_y, component_bounds):
        return
    if not _support_area_sufficient(view, support_plane, orientation, center_x, center_y):
        return
    position = (float(center_x), float(center_y), float(center_z))
    bucket_name = _bucket_name_for_style(style_name)
    candidates.append(
        CandidatePlacement(
            group_index=group_index,
            support_plane_id=support_plane.plane_id,
            support_plane_index=support_plane.index,
            orientation_index=orientation.index,
            rotated_dimensions=orientation.rotated_dimensions,
            bucket_name=bucket_name,
            anchor_style=f"{style_name}:component{component_index}",
            anchor_signature=(
                support_plane.plane_id,
                component_index,
                bucket_name,
                style_name,
                round(anchor_x, 6),
                round(anchor_y, 6),
                round(support_plane.z_support, 6),
            ),
            position=position,
            orientation_wxyz=orientation.orientation_wxyz,
            support_component_bounds=component_bounds,
            dedup_key=candidate_dedup_key(position, orientation.orientation_wxyz),
            sort_key=(
                support_plane.index,
                orientation.index,
                component_index,
                style_rank,
                round(center_x, 6),
                round(center_y, 6),
                round(center_z, 6),
            ),
        )
    )


def _row_gap_candidates(
    view: DecisionStateView,
    support_plane: SupportPlane,
    orientation: OrientationOption,
    *,
    group_index: int,
    component_index: int,
    component_bounds: RectBounds,
    center_z: float,
    allow_secondary_widening: bool,
) -> list[CandidatePlacement]:
    component_boxes = _component_boxes_on_plane(view, support_plane, component_bounds)
    if not component_boxes:
        return []
    frontier_x = _component_frontier_x(component_boxes, default_frontier_x=component_bounds[0])
    depth_band = max(orientation.footprint_depth, float(view.config.dimension_ranges[0][0]))
    active_row_boxes = tuple(
        rectangle
        for rectangle in component_boxes
        if rectangle[2] >= frontier_x - depth_band - view.config.geometry_epsilon
    )
    intervals = _merged_intervals(
        tuple(
            (max(component_bounds[1], rectangle[1]), min(component_bounds[3], rectangle[3]))
            for rectangle in active_row_boxes
        ),
        epsilon=view.config.geometry_epsilon,
    )
    if not intervals:
        return []

    candidates: list[CandidatePlacement] = []
    gap_boundaries = [component_bounds[1], *(value for interval in intervals for value in interval), component_bounds[3]]
    min_future_width = float(view.config.dimension_ranges[1][0])
    for gap_index, (gap_start, gap_end) in enumerate(zip(gap_boundaries, gap_boundaries[1:])):
        gap_width = gap_end - gap_start
        if gap_width + view.config.geometry_epsilon < orientation.footprint_width:
            continue
        center_x = float(frontier_x - orientation.min_x)
        _append_candidate(
            candidates,
            view=view,
            support_plane=support_plane,
            orientation=orientation,
            group_index=group_index,
            component_index=component_index,
            component_bounds=component_bounds,
            style_name=f"row_gap_left_{gap_index}",
            style_rank=-10,
            anchor_x=frontier_x,
            anchor_y=gap_start,
            center_x=center_x,
            center_y=float(gap_start - orientation.min_y),
            center_z=center_z,
        )
        _append_candidate(
            candidates,
            view=view,
            support_plane=support_plane,
            orientation=orientation,
            group_index=group_index,
            component_index=component_index,
            component_bounds=component_bounds,
            style_name=f"row_gap_right_{gap_index}",
            style_rank=-9,
            anchor_x=frontier_x,
            anchor_y=gap_end,
            center_x=center_x,
            center_y=float(gap_end - orientation.max_y),
            center_z=center_z,
        )
        if allow_secondary_widening or gap_width <= orientation.footprint_width + min_future_width + view.config.geometry_epsilon:
            _append_candidate(
                candidates,
                view=view,
                support_plane=support_plane,
                orientation=orientation,
                group_index=group_index,
                component_index=component_index,
                component_bounds=component_bounds,
                style_name=f"row_gap_center_{gap_index}",
                style_rank=-8,
                anchor_x=frontier_x,
                anchor_y=float((gap_start + gap_end) * 0.5),
                center_x=center_x,
                center_y=float((gap_start + gap_end) * 0.5),
                center_z=center_z,
            )
    return candidates


def _back_wall_floor_candidates(
    view: DecisionStateView,
    support_plane: SupportPlane,
    orientation: OrientationOption,
    *,
    group_index: int,
    component_index: int,
    component_bounds: RectBounds,
    center_z: float,
    allow_secondary_widening: bool,
) -> list[CandidatePlacement]:
    if support_plane.z_support > view.config.support_plane_epsilon:
        return []
    center_x = float(component_bounds[2] - orientation.max_x)
    candidates: list[CandidatePlacement] = []
    _append_candidate(
        candidates,
        view=view,
        support_plane=support_plane,
        orientation=orientation,
        group_index=group_index,
        component_index=component_index,
        component_bounds=component_bounds,
        style_name="bucket_back_wall_left",
        style_rank=-40,
        anchor_x=component_bounds[2],
        anchor_y=component_bounds[1],
        center_x=center_x,
        center_y=float(component_bounds[1] - orientation.min_y),
        center_z=center_z,
    )
    _append_candidate(
        candidates,
        view=view,
        support_plane=support_plane,
        orientation=orientation,
        group_index=group_index,
        component_index=component_index,
        component_bounds=component_bounds,
        style_name="bucket_back_wall_right",
        style_rank=-39,
        anchor_x=component_bounds[2],
        anchor_y=component_bounds[3],
        center_x=center_x,
        center_y=float(component_bounds[3] - orientation.max_y),
        center_z=center_z,
    )
    width_slack = (component_bounds[3] - component_bounds[1]) - orientation.footprint_width
    if allow_secondary_widening or width_slack <= float(view.config.dimension_ranges[1][0]) + view.config.geometry_epsilon:
        _append_candidate(
            candidates,
            view=view,
            support_plane=support_plane,
            orientation=orientation,
            group_index=group_index,
            component_index=component_index,
            component_bounds=component_bounds,
            style_name="bucket_back_wall_center",
            style_rank=-38,
            anchor_x=component_bounds[2],
            anchor_y=float((component_bounds[1] + component_bounds[3]) * 0.5),
            center_x=center_x,
            center_y=float((component_bounds[1] + component_bounds[3]) * 0.5),
            center_z=center_z,
        )
    return candidates


def _wall_neighbor_lock_candidates(
    view: DecisionStateView,
    support_plane: SupportPlane,
    orientation: OrientationOption,
    *,
    group_index: int,
    component_index: int,
    component_bounds: RectBounds,
    center_z: float,
    allow_secondary_widening: bool,
) -> list[CandidatePlacement]:
    component_boxes = _component_boxes_on_plane(view, support_plane, component_bounds)
    if not component_boxes:
        return []
    frontier_x = _component_frontier_x(component_boxes, default_frontier_x=component_bounds[0])
    center_x = float(frontier_x - orientation.min_x)
    candidates: list[CandidatePlacement] = []
    _append_candidate(
        candidates,
        view=view,
        support_plane=support_plane,
        orientation=orientation,
        group_index=group_index,
        component_index=component_index,
        component_bounds=component_bounds,
        style_name="bucket_lock_left_wall",
        style_rank=-34,
        anchor_x=frontier_x,
        anchor_y=component_bounds[1],
        center_x=center_x,
        center_y=float(component_bounds[1] - orientation.min_y),
        center_z=center_z,
    )
    _append_candidate(
        candidates,
        view=view,
        support_plane=support_plane,
        orientation=orientation,
        group_index=group_index,
        component_index=component_index,
        component_bounds=component_bounds,
        style_name="bucket_lock_right_wall",
        style_rank=-33,
        anchor_x=frontier_x,
        anchor_y=component_bounds[3],
        center_x=center_x,
        center_y=float(component_bounds[3] - orientation.max_y),
        center_z=center_z,
    )
    component_center_y = (component_bounds[1] + component_bounds[3]) * 0.5
    if allow_secondary_widening:
        _append_candidate(
            candidates,
            view=view,
            support_plane=support_plane,
            orientation=orientation,
            group_index=group_index,
            component_index=component_index,
            component_bounds=component_bounds,
            style_name="bucket_lock_center",
            style_rank=-32,
            anchor_x=frontier_x,
            anchor_y=component_center_y,
            center_x=center_x,
            center_y=float(component_center_y),
            center_z=center_z,
        )
    return candidates


def _stable_stack_candidates(
    view: DecisionStateView,
    support_plane: SupportPlane,
    orientation: OrientationOption,
    *,
    group_index: int,
    component_index: int,
    component_bounds: RectBounds,
    center_z: float,
    allow_secondary_widening: bool,
) -> list[CandidatePlacement]:
    if support_plane.z_support <= view.config.support_plane_epsilon:
        return []
    epsilon = view.config.geometry_epsilon
    support_rectangles = tuple(
        rectangle
        for rectangle in support_plane.support_rectangles
        if _rectangles_overlap(rectangle, component_bounds, epsilon=epsilon)
    )
    if not support_rectangles:
        return []

    candidates: list[CandidatePlacement] = []
    for support_rect_index, support_rect in enumerate(sorted(support_rectangles)):
        support_depth = support_rect[2] - support_rect[0]
        support_width = support_rect[3] - support_rect[1]
        if support_depth + epsilon < orientation.footprint_depth or support_width + epsilon < orientation.footprint_width:
            continue
        rect_component_index = (component_index * 100) + support_rect_index
        _append_candidate(
            candidates,
            view=view,
            support_plane=support_plane,
            orientation=orientation,
            group_index=group_index,
            component_index=rect_component_index,
            component_bounds=support_rect,
            style_name="bucket_stack_center",
            style_rank=-37,
            anchor_x=float((support_rect[0] + support_rect[2]) * 0.5),
            anchor_y=float((support_rect[1] + support_rect[3]) * 0.5),
            center_x=float((support_rect[0] + support_rect[2]) * 0.5),
            center_y=float((support_rect[1] + support_rect[3]) * 0.5),
            center_z=center_z,
        )
        _append_candidate(
            candidates,
            view=view,
            support_plane=support_plane,
            orientation=orientation,
            group_index=group_index,
            component_index=rect_component_index,
            component_bounds=support_rect,
            style_name="bucket_stack_min_min",
            style_rank=-36,
            anchor_x=float(support_rect[0]),
            anchor_y=float(support_rect[1]),
            center_x=float(support_rect[0] - orientation.min_x),
            center_y=float(support_rect[1] - orientation.min_y),
            center_z=center_z,
        )
        _append_candidate(
            candidates,
            view=view,
            support_plane=support_plane,
            orientation=orientation,
            group_index=group_index,
            component_index=rect_component_index,
            component_bounds=support_rect,
            style_name="bucket_stack_min_max",
            style_rank=-35,
            anchor_x=float(support_rect[0]),
            anchor_y=float(support_rect[3]),
            center_x=float(support_rect[0] - orientation.min_x),
            center_y=float(support_rect[3] - orientation.max_y),
            center_z=center_z,
        )
        _append_candidate(
            candidates,
            view=view,
            support_plane=support_plane,
            orientation=orientation,
            group_index=group_index,
            component_index=rect_component_index,
            component_bounds=support_rect,
            style_name="bucket_stack_max_min",
            style_rank=-34,
            anchor_x=float(support_rect[2]),
            anchor_y=float(support_rect[1]),
            center_x=float(support_rect[2] - orientation.max_x),
            center_y=float(support_rect[1] - orientation.min_y),
            center_z=center_z,
        )
        _append_candidate(
            candidates,
            view=view,
            support_plane=support_plane,
            orientation=orientation,
            group_index=group_index,
            component_index=rect_component_index,
            component_bounds=support_rect,
            style_name="bucket_stack_max_max",
            style_rank=-33,
            anchor_x=float(support_rect[2]),
            anchor_y=float(support_rect[3]),
            center_x=float(support_rect[2] - orientation.max_x),
            center_y=float(support_rect[3] - orientation.max_y),
            center_z=center_z,
        )
        if allow_secondary_widening and support_width - orientation.footprint_width > epsilon:
            _append_candidate(
                candidates,
                view=view,
                support_plane=support_plane,
                orientation=orientation,
                group_index=group_index,
                component_index=rect_component_index,
                component_bounds=support_rect,
                style_name="bucket_stack_midline",
                style_rank=-32,
                anchor_x=float((support_rect[0] + support_rect[2]) * 0.5),
                anchor_y=float((support_rect[1] + support_rect[3]) * 0.5),
                center_x=float((support_rect[0] + support_rect[2]) * 0.5),
                center_y=float((support_rect[1] + support_rect[3]) * 0.5),
                center_z=center_z,
            )
    return candidates


def _group_generation_cap(
    support_plane: SupportPlane,
    *,
    remaining_generation_budget: int | None,
    allow_secondary_widening: bool,
) -> int | None:
    if remaining_generation_budget is None:
        return None
    if support_plane.index == 0:
        base_cap = 64
    elif support_plane.index <= 1:
        base_cap = 40
    elif support_plane.index <= 3:
        base_cap = 28
    elif support_plane.index <= 5:
        base_cap = 20
    else:
        base_cap = 12
    if allow_secondary_widening:
        base_cap = int(base_cap * 1.5)
    return min(remaining_generation_budget, max(8, base_cap))


def generate_group_candidates(
    view: DecisionStateView,
    support_plane: SupportPlane,
    orientation: OrientationOption,
    *,
    group_index: int,
    heuristic_profile: HeuristicProfile = "baseline",
    allow_secondary_widening: bool = False,
    max_candidates: int | None = None,
    deadline_monotonic: float | None = None,
) -> list[CandidatePlacement]:
    center_z = float(support_plane.z_support - orientation.bottom_z)
    candidates: list[CandidatePlacement] = []
    support_components = _support_components(view, support_plane)
    for component_index, component_rectangles in enumerate(support_components):
        component_bounds = _component_bounds(
            component_rectangles,
            truck_depth=view.truck.depth,
            truck_width=view.truck.width,
        )
        component_min_x, component_min_y, component_max_x, component_max_y = component_bounds
        component_center_x = round((component_min_x + component_max_x) * 0.5, 6)
        component_center_y = round((component_min_y + component_max_y) * 0.5, 6)
        cavity_depth = component_bounds[2] - component_bounds[0]
        cavity_width = component_bounds[3] - component_bounds[1]
        if cavity_depth + view.config.geometry_epsilon < orientation.footprint_depth:
            continue
        if cavity_width + view.config.geometry_epsilon < orientation.footprint_width:
            continue
        candidates.extend(
            _back_wall_floor_candidates(
                view,
                support_plane,
                orientation,
                group_index=group_index,
                component_index=component_index,
                component_bounds=component_bounds,
                center_z=center_z,
                allow_secondary_widening=allow_secondary_widening,
            )
        )
        generate_stack_bucket = bool(
            support_plane.z_support > view.config.support_plane_epsilon
            and (
                allow_secondary_widening
                or view.current_max_x >= max(float(view.truck.depth) * 0.35, orientation.footprint_depth * 1.5)
                or len(view.placed_boxes) >= 6
                or support_plane.frontier_gap <= (0.25 * max(orientation.footprint_depth, view.config.geometry_epsilon))
            )
        )
        if generate_stack_bucket:
            candidates.extend(
                _stable_stack_candidates(
                    view,
                    support_plane,
                    orientation,
                    group_index=group_index,
                    component_index=component_index,
                    component_bounds=component_bounds,
                    center_z=center_z,
                    allow_secondary_widening=allow_secondary_widening,
                )
            )
        candidates.extend(
            _wall_neighbor_lock_candidates(
                view,
                support_plane,
                orientation,
                group_index=group_index,
                component_index=component_index,
                component_bounds=component_bounds,
                center_z=center_z,
                allow_secondary_widening=allow_secondary_widening,
            )
        )
        if max_candidates is not None and len(candidates) >= max_candidates:
            return candidates[:max_candidates]
        row_candidates = _row_gap_candidates(
            view,
            support_plane,
            orientation,
            group_index=group_index,
            component_index=component_index,
            component_bounds=component_bounds,
            center_z=center_z,
            allow_secondary_widening=allow_secondary_widening,
        )
        candidates.extend(row_candidates)
        if max_candidates is not None and len(candidates) >= max_candidates:
            return candidates[:max_candidates]
        anchors = generate_edge_anchors(
            view,
            support_plane,
            orientation,
            anchor_rectangles=component_rectangles,
        )
        allowed_styles = _allowed_anchor_styles(
            view,
            orientation,
            component_bounds,
            heuristic_profile=heuristic_profile,
            allow_secondary_widening=allow_secondary_widening,
        )
        anchors_by_style = {
            "min_x_min_y": (anchors.min_x, anchors.min_y),
            "min_x_max_y": (anchors.min_x, anchors.max_y),
            "max_x_min_y": (anchors.max_x, anchors.min_y),
            "max_x_max_y": (anchors.max_x, anchors.max_y),
        }
        if "center_x_min_y" in allowed_styles:
            anchors_by_style["center_x_min_y"] = (anchors.center_x, anchors.min_y)
        if "center_x_max_y" in allowed_styles:
            anchors_by_style["center_x_max_y"] = (anchors.center_x, anchors.max_y)
        if "min_x_center_y" in allowed_styles:
            anchors_by_style["min_x_center_y"] = (anchors.min_x, (component_center_y,))
        if "max_x_center_y" in allowed_styles:
            anchors_by_style["max_x_center_y"] = (anchors.max_x, (component_center_y,))
        for style_index, style in enumerate(ANCHOR_STYLES):
            if style not in allowed_styles or style not in anchors_by_style:
                continue
            x_anchors, y_anchors = anchors_by_style[style]
            for x_anchor in x_anchors:
                for y_anchor in y_anchors:
                    if deadline_monotonic is not None and perf_counter() >= deadline_monotonic:
                        return candidates
                    if max_candidates is not None and len(candidates) >= max_candidates:
                        return candidates
                    center_x = _center_x_from_anchor(x_anchor, orientation, style)
                    center_y = _center_y_from_anchor(y_anchor, orientation, style)
                    _append_candidate(
                        candidates,
                        view=view,
                        support_plane=support_plane,
                        orientation=orientation,
                        group_index=group_index,
                        component_index=component_index,
                        component_bounds=component_bounds,
                        style_name=style,
                        style_rank=style_index,
                        anchor_x=x_anchor,
                        anchor_y=y_anchor,
                        center_x=center_x,
                        center_y=center_y,
                        center_z=center_z,
                    )
    return candidates


def generate_candidate_groups_limited(
    view: DecisionStateView,
    support_planes: list[SupportPlane],
    orientations: list[OrientationOption],
    *,
    heuristic_profile: HeuristicProfile = "baseline",
    allow_secondary_widening: bool = False,
    start_group_index: int = 0,
    seen_dedup_keys: set[tuple[float, ...]] | None = None,
    max_generated_candidates: int | None = None,
    deadline_monotonic: float | None = None,
) -> tuple[list[CandidateGroup], int, int, set[tuple[float, ...]]]:
    groups: list[CandidateGroup] = []
    seen_keys = set() if seen_dedup_keys is None else set(seen_dedup_keys)
    total_generated_count = 0
    next_group_index = start_group_index

    for support_plane in support_planes:
        for orientation in orientations:
            if deadline_monotonic is not None and perf_counter() >= deadline_monotonic:
                return groups, total_generated_count, next_group_index, seen_keys
            if max_generated_candidates is not None and total_generated_count >= max_generated_candidates:
                return groups, total_generated_count, next_group_index, seen_keys

            group_index = next_group_index
            next_group_index += 1
            remaining_generation_budget = None
            if max_generated_candidates is not None:
                remaining_generation_budget = max_generated_candidates - total_generated_count
                if remaining_generation_budget <= 0:
                    return groups, total_generated_count, next_group_index, seen_keys
            group_generation_cap = _group_generation_cap(
                support_plane,
                remaining_generation_budget=remaining_generation_budget,
                allow_secondary_widening=allow_secondary_widening,
            )

            raw_candidates = generate_group_candidates(
                view,
                support_plane,
                orientation,
                group_index=group_index,
                heuristic_profile=heuristic_profile,
                allow_secondary_widening=allow_secondary_widening,
                max_candidates=group_generation_cap,
                deadline_monotonic=deadline_monotonic,
            )
            if not raw_candidates:
                continue

            deduped_candidates: dict[tuple[float, ...], CandidatePlacement] = {}
            for candidate in raw_candidates:
                if candidate.dedup_key in seen_keys:
                    continue
                existing = deduped_candidates.get(candidate.dedup_key)
                if existing is None or candidate.sort_key < existing.sort_key:
                    deduped_candidates[candidate.dedup_key] = candidate

            if not deduped_candidates:
                continue

            candidates = sorted(deduped_candidates.values(), key=lambda item: item.sort_key)
            groups.append(
                CandidateGroup(
                    index=group_index,
                    support_plane=support_plane,
                    orientation=orientation,
                    candidates=candidates,
                )
            )
            seen_keys.update(candidate.dedup_key for candidate in candidates)
            total_generated_count += len(candidates)

    return groups, total_generated_count, next_group_index, seen_keys


def generate_candidate_groups(
    view: DecisionStateView,
    support_planes: list[SupportPlane],
    orientations: list[OrientationOption],
) -> list[CandidateGroup]:
    groups, _, _, _ = generate_candidate_groups_limited(view, support_planes, orientations)
    return groups
