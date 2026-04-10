from __future__ import annotations

from time import perf_counter

from app.agents.extreme_point.anchors import ANCHOR_STYLES, candidate_dedup_key, generate_edge_anchors
from app.agents.extreme_point.state_view import DecisionStateView
from app.agents.extreme_point.types import CandidateGroup, CandidatePlacement, OrientationOption, SupportPlane


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


def _free_cavity_rectangles(
    view: DecisionStateView,
    support_plane: SupportPlane,
    orientation: OrientationOption,
) -> tuple[tuple[float, float, float, float], ...]:
    fallback_bounds = (0.0, 0.0, float(view.truck.depth), float(view.truck.width))
    return view.free_rectangles(
        cache_key=f"{support_plane.plane_id}:h{orientation.height:.6f}",
        base_rectangles=support_plane.support_rectangles,
        fallback_bounds=fallback_bounds,
        bottom_z=support_plane.z_support,
        top_z=support_plane.z_support + orientation.height,
    )


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


def generate_group_candidates(
    view: DecisionStateView,
    support_plane: SupportPlane,
    orientation: OrientationOption,
    *,
    group_index: int,
    max_candidates: int | None = None,
    deadline_monotonic: float | None = None,
) -> list[CandidatePlacement]:
    center_z = float(support_plane.z_support - orientation.bottom_z)
    candidates: list[CandidatePlacement] = []
    cavity_rectangles = _free_cavity_rectangles(view, support_plane, orientation)
    if not cavity_rectangles and not support_plane.support_rectangles:
        cavity_rectangles = ((0.0, 0.0, float(view.truck.depth), float(view.truck.width)),)
    for component_index, cavity_bounds in enumerate(cavity_rectangles):
        component_min_x, component_min_y, component_max_x, component_max_y = cavity_bounds
        component_center_x = round((component_min_x + component_max_x) * 0.5, 6)
        component_center_y = round((component_min_y + component_max_y) * 0.5, 6)
        component_bounds = (
            float(component_min_x),
            float(component_min_y),
            float(component_max_x),
            float(component_max_y),
        )
        cavity_depth = component_bounds[2] - component_bounds[0]
        cavity_width = component_bounds[3] - component_bounds[1]
        if cavity_depth + view.config.geometry_epsilon < orientation.footprint_depth:
            continue
        if cavity_width + view.config.geometry_epsilon < orientation.footprint_width:
            continue
        anchors = generate_edge_anchors(
            view,
            support_plane,
            orientation,
            anchor_rectangles=(component_bounds,),
        )
        anchors_by_style = {
            "min_x_min_y": (anchors.min_x, anchors.min_y),
            "min_x_max_y": (anchors.min_x, anchors.max_y),
            "max_x_min_y": (anchors.max_x, anchors.min_y),
            "max_x_max_y": (anchors.max_x, anchors.max_y),
            "center_x_min_y": (anchors.center_x, anchors.min_y),
            "center_x_max_y": (anchors.center_x, anchors.max_y),
        }
        anchors_by_style["min_x_center_y"] = (anchors.min_x, (component_center_y,))
        anchors_by_style["max_x_center_y"] = (anchors.max_x, (component_center_y,))
        anchors_by_style["center_x_center_y"] = ((component_center_x,), (component_center_y,))
        for style_index, style in enumerate(ANCHOR_STYLES):
            if style not in anchors_by_style:
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
                    if not _candidate_within_truck(view, orientation, center_x, center_y):
                        continue
                    if not _candidate_within_bounds(view, orientation, center_x, center_y, component_bounds):
                        continue
                    if not _support_area_sufficient(view, support_plane, orientation, center_x, center_y):
                        continue
                    position = (float(center_x), float(center_y), center_z)
                    candidates.append(
                        CandidatePlacement(
                            group_index=group_index,
                            support_plane_id=support_plane.plane_id,
                            support_plane_index=support_plane.index,
                            orientation_index=orientation.index,
                            anchor_style=f"{style}:component{component_index}",
                            position=position,
                            orientation_wxyz=orientation.orientation_wxyz,
                            support_component_bounds=component_bounds,
                            dedup_key=candidate_dedup_key(position, orientation.orientation_wxyz),
                            sort_key=(
                                support_plane.index,
                                orientation.index,
                                component_index,
                                style_index,
                                round(center_x, 6),
                                round(center_y, 6),
                                round(center_z, 6),
                            ),
                        )
                    )
    return candidates


def generate_candidate_groups_limited(
    view: DecisionStateView,
    support_planes: list[SupportPlane],
    orientations: list[OrientationOption],
    *,
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

            raw_candidates = generate_group_candidates(
                view,
                support_plane,
                orientation,
                group_index=group_index,
                max_candidates=remaining_generation_budget,
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
