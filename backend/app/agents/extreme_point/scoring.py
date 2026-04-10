from __future__ import annotations

from shapely.geometry import box

from app.agents.extreme_point.state_view import DecisionStateView
from app.agents.extreme_point.types import CandidatePlacement, OrientationOption, ScoreBreakdown, ScoreWeights


def _interval_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def _merged_interval_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    merged_start, merged_end = ordered[0]
    total = 0.0
    for start, end in ordered[1:]:
        if start <= merged_end:
            merged_end = max(merged_end, end)
            continue
        total += merged_end - merged_start
        merged_start, merged_end = start, end
    total += merged_end - merged_start
    return max(0.0, total)


def candidate_bounds(candidate: CandidatePlacement, orientation: OrientationOption) -> tuple[float, float, float, float]:
    center_x, center_y, _ = candidate.position
    return (
        center_x + orientation.min_x,
        center_y + orientation.min_y,
        center_x + orientation.max_x,
        center_y + orientation.max_y,
    )


def _relevant_obstacles(view: DecisionStateView, candidate: CandidatePlacement, orientation: OrientationOption) -> tuple[tuple[float, float, float, float], ...]:
    candidate_bottom_z = candidate.position[2] + orientation.bottom_z
    candidate_top_z = candidate.position[2] + orientation.top_z
    z_overlap_obstacles = view.obstacle_rectangles_for_z_range(candidate_bottom_z, candidate_top_z)
    support_surfaces = view.support_surface_rectangles(candidate_bottom_z)
    if not support_surfaces:
        return z_overlap_obstacles
    combined: dict[tuple[float, float, float, float], tuple[float, float, float, float]] = {
        tuple(round(value, 6) for value in obstacle): obstacle
        for obstacle in z_overlap_obstacles
    }
    for surface in support_surfaces:
        key = tuple(round(value, 6) for value in surface)
        combined.setdefault(key, surface)
    return tuple(combined.values())


def _local_frontier_x(
    view: DecisionStateView,
    bounds: tuple[float, float, float, float],
) -> float:
    _, min_y, _, max_y = bounds
    local_frontier_x = 0.0
    for obstacle in view.obstacle_rectangles:
        overlap = _interval_overlap(min_y, max_y, obstacle[1], obstacle[3])
        if overlap <= view.config.geometry_epsilon:
            continue
        if obstacle[2] > local_frontier_x:
            local_frontier_x = obstacle[2]
    return float(local_frontier_x)


def _frontier_slack(
    view: DecisionStateView,
    bounds: tuple[float, float, float, float],
) -> float:
    max_x = bounds[2]
    return max(0.0, _local_frontier_x(view, bounds) - max_x)


def _frontier_reach_reward(
    view: DecisionStateView,
    bounds: tuple[float, float, float, float],
) -> float:
    candidate_depth = max(view.config.geometry_epsilon, bounds[2] - bounds[0])
    return max(0.0, 1.0 - (_frontier_slack(view, bounds) / candidate_depth))


def _future_sliver_penalty(
    view: DecisionStateView,
    candidate: CandidatePlacement,
    bounds: tuple[float, float, float, float],
) -> float:
    component_bounds = candidate.support_component_bounds
    if component_bounds is None:
        return 0.0
    component_min_x, component_min_y, component_max_x, component_max_y = component_bounds
    min_future_depth = float(view.config.dimension_ranges[0][0])
    min_future_width = float(view.config.dimension_ranges[1][0])
    strips = (
        (bounds[0] - component_min_x, min_future_depth),
        (component_max_x - bounds[2], min_future_depth),
        (bounds[1] - component_min_y, min_future_width),
        (component_max_y - bounds[3], min_future_width),
    )
    penalty = 0.0
    for strip_size, threshold in strips:
        if strip_size <= view.config.geometry_epsilon or strip_size >= threshold:
            continue
        penalty += (threshold - strip_size) / max(view.config.geometry_epsilon, threshold)
    return float(penalty)


def _remaining_fragment_metrics(
    view: DecisionStateView,
    candidate: CandidatePlacement,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    component_bounds = candidate.support_component_bounds
    if component_bounds is None:
        return 0.0, 0.0, 0.0
    component_polygon = box(*component_bounds)
    candidate_polygon = box(*bounds)
    remaining = component_polygon.difference(candidate_polygon)
    if remaining.is_empty:
        return 0.0, 0.0, 1.0
    if remaining.geom_type == "Polygon":
        fragments = [remaining]
    else:
        fragments = [geometry for geometry in getattr(remaining, "geoms", []) if geometry.geom_type == "Polygon"]

    component_area = max(view.config.geometry_epsilon, float(component_polygon.area))
    min_depth = float(view.config.dimension_ranges[0][0])
    min_width = float(view.config.dimension_ranges[1][0])
    usable_area = 0.0
    usable_fragments = 0
    total_fragments = 0
    for fragment in fragments:
        total_fragments += 1
        min_x, min_y, max_x, max_y = fragment.bounds
        depth = float(max_x - min_x)
        width = float(max_y - min_y)
        area = float(fragment.area)
        if depth + view.config.geometry_epsilon >= min_depth and width + view.config.geometry_epsilon >= min_width:
            usable_area += area
            usable_fragments += 1
    usable_area_reward = usable_area / component_area
    fragmentation_penalty = float(max(0, total_fragments - 1) + max(0, usable_fragments - 1))
    min_y_gap = abs(bounds[1] - component_bounds[1])
    max_y_gap = abs(component_bounds[3] - bounds[3])
    shelf_completion_reward = 1.0 if min_y_gap <= view.config.geometry_epsilon and max_y_gap <= view.config.geometry_epsilon else 0.0
    return float(usable_area_reward), fragmentation_penalty, shelf_completion_reward


def _front_gap(
    view: DecisionStateView,
    bounds: tuple[float, float, float, float],
    obstacles: tuple[tuple[float, float, float, float], ...],
) -> float:
    min_x, min_y, _, max_y = bounds
    best_gap = min_x
    for obstacle in obstacles:
        overlap = _interval_overlap(min_y, max_y, obstacle[1], obstacle[3])
        if overlap <= view.config.geometry_epsilon:
            continue
        if obstacle[2] <= min_x + view.config.geometry_epsilon:
            best_gap = min(best_gap, min_x - obstacle[2])
    return max(0.0, float(best_gap))


def _left_gap(
    view: DecisionStateView,
    bounds: tuple[float, float, float, float],
    obstacles: tuple[tuple[float, float, float, float], ...],
) -> float:
    min_x, min_y, max_x, _ = bounds
    best_gap = min_y
    for obstacle in obstacles:
        overlap = _interval_overlap(min_x, max_x, obstacle[0], obstacle[2])
        if overlap <= view.config.geometry_epsilon:
            continue
        if obstacle[3] <= min_y + view.config.geometry_epsilon:
            best_gap = min(best_gap, min_y - obstacle[3])
    return max(0.0, float(best_gap))


def _right_gap(
    view: DecisionStateView,
    bounds: tuple[float, float, float, float],
    obstacles: tuple[tuple[float, float, float, float], ...],
) -> float:
    min_x, _, max_x, max_y = bounds
    best_gap = view.truck.width - max_y
    for obstacle in obstacles:
        overlap = _interval_overlap(min_x, max_x, obstacle[0], obstacle[2])
        if overlap <= view.config.geometry_epsilon:
            continue
        if obstacle[1] >= max_y - view.config.geometry_epsilon:
            best_gap = min(best_gap, obstacle[1] - max_y)
    return max(0.0, float(best_gap))


def _front_contact_fraction(
    view: DecisionStateView,
    bounds: tuple[float, float, float, float],
    obstacles: tuple[tuple[float, float, float, float], ...],
    weights: ScoreWeights,
) -> float:
    min_x, min_y, _, max_y = bounds
    intervals: list[tuple[float, float]] = []
    if min_x <= weights.flush_contact_tolerance:
        intervals.append((min_y, max_y))
    for obstacle in obstacles:
        if abs(obstacle[2] - min_x) > weights.flush_contact_tolerance:
            continue
        overlap = _interval_overlap(min_y, max_y, obstacle[1], obstacle[3])
        if overlap <= view.config.geometry_epsilon:
            continue
        intervals.append((max(min_y, obstacle[1]), min(max_y, obstacle[3])))
    width = max(view.config.geometry_epsilon, max_y - min_y)
    return min(1.0, _merged_interval_length(intervals) / width)


def _left_contact_fraction(
    view: DecisionStateView,
    bounds: tuple[float, float, float, float],
    obstacles: tuple[tuple[float, float, float, float], ...],
    weights: ScoreWeights,
) -> float:
    min_x, min_y, max_x, _ = bounds
    intervals: list[tuple[float, float]] = []
    if min_y <= weights.flush_contact_tolerance:
        intervals.append((min_x, max_x))
    for obstacle in obstacles:
        if abs(obstacle[3] - min_y) > weights.flush_contact_tolerance:
            continue
        overlap = _interval_overlap(min_x, max_x, obstacle[0], obstacle[2])
        if overlap <= view.config.geometry_epsilon:
            continue
        intervals.append((max(min_x, obstacle[0]), min(max_x, obstacle[2])))
    depth = max(view.config.geometry_epsilon, max_x - min_x)
    return min(1.0, _merged_interval_length(intervals) / depth)


def _right_contact_fraction(
    view: DecisionStateView,
    bounds: tuple[float, float, float, float],
    obstacles: tuple[tuple[float, float, float, float], ...],
    weights: ScoreWeights,
) -> float:
    min_x, _, max_x, max_y = bounds
    intervals: list[tuple[float, float]] = []
    if abs(view.truck.width - max_y) <= weights.flush_contact_tolerance:
        intervals.append((min_x, max_x))
    for obstacle in obstacles:
        if abs(obstacle[1] - max_y) > weights.flush_contact_tolerance:
            continue
        overlap = _interval_overlap(min_x, max_x, obstacle[0], obstacle[2])
        if overlap <= view.config.geometry_epsilon:
            continue
        intervals.append((max(min_x, obstacle[0]), min(max_x, obstacle[2])))
    depth = max(view.config.geometry_epsilon, max_x - min_x)
    return min(1.0, _merged_interval_length(intervals) / depth)


def compute_score_breakdown(
    view: DecisionStateView,
    candidate: CandidatePlacement,
    orientation: OrientationOption,
    *,
    support_ratio: float | None,
    weights: ScoreWeights,
) -> ScoreBreakdown:
    bounds = candidate_bounds(candidate, orientation)
    obstacles = _relevant_obstacles(view, candidate, orientation)
    candidate_max_x = bounds[2]
    delta_x = max(0.0, candidate_max_x - view.current_max_x)
    front_gap = _front_gap(view, bounds, obstacles)
    frontier_slack = _frontier_slack(view, bounds)
    future_sliver_penalty = _future_sliver_penalty(view, candidate, bounds)
    future_usable_area_reward, fragmentation_penalty, shelf_completion_reward = _remaining_fragment_metrics(
        view,
        candidate,
        bounds,
    )
    left_gap = _left_gap(view, bounds, obstacles)
    right_gap = _right_gap(view, bounds, obstacles)
    gap_penalty = (
        weights.front_gap_weight * front_gap
        + weights.frontier_slack_weight * frontier_slack
        + weights.future_sliver_weight * future_sliver_penalty
        + weights.fragmentation_penalty_weight * fragmentation_penalty
        + weights.left_gap_weight * left_gap
        + weights.right_gap_weight * right_gap
    )
    support_reward = float(0.0 if support_ratio is None else support_ratio)
    frontier_reach_reward = _frontier_reach_reward(view, bounds)
    front_contact_fraction = _front_contact_fraction(view, bounds, obstacles, weights)
    left_contact_fraction = _left_contact_fraction(view, bounds, obstacles, weights)
    right_contact_fraction = _right_contact_fraction(view, bounds, obstacles, weights)
    floor_bonus = weights.floor_contact_bonus if candidate.position[2] + orientation.bottom_z <= weights.flush_contact_tolerance else 0.0
    contact_reward = (
        floor_bonus
        + weights.frontier_reach_weight * frontier_reach_reward
        + front_contact_fraction
        + 0.5 * left_contact_fraction
        + 0.5 * right_contact_fraction
    )
    total_score = (
        -weights.alpha * delta_x
        -weights.beta * gap_penalty
        + weights.gamma * support_reward
        + weights.eta * contact_reward
        + weights.future_usable_area_weight * future_usable_area_reward
        + weights.shelf_completion_weight * shelf_completion_reward
    )
    return ScoreBreakdown(
        total_score=float(total_score),
        delta_x=float(delta_x),
        gap_penalty=float(gap_penalty),
        front_gap=float(front_gap),
        frontier_slack=float(frontier_slack),
        future_sliver_penalty=float(future_sliver_penalty),
        fragmentation_penalty=float(fragmentation_penalty),
        left_gap=float(left_gap),
        right_gap=float(right_gap),
        support_reward=float(support_reward),
        contact_reward=float(contact_reward),
        frontier_reach_reward=float(frontier_reach_reward),
        future_usable_area_reward=float(future_usable_area_reward),
        shelf_completion_reward=float(shelf_completion_reward),
        front_contact_fraction=float(front_contact_fraction),
        left_contact_fraction=float(left_contact_fraction),
        right_contact_fraction=float(right_contact_fraction),
    )


def score_breakdown_with_support_ratio(
    score: ScoreBreakdown,
    *,
    support_ratio: float | None,
    weights: ScoreWeights,
) -> ScoreBreakdown:
    support_reward = float(0.0 if support_ratio is None else support_ratio)
    total_score = score.total_score + weights.gamma * (support_reward - score.support_reward)
    return ScoreBreakdown(
        total_score=float(total_score),
        delta_x=score.delta_x,
        gap_penalty=score.gap_penalty,
        front_gap=score.front_gap,
        frontier_slack=score.frontier_slack,
        future_sliver_penalty=score.future_sliver_penalty,
        fragmentation_penalty=score.fragmentation_penalty,
        left_gap=score.left_gap,
        right_gap=score.right_gap,
        support_reward=support_reward,
        contact_reward=score.contact_reward,
        frontier_reach_reward=score.frontier_reach_reward,
        future_usable_area_reward=score.future_usable_area_reward,
        shelf_completion_reward=score.shelf_completion_reward,
        front_contact_fraction=score.front_contact_fraction,
        left_contact_fraction=score.left_contact_fraction,
        right_contact_fraction=score.right_contact_fraction,
    )
