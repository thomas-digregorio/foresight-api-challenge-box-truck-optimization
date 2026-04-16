from __future__ import annotations

from app.agents.extreme_point.state_view import DecisionStateView
from app.agents.extreme_point.types import CandidatePlacement, OrientationOption, RectBounds, ScoreBreakdown, ScoreWeights


def _interval_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def _overlap_bounds(lhs: RectBounds, rhs: RectBounds) -> RectBounds | None:
    min_x = max(lhs[0], rhs[0])
    min_y = max(lhs[1], rhs[1])
    max_x = min(lhs[2], rhs[2])
    max_y = min(lhs[3], rhs[3])
    if max_x <= min_x or max_y <= min_y:
        return None
    return (float(min_x), float(min_y), float(max_x), float(max_y))


def _rect_area(rectangle: RectBounds) -> float:
    return max(0.0, rectangle[2] - rectangle[0]) * max(0.0, rectangle[3] - rectangle[1])


def candidate_bounds(candidate: CandidatePlacement, orientation: OrientationOption) -> RectBounds:
    center_x, center_y, _ = candidate.position
    return (
        center_x + orientation.min_x,
        center_y + orientation.min_y,
        center_x + orientation.max_x,
        center_y + orientation.max_y,
    )


def _exact_density_after(view: DecisionStateView, candidate_max_x: float) -> float:
    if view.current_box is None:
        return float(view.game_state.density)
    max_x_reached = max(view.current_max_x, candidate_max_x)
    if max_x_reached <= view.config.geometry_epsilon:
        return 0.0
    total_volume = view.placed_volume + view.current_box.volume
    denominator = max_x_reached * view.truck.width * view.truck.height
    if denominator <= view.config.geometry_epsilon:
        return 0.0
    return float(total_volume / denominator)


def _support_geometry_metrics(
    view: DecisionStateView,
    bounds: RectBounds,
    candidate: CandidatePlacement,
    orientation: OrientationOption,
    *,
    support_ratio: float | None,
) -> tuple[float, float, tuple[float, float] | None, RectBounds | None]:
    footprint_area = max(view.config.geometry_epsilon, orientation.footprint_depth * orientation.footprint_width)
    candidate_center = ((bounds[0] + bounds[2]) * 0.5, (bounds[1] + bounds[3]) * 0.5)
    z_support = candidate.position[2] + orientation.bottom_z
    if z_support <= view.config.support_plane_epsilon:
        effective_support_ratio = 1.0 if support_ratio is None else float(support_ratio)
        return footprint_area, effective_support_ratio, candidate_center, bounds

    overlap_area = 0.0
    weighted_center_x = 0.0
    weighted_center_y = 0.0
    primary_support_rect: RectBounds | None = None
    primary_support_area = 0.0
    for surface in view.support_surface_rectangles(z_support):
        overlap = _overlap_bounds(bounds, surface)
        if overlap is None:
            continue
        area = _rect_area(overlap)
        if area <= view.config.geometry_epsilon:
            continue
        overlap_area += area
        weighted_center_x += ((overlap[0] + overlap[2]) * 0.5) * area
        weighted_center_y += ((overlap[1] + overlap[3]) * 0.5) * area
        if area > primary_support_area:
            primary_support_area = area
            primary_support_rect = overlap
    effective_support_ratio = float(support_ratio) if support_ratio is not None else min(1.0, overlap_area / footprint_area)
    support_centroid = None
    if overlap_area > view.config.geometry_epsilon:
        support_centroid = (weighted_center_x / overlap_area, weighted_center_y / overlap_area)
    return overlap_area, effective_support_ratio, support_centroid, primary_support_rect


def _future_sliver_penalty(
    view: DecisionStateView,
    candidate: CandidatePlacement,
    bounds: RectBounds,
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


def _fragmentation_penalty(
    view: DecisionStateView,
    candidate: CandidatePlacement,
    bounds: RectBounds,
) -> float:
    component_bounds = candidate.support_component_bounds
    if component_bounds is None:
        return 0.0

    component_min_x, component_min_y, component_max_x, component_max_y = component_bounds
    component_depth = component_max_x - component_min_x
    component_width = component_max_y - component_min_y
    component_area = max(view.config.geometry_epsilon, component_depth * component_width)
    candidate_depth = max(view.config.geometry_epsilon, bounds[2] - bounds[0])
    min_future_depth = float(view.config.dimension_ranges[0][0])
    min_future_width = float(view.config.dimension_ranges[1][0])

    strip_dimensions = (
        (bounds[0] - component_min_x, component_width),
        (component_max_x - bounds[2], component_width),
        (candidate_depth, bounds[1] - component_min_y),
        (candidate_depth, component_max_y - bounds[3]),
    )
    usable_strip_areas: list[float] = []
    narrow_strip_count = 0
    for depth, width in strip_dimensions:
        if depth <= view.config.geometry_epsilon or width <= view.config.geometry_epsilon:
            continue
        if depth + view.config.geometry_epsilon >= min_future_depth and width + view.config.geometry_epsilon >= min_future_width:
            usable_strip_areas.append(depth * width)
            continue
        narrow_strip_count += 1
    if component_area <= view.config.geometry_epsilon:
        return float(0.5 * narrow_strip_count)
    return float(max(0, len(usable_strip_areas) - 1) + (0.5 * narrow_strip_count))


def _contact_metrics(
    view: DecisionStateView,
    bounds: RectBounds,
    candidate: CandidatePlacement,
    orientation: OrientationOption,
    *,
    support_overlap_area: float,
    support_ratio: float,
    weights: ScoreWeights,
) -> tuple[float, float]:
    candidate_bottom_z = candidate.position[2] + orientation.bottom_z
    candidate_top_z = candidate.position[2] + orientation.top_z
    candidate_height = max(view.config.geometry_epsilon, candidate_top_z - candidate_bottom_z)
    footprint_area = max(view.config.geometry_epsilon, orientation.footprint_depth * orientation.footprint_width)
    x_face_area = max(view.config.geometry_epsilon, orientation.footprint_width * candidate_height)
    y_face_area = max(view.config.geometry_epsilon, orientation.footprint_depth * candidate_height)

    bottom_area = footprint_area if candidate_bottom_z <= weights.flush_contact_tolerance else max(0.0, support_overlap_area)
    front_area = 0.0
    back_area = 0.0
    left_area = 0.0
    right_area = 0.0

    if bounds[0] <= weights.flush_contact_tolerance:
        front_area += x_face_area
    if abs(view.truck.depth - bounds[2]) <= weights.flush_contact_tolerance:
        back_area += x_face_area
    if bounds[1] <= weights.flush_contact_tolerance:
        left_area += y_face_area
    if abs(view.truck.width - bounds[3]) <= weights.flush_contact_tolerance:
        right_area += y_face_area

    for placed_box in view.placed_boxes:
        z_overlap = _interval_overlap(candidate_bottom_z, candidate_top_z, placed_box.bottom_z, placed_box.top_z)
        if z_overlap <= view.config.geometry_epsilon:
            continue
        y_overlap = _interval_overlap(bounds[1], bounds[3], placed_box.min_y, placed_box.max_y)
        x_overlap = _interval_overlap(bounds[0], bounds[2], placed_box.min_x, placed_box.max_x)
        if abs(placed_box.max_x - bounds[0]) <= weights.flush_contact_tolerance and y_overlap > view.config.geometry_epsilon:
            front_area += y_overlap * z_overlap
        if abs(placed_box.min_x - bounds[2]) <= weights.flush_contact_tolerance and y_overlap > view.config.geometry_epsilon:
            back_area += y_overlap * z_overlap
        if abs(placed_box.max_y - bounds[1]) <= weights.flush_contact_tolerance and x_overlap > view.config.geometry_epsilon:
            left_area += x_overlap * z_overlap
        if abs(placed_box.min_y - bounds[3]) <= weights.flush_contact_tolerance and x_overlap > view.config.geometry_epsilon:
            right_area += x_overlap * z_overlap

    bottom_area = min(bottom_area, footprint_area)
    front_area = min(front_area, x_face_area)
    back_area = min(back_area, x_face_area)
    left_area = min(left_area, y_face_area)
    right_area = min(right_area, y_face_area)

    total_contact_area = bottom_area + front_area + back_area + left_area + right_area
    shared_contact_area_ratio = total_contact_area / (footprint_area + (2.0 * x_face_area) + (2.0 * y_face_area))
    front_contact_fraction = front_area / x_face_area
    back_contact_fraction = back_area / x_face_area
    left_contact_fraction = left_area / y_face_area
    right_contact_fraction = right_area / y_face_area

    wall_lock_bonus = (
        back_contact_fraction * max(left_contact_fraction, right_contact_fraction, front_contact_fraction, support_ratio)
        + (0.35 * max(left_contact_fraction, right_contact_fraction) * support_ratio)
    )

    return float(min(1.0, shared_contact_area_ratio)), float(min(2.0, wall_lock_bonus))


def _footprint_match_below(
    view: DecisionStateView,
    bounds: RectBounds,
    candidate: CandidatePlacement,
    orientation: OrientationOption,
    *,
    support_ratio: float,
    primary_support_rect: RectBounds | None,
) -> float:
    z_support = candidate.position[2] + orientation.bottom_z
    if z_support <= view.config.support_plane_epsilon or primary_support_rect is None:
        return 0.0
    support_depth = max(view.config.geometry_epsilon, primary_support_rect[2] - primary_support_rect[0])
    support_width = max(view.config.geometry_epsilon, primary_support_rect[3] - primary_support_rect[1])
    support_area = support_depth * support_width
    candidate_depth = max(view.config.geometry_epsilon, orientation.footprint_depth)
    candidate_width = max(view.config.geometry_epsilon, orientation.footprint_width)
    candidate_area = candidate_depth * candidate_width

    depth_match = min(candidate_depth, support_depth) / max(candidate_depth, support_depth)
    width_match = min(candidate_width, support_width) / max(candidate_width, support_width)
    candidate_center_x = (bounds[0] + bounds[2]) * 0.5
    candidate_center_y = (bounds[1] + bounds[3]) * 0.5
    support_center_x = (primary_support_rect[0] + primary_support_rect[2]) * 0.5
    support_center_y = (primary_support_rect[1] + primary_support_rect[3]) * 0.5
    offset_x = abs(candidate_center_x - support_center_x) / max(
        view.config.geometry_epsilon,
        max(candidate_depth, support_depth) * 0.5,
    )
    offset_y = abs(candidate_center_y - support_center_y) / max(
        view.config.geometry_epsilon,
        max(candidate_width, support_width) * 0.5,
    )
    center_alignment = max(0.0, 1.0 - (0.5 * (offset_x + offset_y)))
    area_match = min(candidate_area, support_area) / max(candidate_area, support_area)
    return float(support_ratio * depth_match * width_match * center_alignment * area_match)


def _cavity_penalty(
    view: DecisionStateView,
    candidate: CandidatePlacement,
    bounds: RectBounds,
    *,
    future_sliver_penalty: float,
    fragmentation_penalty: float,
) -> float:
    component_bounds = candidate.support_component_bounds
    if component_bounds is None:
        return float(0.5 * future_sliver_penalty + 0.5 * fragmentation_penalty)
    component_depth = max(view.config.geometry_epsilon, component_bounds[2] - component_bounds[0])
    component_width = max(view.config.geometry_epsilon, component_bounds[3] - component_bounds[1])
    candidate_depth = max(view.config.geometry_epsilon, bounds[2] - bounds[0])
    min_future_depth = float(view.config.dimension_ranges[0][0])
    min_future_width = float(view.config.dimension_ranges[1][0])

    front_strip = max(0.0, bounds[0] - component_bounds[0])
    back_strip = max(0.0, component_bounds[2] - bounds[2])
    left_strip = max(0.0, bounds[1] - component_bounds[1])
    right_strip = max(0.0, component_bounds[3] - bounds[3])

    penalty = 0.0
    for strip_depth in (front_strip, back_strip):
        if strip_depth <= view.config.geometry_epsilon or strip_depth >= min_future_depth:
            continue
        thinness = (min_future_depth - strip_depth) / max(min_future_depth, view.config.geometry_epsilon)
        run_factor = max(1.0, component_width / max(min_future_width, view.config.geometry_epsilon))
        penalty += thinness * run_factor
    for strip_width in (left_strip, right_strip):
        if strip_width <= view.config.geometry_epsilon or strip_width >= min_future_width:
            continue
        thinness = (min_future_width - strip_width) / max(min_future_width, view.config.geometry_epsilon)
        run_factor = max(1.0, candidate_depth / max(min_future_depth, view.config.geometry_epsilon))
        penalty += thinness * run_factor
    trench_factor = candidate_depth / max(component_depth, view.config.geometry_epsilon)
    penalty += trench_factor * (future_sliver_penalty + (0.75 * fragmentation_penalty))
    return float(penalty)


def _skyline_roughness(
    view: DecisionStateView,
    bounds: RectBounds,
    candidate: CandidatePlacement,
    orientation: OrientationOption,
) -> float:
    candidate_top_z = candidate.position[2] + orientation.top_z
    candidate_depth = max(view.config.geometry_epsilon, bounds[2] - bounds[0])
    neighboring_tops = [
        placed_box.top_z
        for placed_box in view.placed_boxes
        if _interval_overlap(bounds[1], bounds[3], placed_box.min_y, placed_box.max_y) > view.config.geometry_epsilon
        and placed_box.max_x >= bounds[0] - candidate_depth - view.config.geometry_epsilon
        and placed_box.min_x <= bounds[2] + view.config.geometry_epsilon
    ]
    if not neighboring_tops:
        if candidate.position[2] + orientation.bottom_z <= view.config.support_plane_epsilon:
            return 0.0
        return float(candidate_top_z / view.truck.height)
    top_values = [*neighboring_tops, candidate_top_z]
    mean_top = sum(top_values) / len(top_values)
    average_absolute_deviation = sum(abs(value - mean_top) for value in top_values) / len(top_values)
    top_range = max(top_values) - min(top_values)
    height_scale = max(view.truck.height, view.config.geometry_epsilon)
    return float((average_absolute_deviation / height_scale) + (0.5 * top_range / height_scale))


def _instability_risk(
    view: DecisionStateView,
    bounds: RectBounds,
    candidate: CandidatePlacement,
    orientation: OrientationOption,
    *,
    support_ratio: float,
    support_centroid: tuple[float, float] | None,
) -> float:
    candidate_top_z = candidate.position[2] + orientation.top_z
    z_support = candidate.position[2] + orientation.bottom_z
    height_factor = candidate_top_z / max(view.truck.height, view.config.geometry_epsilon)
    slenderness = orientation.height / max(
        view.config.geometry_epsilon,
        min(orientation.footprint_depth, orientation.footprint_width),
    )
    slenderness_penalty = max(0.0, min(1.5, slenderness - 0.85))
    if support_centroid is None:
        center_offset = 0.0 if z_support <= view.config.support_plane_epsilon else 1.0
    else:
        candidate_center_x = (bounds[0] + bounds[2]) * 0.5
        candidate_center_y = (bounds[1] + bounds[3]) * 0.5
        offset_x = abs(candidate_center_x - support_centroid[0]) / max(
            view.config.geometry_epsilon,
            orientation.footprint_depth * 0.5,
        )
        offset_y = abs(candidate_center_y - support_centroid[1]) / max(
            view.config.geometry_epsilon,
            orientation.footprint_width * 0.5,
        )
        center_offset = min(1.5, 0.5 * (offset_x + offset_y))
    risk = (
        (0.6 * max(0.0, 1.0 - support_ratio))
        + (0.2 * height_factor)
        + (0.15 * slenderness_penalty)
        + (0.35 * center_offset)
    )
    if z_support <= view.config.support_plane_epsilon:
        risk *= 0.5
    return float(max(0.0, risk))


def compute_score_breakdown(
    view: DecisionStateView,
    candidate: CandidatePlacement,
    orientation: OrientationOption,
    *,
    support_ratio: float | None,
    weights: ScoreWeights,
) -> ScoreBreakdown:
    bounds = candidate_bounds(candidate, orientation)
    candidate_max_x = bounds[2]
    frontier_jump = max(0.0, candidate_max_x - view.current_max_x)
    exact_density_after = _exact_density_after(view, candidate_max_x)

    support_overlap_area, support_reward, support_centroid, primary_support_rect = _support_geometry_metrics(
        view,
        bounds,
        candidate,
        orientation,
        support_ratio=support_ratio,
    )
    shared_contact_area_ratio, wall_lock_bonus = _contact_metrics(
        view,
        bounds,
        candidate,
        orientation,
        support_overlap_area=support_overlap_area,
        support_ratio=support_reward,
        weights=weights,
    )
    footprint_match_below = _footprint_match_below(
        view,
        bounds,
        candidate,
        orientation,
        support_ratio=support_reward,
        primary_support_rect=primary_support_rect,
    )
    future_sliver_penalty = _future_sliver_penalty(view, candidate, bounds)
    fragmentation_penalty = _fragmentation_penalty(view, candidate, bounds)
    cavity_penalty = _cavity_penalty(
        view,
        candidate,
        bounds,
        future_sliver_penalty=future_sliver_penalty,
        fragmentation_penalty=fragmentation_penalty,
    )
    skyline_roughness = _skyline_roughness(view, bounds, candidate, orientation)
    instability_risk = _instability_risk(
        view,
        bounds,
        candidate,
        orientation,
        support_ratio=support_reward,
        support_centroid=support_centroid,
    )

    total_score = (
        (weights.exact_density_weight * exact_density_after)
        + (weights.contact_area_weight * shared_contact_area_ratio)
        + (weights.gamma * support_reward)
        + (weights.wall_lock_bonus_weight * wall_lock_bonus)
        + (weights.footprint_match_weight * footprint_match_below)
        - (weights.frontier_jump_weight * frontier_jump)
        - (weights.cavity_penalty_weight * cavity_penalty)
        - (weights.skyline_roughness_weight * skyline_roughness)
        - (weights.instability_risk_weight * instability_risk)
    )
    return ScoreBreakdown(
        total_score=float(total_score),
        exact_density_after=float(exact_density_after),
        frontier_jump=float(frontier_jump),
        cavity_penalty=float(cavity_penalty),
        support_reward=float(support_reward),
        shared_contact_area_ratio=float(shared_contact_area_ratio),
        wall_lock_bonus=float(wall_lock_bonus),
        footprint_match_below=float(footprint_match_below),
        skyline_roughness=float(skyline_roughness),
        instability_risk=float(instability_risk),
    )


def score_breakdown_with_support_ratio(
    score: ScoreBreakdown,
    *,
    support_ratio: float | None,
    weights: ScoreWeights,
) -> ScoreBreakdown:
    support_reward = float(0.0 if support_ratio is None else support_ratio)
    total_score = (
        (weights.exact_density_weight * score.exact_density_after)
        + (weights.contact_area_weight * score.shared_contact_area_ratio)
        + (weights.gamma * support_reward)
        + (weights.wall_lock_bonus_weight * score.wall_lock_bonus)
        + (weights.footprint_match_weight * score.footprint_match_below)
        - (weights.frontier_jump_weight * score.frontier_jump)
        - (weights.cavity_penalty_weight * score.cavity_penalty)
        - (weights.skyline_roughness_weight * score.skyline_roughness)
        - (weights.instability_risk_weight * score.instability_risk)
    )
    return ScoreBreakdown(
        total_score=float(total_score),
        exact_density_after=score.exact_density_after,
        frontier_jump=score.frontier_jump,
        cavity_penalty=score.cavity_penalty,
        support_reward=support_reward,
        shared_contact_area_ratio=score.shared_contact_area_ratio,
        wall_lock_bonus=score.wall_lock_bonus,
        footprint_match_below=score.footprint_match_below,
        skyline_roughness=score.skyline_roughness,
        instability_risk=score.instability_risk,
    )
