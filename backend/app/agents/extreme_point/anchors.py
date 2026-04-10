from __future__ import annotations

from dataclasses import dataclass

from app.agents.extreme_point.orientations import canonicalize_quaternion_sign
from app.agents.extreme_point.state_view import DecisionStateView
from app.agents.extreme_point.types import OrientationOption, SupportPlane

ANCHOR_STYLES: tuple[str, ...] = (
    "min_x_min_y",
    "min_x_max_y",
    "max_x_min_y",
    "max_x_max_y",
    "min_x_center_y",
    "max_x_center_y",
    "center_x_min_y",
    "center_x_max_y",
    "center_x_center_y",
)


@dataclass(slots=True, frozen=True)
class EdgeAnchors:
    min_x: tuple[float, ...]
    max_x: tuple[float, ...]
    center_x: tuple[float, ...]
    min_y: tuple[float, ...]
    max_y: tuple[float, ...]
    center_y: tuple[float, ...]


def candidate_dedup_key(
    position: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
) -> tuple[float, ...]:
    normalized = canonicalize_quaternion_sign(orientation_wxyz)
    rounded_position = tuple(round(value, 6) for value in position)
    rounded_orientation = tuple(round(value, 6) for value in normalized)
    return rounded_position + rounded_orientation


def _clamped_sorted_anchors(
    anchors: set[float],
    *,
    lower_bound: float,
    upper_bound: float,
    epsilon: float,
) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                round(anchor, 6)
                for anchor in anchors
                if lower_bound - epsilon <= anchor <= upper_bound + epsilon
            }
        )
    )


def _gap_center_anchors(
    boundaries: set[float],
    *,
    lower_bound: float,
    upper_bound: float,
    required_span: float,
    epsilon: float,
) -> tuple[float, ...]:
    ordered = sorted(
        {
            round(boundary, 6)
            for boundary in boundaries
            if lower_bound - epsilon <= boundary <= upper_bound + epsilon
        }
    )
    centers: set[float] = set()
    if not ordered:
        return ()
    for start, end in zip(ordered, ordered[1:]):
        if end - start + epsilon < required_span:
            continue
        centers.add(float(round((start + end) * 0.5, 6)))
    return tuple(sorted(centers))


def generate_edge_anchors(
    view: DecisionStateView,
    support_plane: SupportPlane,
    orientation: OrientationOption,
    *,
    anchor_rectangles: tuple[tuple[float, float, float, float], ...] | None = None,
) -> EdgeAnchors:
    candidate_bottom_z = support_plane.z_support
    candidate_top_z = support_plane.z_support + orientation.height
    relevant_obstacles = view.obstacle_rectangles_for_z_range(candidate_bottom_z, candidate_top_z)
    rectangles = support_plane.support_rectangles if anchor_rectangles is None else anchor_rectangles
    min_x_anchors = {
        0.0,
        float(view.current_max_x),
    }
    max_x_anchors = {
        float(view.current_max_x),
        float(view.truck.depth),
    }
    min_y_anchors = {
        0.0,
    }
    max_y_anchors = {
        float(view.truck.width),
    }
    for min_x, min_y, max_x, max_y in relevant_obstacles:
        min_x_anchors.add(float(max_x))
        max_x_anchors.add(float(min_x))
        min_y_anchors.add(float(max_y))
        max_y_anchors.add(float(min_y))
    for min_x, min_y, max_x, max_y in rectangles:
        min_x_anchors.add(float(min_x))
        min_x_anchors.add(float(max_x))
        max_x_anchors.add(float(min_x))
        max_x_anchors.add(float(max_x))
        min_y_anchors.add(float(min_y))
        min_y_anchors.add(float(max_y))
        max_y_anchors.add(float(min_y))
        max_y_anchors.add(float(max_y))

    epsilon = view.config.geometry_epsilon
    x_boundaries = set(min_x_anchors) | set(max_x_anchors)
    y_boundaries = set(min_y_anchors) | set(max_y_anchors)
    return EdgeAnchors(
        min_x=_clamped_sorted_anchors(
            min_x_anchors,
            lower_bound=0.0,
            upper_bound=view.truck.depth,
            epsilon=epsilon,
        ),
        max_x=_clamped_sorted_anchors(
            max_x_anchors,
            lower_bound=0.0,
            upper_bound=view.truck.depth,
            epsilon=epsilon,
        ),
        center_x=_gap_center_anchors(
            x_boundaries,
            lower_bound=0.0,
            upper_bound=view.truck.depth,
            required_span=orientation.footprint_depth,
            epsilon=epsilon,
        ),
        min_y=_clamped_sorted_anchors(
            min_y_anchors,
            lower_bound=0.0,
            upper_bound=view.truck.width,
            epsilon=epsilon,
        ),
        max_y=_clamped_sorted_anchors(
            max_y_anchors,
            lower_bound=0.0,
            upper_bound=view.truck.width,
            epsilon=epsilon,
        ),
        center_y=_gap_center_anchors(
            y_boundaries,
            lower_bound=0.0,
            upper_bound=view.truck.width,
            required_span=orientation.footprint_width,
            epsilon=epsilon,
        ),
    )
