from __future__ import annotations

from shapely.geometry import box
from shapely.ops import unary_union

from app.agents.extreme_point.state_view import DecisionStateView
from app.agents.extreme_point.types import SupportPlane


def _component_can_host_current_box(
    view: DecisionStateView,
    *,
    component_bounds: tuple[float, float, float, float],
) -> bool:
    if view.current_box is None:
        return True
    component_depth = component_bounds[2] - component_bounds[0]
    component_width = component_bounds[3] - component_bounds[1]
    dimensions = tuple(float(value) for value in view.current_box.dimensions)
    footprint_pairs = (
        (dimensions[0], dimensions[1]),
        (dimensions[0], dimensions[2]),
        (dimensions[1], dimensions[2]),
    )
    epsilon = view.config.geometry_epsilon
    return any(
        (
            component_depth + epsilon >= first and component_width + epsilon >= second
        )
        or (
            component_depth + epsilon >= second and component_width + epsilon >= first
        )
        for first, second in footprint_pairs
    )


def _plane_metrics(
    view: DecisionStateView,
    *,
    z_support: float,
    support_rectangles: tuple[tuple[float, float, float, float], ...],
) -> tuple[float, int, float, float, float]:
    if not support_rectangles:
        support_area = float(view.truck.depth * view.truck.width)
        component_count = 1
        max_component_area = support_area
        frontier_gap = max(0.0, 0.0 - view.current_max_x)
        quality_score = support_area + max_component_area
        return support_area, component_count, max_component_area, frontier_gap, quality_score

    polygons = [box(*rectangle) for rectangle in support_rectangles]
    support_union = unary_union(polygons)
    components = [support_union] if support_union.geom_type == "Polygon" else list(getattr(support_union, "geoms", []))
    support_area = float(support_union.area)
    component_areas = [float(component.area) for component in components]
    max_component_area = max(component_areas, default=0.0)
    plane_front_x = max((rectangle[2] for rectangle in support_rectangles), default=0.0)
    frontier_gap = max(0.0, view.current_max_x - plane_front_x)
    fit_component_area = 0.0
    fit_component_count = 0
    for component in components:
        component_bounds = tuple(float(value) for value in component.bounds)
        if not _component_can_host_current_box(view, component_bounds=component_bounds):
            continue
        fit_component_area += float(component.area)
        fit_component_count += 1
    unusable_area = max(0.0, support_area - fit_component_area)
    quality_score = (
        1.25 * max_component_area
        + 0.35 * support_area
        + 0.45 * fit_component_area
        + 0.15 * fit_component_count
        - 0.5 * frontier_gap
        - 0.35 * unusable_area
        - 0.2 * max(0, len(components) - 1)
        - 0.05 * z_support
    )
    return support_area, len(components), max_component_area, frontier_gap, quality_score


def extract_support_planes(view: DecisionStateView) -> list[SupportPlane]:
    floor_area = float(view.truck.depth * view.truck.width)
    planes: list[SupportPlane] = [
        SupportPlane(
            index=0,
            plane_id="floor",
            z_support=0.0,
            supporting_box_ids=(),
            support_rectangles=(),
            support_area=floor_area,
            component_count=1,
            max_component_area=floor_area,
            frontier_gap=max(0.0, view.current_max_x),
            quality_score=floor_area * 2.0,
        )
    ]

    stable_top_boxes = [
        box
        for box in view.placed_boxes
        if box.geometry.is_gravity_compatible and box.top_bounds is not None
    ]
    stable_top_boxes.sort(key=lambda box: (round(box.top_z, 6), box.id))

    grouped: list[list] = []
    for box in stable_top_boxes:
        if not grouped or abs(grouped[-1][0].top_z - box.top_z) > view.config.support_plane_epsilon:
            grouped.append([box])
        else:
            grouped[-1].append(box)

    for index, group in enumerate(grouped, start=1):
        supporting_ids = tuple(sorted(box.id for box in group))
        support_rectangles = tuple(box.top_bounds for box in group if box.top_bounds is not None)
        plane_height = float(sum(box.top_z for box in group) / len(group))
        plane_id = f"plane:{plane_height:.6f}:{','.join(supporting_ids)}"
        support_area, component_count, max_component_area, frontier_gap, quality_score = _plane_metrics(
            view,
            z_support=plane_height,
            support_rectangles=support_rectangles,
        )
        planes.append(
            SupportPlane(
                index=index,
                plane_id=plane_id,
                z_support=plane_height,
                supporting_box_ids=supporting_ids,
                support_rectangles=support_rectangles,
                support_area=support_area,
                component_count=component_count,
                max_component_area=max_component_area,
                frontier_gap=frontier_gap,
                quality_score=quality_score,
            )
        )
    planes.sort(
        key=lambda plane: (
            plane.index != 0,
            -round(plane.quality_score, 12),
            round(plane.frontier_gap, 12),
            round(plane.z_support, 6),
            plane.supporting_box_ids,
            plane.plane_id,
        )
    )
    return [
        SupportPlane(
            index=index,
            plane_id=plane.plane_id,
            z_support=plane.z_support,
            supporting_box_ids=plane.supporting_box_ids,
            support_rectangles=plane.support_rectangles,
            support_area=plane.support_area,
            component_count=plane.component_count,
            max_component_area=plane.max_component_area,
            frontier_gap=plane.frontier_gap,
            quality_score=plane.quality_score,
        )
        for index, plane in enumerate(planes)
    ]
