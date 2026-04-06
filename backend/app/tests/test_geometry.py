from __future__ import annotations

import pytest

from app.engine.geometry import compute_box_geometry, corners_within_truck, normalize_quaternion, obb_intersects
from app.models.entities import Truck


def test_quaternion_normalization() -> None:
    normalized = normalize_quaternion((2.0, 0.0, 0.0, 0.0))
    assert normalized == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_oriented_box_corner_generation_for_identity_quaternion() -> None:
    geometry = compute_box_geometry(
        position=(1.0, 2.0, 3.0),
        dimensions=(2.0, 4.0, 6.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        vertical_axis_cos_tolerance=0.99,
    )
    assert geometry.corners.min(axis=0) == pytest.approx((0.0, 0.0, 0.0))
    assert geometry.corners.max(axis=0) == pytest.approx((2.0, 4.0, 6.0))


def test_sat_obb_collision_treats_touching_faces_as_non_intersection() -> None:
    box_a = compute_box_geometry(
        position=(0.5, 0.5, 0.5),
        dimensions=(1.0, 1.0, 1.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        vertical_axis_cos_tolerance=0.99,
    )
    box_b = compute_box_geometry(
        position=(1.5, 0.5, 0.5),
        dimensions=(1.0, 1.0, 1.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        vertical_axis_cos_tolerance=0.99,
    )
    box_c = compute_box_geometry(
        position=(1.49, 0.5, 0.5),
        dimensions=(1.0, 1.0, 1.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        vertical_axis_cos_tolerance=0.99,
    )
    assert obb_intersects(box_a, box_b, epsilon=1e-5) is False
    assert obb_intersects(box_a, box_c, epsilon=1e-5) is True


def test_bounds_check_rejects_box_outside_truck() -> None:
    geometry = compute_box_geometry(
        position=(-0.1, 0.5, 0.5),
        dimensions=(1.0, 1.0, 1.0),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        vertical_axis_cos_tolerance=0.99,
    )
    valid, details = corners_within_truck(Truck(), geometry, epsilon=1e-6)
    assert valid is False
    assert details["min_x"] < 0.0

