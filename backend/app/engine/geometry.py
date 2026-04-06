from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation
from shapely.geometry import MultiPoint, Polygon

from app.models.entities import Truck


WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=float)


@dataclass(slots=True)
class OrientedBoxGeometry:
    center: np.ndarray
    dimensions: np.ndarray
    half_extents: np.ndarray
    axes: np.ndarray
    corners: np.ndarray
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    bottom_z: float
    top_z: float
    footprint_polygon: Polygon | None
    top_polygon: Polygon | None
    base_corners: np.ndarray | None
    top_corners: np.ndarray | None
    vertical_axis_index: int | None
    vertical_alignment: float
    is_gravity_compatible: bool


def wxyz_to_xyzw(quaternion_wxyz: Iterable[float]) -> tuple[float, float, float, float]:
    w, x, y, z = quaternion_wxyz
    return float(x), float(y), float(z), float(w)


def xyzw_to_wxyz(quaternion_xyzw: Iterable[float]) -> tuple[float, float, float, float]:
    x, y, z, w = quaternion_xyzw
    return float(w), float(x), float(y), float(z)


def normalize_quaternion(quaternion_wxyz: Iterable[float], *, epsilon: float = 1e-12) -> tuple[float, float, float, float]:
    quat = np.asarray(tuple(quaternion_wxyz), dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= epsilon:
        raise ValueError("Quaternion norm must be non-zero.")
    quat /= norm
    return tuple(float(value) for value in quat)


def quaternion_to_matrix(quaternion_wxyz: Iterable[float]) -> np.ndarray:
    rotation = Rotation.from_quat(wxyz_to_xyzw(normalize_quaternion(quaternion_wxyz)))
    return rotation.as_matrix()


def rotation_from_wxyz(quaternion_wxyz: Iterable[float]) -> Rotation:
    return Rotation.from_quat(wxyz_to_xyzw(normalize_quaternion(quaternion_wxyz)))


def quaternion_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    return xyzw_to_wxyz(Rotation.from_matrix(matrix).as_quat())


def quaternion_from_euler_xyz_degrees(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    return xyzw_to_wxyz(Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_quat())


def stable_base_orientation_matrices() -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for up_idx in range(3):
        for forward_idx in range(3):
            if forward_idx == up_idx:
                continue
            side_idx = ({0, 1, 2} - {up_idx, forward_idx}).pop()
            matrix = np.zeros((3, 3), dtype=float)
            matrix[:, forward_idx] = np.array([1.0, 0.0, 0.0], dtype=float)
            matrix[:, up_idx] = np.array([0.0, 0.0, 1.0], dtype=float)
            matrix[:, side_idx] = np.array([0.0, 1.0, 0.0], dtype=float)
            if np.linalg.det(matrix) < 0:
                matrix[:, side_idx] *= -1.0
            matrices.append(matrix)
    unique: list[np.ndarray] = []
    for candidate in matrices:
        if not any(np.allclose(candidate, existing, atol=1e-8) for existing in unique):
            unique.append(candidate)
    return unique


def stable_orientation_quaternions(yaw_samples: int) -> list[tuple[float, float, float, float]]:
    quaternions: list[tuple[float, float, float, float]] = []
    base_matrices = stable_base_orientation_matrices()
    yaw_angles = np.linspace(0.0, 2.0 * np.pi, num=max(1, yaw_samples), endpoint=False)
    for base in base_matrices:
        for yaw in yaw_angles:
            world_yaw = Rotation.from_rotvec(np.array([0.0, 0.0, yaw], dtype=float)).as_matrix()
            matrix = world_yaw @ base
            quat = quaternion_from_matrix(matrix)
            if not any(np.allclose(quat, existing, atol=1e-6) or np.allclose(np.negative(quat), existing, atol=1e-6) for existing in quaternions):
                quaternions.append(quat)
    return quaternions


def _face_polygon_xy(points_3d: np.ndarray) -> Polygon:
    hull = MultiPoint([(float(point[0]), float(point[1])) for point in points_3d]).convex_hull
    if hull.geom_type != "Polygon":
        return Polygon()
    return hull


def compute_box_geometry(
    position: Iterable[float],
    dimensions: Iterable[float],
    orientation_wxyz: Iterable[float],
    *,
    vertical_axis_cos_tolerance: float,
) -> OrientedBoxGeometry:
    center = np.asarray(tuple(position), dtype=float)
    dims = np.asarray(tuple(dimensions), dtype=float)
    half_extents = dims / 2.0
    rotation = quaternion_to_matrix(orientation_wxyz)
    axes = rotation
    corners = []
    for signs in product((-1.0, 1.0), repeat=3):
        offset = (axes * (half_extents * np.asarray(signs, dtype=float))).sum(axis=1)
        corners.append(center + offset)
    corner_array = np.vstack(corners)
    alignments = np.abs(axes.T @ WORLD_UP)
    vertical_axis_index = int(np.argmax(alignments))
    vertical_alignment = float(alignments[vertical_axis_index])
    gravity_compatible = vertical_alignment >= vertical_axis_cos_tolerance

    footprint_polygon: Polygon | None = None
    top_polygon: Polygon | None = None
    base_corners: np.ndarray | None = None
    top_corners: np.ndarray | None = None

    if gravity_compatible:
        up_axis = axes[:, vertical_axis_index]
        if float(np.dot(up_axis, WORLD_UP)) < 0:
            up_axis = -up_axis
        remaining = [idx for idx in range(3) if idx != vertical_axis_index]
        axis_a = axes[:, remaining[0]]
        axis_b = axes[:, remaining[1]]
        half_a = half_extents[remaining[0]]
        half_b = half_extents[remaining[1]]
        vertical_half = half_extents[vertical_axis_index]
        bottom_center = center - up_axis * vertical_half
        top_center = center + up_axis * vertical_half
        base = []
        top = []
        for sign_a, sign_b in ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)):
            base.append(bottom_center + axis_a * half_a * sign_a + axis_b * half_b * sign_b)
            top.append(top_center + axis_a * half_a * sign_a + axis_b * half_b * sign_b)
        base_corners = np.vstack(base)
        top_corners = np.vstack(top)
        footprint_polygon = _face_polygon_xy(base_corners)
        top_polygon = _face_polygon_xy(top_corners)
        bottom_z = float(np.mean(base_corners[:, 2]))
        top_z = float(np.mean(top_corners[:, 2]))
    else:
        bottom_z = float(np.min(corner_array[:, 2]))
        top_z = float(np.max(corner_array[:, 2]))

    return OrientedBoxGeometry(
        center=center,
        dimensions=dims,
        half_extents=half_extents,
        axes=axes,
        corners=corner_array,
        aabb_min=np.min(corner_array, axis=0),
        aabb_max=np.max(corner_array, axis=0),
        bottom_z=bottom_z,
        top_z=top_z,
        footprint_polygon=footprint_polygon,
        top_polygon=top_polygon,
        base_corners=base_corners,
        top_corners=top_corners,
        vertical_axis_index=vertical_axis_index if gravity_compatible else None,
        vertical_alignment=vertical_alignment,
        is_gravity_compatible=gravity_compatible,
    )


def corners_within_truck(truck: Truck, geometry: OrientedBoxGeometry, *, epsilon: float) -> tuple[bool, dict[str, float]]:
    min_corner = geometry.aabb_min
    max_corner = geometry.aabb_max
    ok = bool(
        min_corner[0] >= -epsilon
        and min_corner[1] >= -epsilon
        and min_corner[2] >= -epsilon
        and max_corner[0] <= truck.depth + epsilon
        and max_corner[1] <= truck.width + epsilon
        and max_corner[2] <= truck.height + epsilon
    )
    return ok, {
        "min_x": float(min_corner[0]),
        "min_y": float(min_corner[1]),
        "min_z": float(min_corner[2]),
        "max_x": float(max_corner[0]),
        "max_y": float(max_corner[1]),
        "max_z": float(max_corner[2]),
    }


def obb_intersects(a: OrientedBoxGeometry, b: OrientedBoxGeometry, *, epsilon: float) -> bool:
    axes_a = a.axes
    axes_b = b.axes
    half_a = a.half_extents
    half_b = b.half_extents
    rotation = axes_a.T @ axes_b
    abs_rotation = np.abs(rotation)
    translation_world = b.center - a.center
    translation = axes_a.T @ translation_world

    for i in range(3):
        ra = half_a[i]
        rb = float(np.dot(half_b, abs_rotation[i, :]))
        if abs(float(translation[i])) > (ra + rb - epsilon):
            return False

    for j in range(3):
        ra = float(np.dot(half_a, abs_rotation[:, j]))
        rb = half_b[j]
        projection = float(np.dot(translation, rotation[:, j]))
        if abs(projection) > (ra + rb - epsilon):
            return False

    for i in range(3):
        for j in range(3):
            if abs_rotation[i, j] >= 1.0 - 1e-8:
                continue
            ra = half_a[(i + 1) % 3] * abs_rotation[(i + 2) % 3, j] + half_a[(i + 2) % 3] * abs_rotation[(i + 1) % 3, j]
            rb = half_b[(j + 1) % 3] * abs_rotation[i, (j + 2) % 3] + half_b[(j + 2) % 3] * abs_rotation[i, (j + 1) % 3]
            projection = abs(
                float(
                    translation[(i + 2) % 3] * rotation[(i + 1) % 3, j]
                    - translation[(i + 1) % 3] * rotation[(i + 2) % 3, j]
                )
            )
            if projection > (ra + rb - epsilon):
                return False
    return True
