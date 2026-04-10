from __future__ import annotations

from itertools import permutations

import numpy as np

from app.engine.geometry import compute_box_geometry, normalize_quaternion, quaternion_from_matrix, stable_base_orientation_matrices

from app.agents.extreme_point.types import OrientationOption


def canonicalize_quaternion_sign(quaternion_wxyz: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    normalized = normalize_quaternion(quaternion_wxyz)
    for value in normalized:
        if abs(value) <= 1e-12:
            continue
        if value < 0.0:
            return tuple(-component for component in normalized)
        break
    return normalized


def _world_axis_permutation(matrix: np.ndarray) -> tuple[int, int, int]:
    return tuple(int(np.argmax(np.abs(matrix[axis, :]))) for axis in range(3))


def _orientation_option_from_quaternion(
    *,
    index: int,
    permutation: tuple[int, int, int],
    dimensions: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    vertical_axis_cos_tolerance: float,
) -> OrientationOption:
    geometry = compute_box_geometry(
        (0.0, 0.0, 0.0),
        dimensions,
        orientation_wxyz,
        vertical_axis_cos_tolerance=vertical_axis_cos_tolerance,
    )
    if geometry.footprint_polygon is None:
        raise ValueError("Orthogonal orientation must produce a horizontal footprint.")
    min_x, min_y, max_x, max_y = geometry.footprint_polygon.bounds
    return OrientationOption(
        index=index,
        permutation=permutation,
        orientation_wxyz=canonicalize_quaternion_sign(orientation_wxyz),
        min_x=float(min_x),
        max_x=float(max_x),
        min_y=float(min_y),
        max_y=float(max_y),
        bottom_z=float(geometry.bottom_z),
        top_z=float(geometry.top_z),
        footprint_depth=float(max_x - min_x),
        footprint_width=float(max_y - min_y),
        height=float(geometry.top_z - geometry.bottom_z),
    )


def build_orientation_option_for_quaternion(
    dimensions: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    *,
    index: int,
    vertical_axis_cos_tolerance: float,
) -> OrientationOption:
    geometry = compute_box_geometry(
        (0.0, 0.0, 0.0),
        dimensions,
        orientation_wxyz,
        vertical_axis_cos_tolerance=vertical_axis_cos_tolerance,
    )
    permutation = tuple(sorted(range(3), key=lambda axis: (-abs(geometry.axes[0, axis]), axis)))  # best-effort fallback
    return _orientation_option_from_quaternion(
        index=index,
        permutation=permutation,  # type: ignore[arg-type]
        dimensions=dimensions,
        orientation_wxyz=orientation_wxyz,
        vertical_axis_cos_tolerance=vertical_axis_cos_tolerance,
    )


def get_orthogonal_orientations_wxyz() -> list[tuple[float, float, float, float]]:
    options = get_orthogonal_orientation_options((1.0, 1.0, 1.0), vertical_axis_cos_tolerance=0.99)
    return [option.orientation_wxyz for option in options]


def get_orthogonal_orientation_options(
    dimensions: tuple[float, float, float],
    *,
    vertical_axis_cos_tolerance: float,
) -> list[OrientationOption]:
    matrices = stable_base_orientation_matrices()
    permutation_index = {perm: idx for idx, perm in enumerate(permutations((0, 1, 2)))}
    ordered: list[tuple[int, tuple[int, int, int], tuple[float, float, float, float]]] = []
    for matrix in matrices:
        permutation = _world_axis_permutation(matrix)
        ordered.append(
            (
                permutation_index[permutation],
                permutation,
                canonicalize_quaternion_sign(quaternion_from_matrix(matrix)),
            )
        )
    ordered.sort(key=lambda item: item[0])
    options: list[OrientationOption] = []
    for index, (_, permutation, quaternion_wxyz) in enumerate(ordered):
        options.append(
            _orientation_option_from_quaternion(
                index=index,
                permutation=permutation,
                dimensions=dimensions,
                orientation_wxyz=quaternion_wxyz,
                vertical_axis_cos_tolerance=vertical_axis_cos_tolerance,
            )
        )
    return options
