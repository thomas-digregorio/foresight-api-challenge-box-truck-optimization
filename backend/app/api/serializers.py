from __future__ import annotations

from typing import Any

from app.models.entities import GameState, PlacementAction


def serialize_box(box: Any) -> dict[str, Any]:
    payload = {
        "id": box.id,
        "dimensions": tuple(float(value) for value in box.dimensions),
        "weight": float(box.weight),
    }
    if hasattr(box, "position"):
        payload["position"] = tuple(float(value) for value in box.position)
    if hasattr(box, "orientation_wxyz"):
        payload["orientation_wxyz"] = tuple(float(value) for value in box.orientation_wxyz)
    return payload


def serialize_truck(state: GameState) -> dict[str, Any]:
    return {
        "depth": float(state.truck.depth),
        "width": float(state.truck.width),
        "height": float(state.truck.height),
    }


def serialize_action(action: PlacementAction | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return {
        "box_id": action.box_id,
        "position": tuple(float(value) for value in action.position),
        "orientation_wxyz": tuple(float(value) for value in action.orientation_wxyz),
    }


def serialize_state(state: GameState) -> dict[str, Any]:
    return {
        "game_id": state.game_id,
        "status": "terminated" if state.game_status != "in_progress" else "ok",
        "truck": serialize_truck(state),
        "placed_boxes": [serialize_box(box) for box in state.placed_boxes],
        "current_box": serialize_box(state.current_box) if state.current_box is not None else None,
        "boxes_remaining": state.boxes_remaining,
        "density": float(state.density),
        "game_status": state.game_status,
        "termination_reason": state.termination_reason,
        "mode": state.mode,
        "created_at": state.created_at.isoformat(),
        "current_box_started_at": None if state.current_box_started_at is None else state.current_box_started_at.isoformat(),
        "current_box_deadline": None if state.current_box_deadline is None else state.current_box_deadline.isoformat(),
        "timeout_seconds": float(state.timeout_seconds),
    }

