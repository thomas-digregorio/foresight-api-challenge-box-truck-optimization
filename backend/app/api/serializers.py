from __future__ import annotations

from typing import Any

from app.models.entities import EngineConfig, GameState, PlacementAction


def serialize_current_box(box: Any) -> dict[str, Any]:
    return {
        "id": box.id,
        "dimensions": tuple(float(value) for value in box.dimensions),
        "weight": float(box.weight),
    }


def serialize_placed_box(box: Any) -> dict[str, Any]:
    return {
        "id": box.id,
        "dimensions": tuple(float(value) for value in box.dimensions),
        "position": tuple(float(value) for value in box.position),
        "orientation_wxyz": tuple(float(value) for value in box.orientation_wxyz),
    }


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


def serialize_challenge_start(state: GameState) -> dict[str, Any]:
    return {
        "game_id": state.game_id,
        "truck": serialize_truck(state),
        "current_box": None if state.current_box is None else serialize_current_box(state.current_box),
        "boxes_remaining": state.boxes_remaining,
        "mode": state.mode,
    }


def serialize_challenge_place_response(state: GameState) -> dict[str, Any]:
    return {
        "status": "terminated" if state.game_status != "in_progress" else "ok",
        "placed_boxes": [serialize_placed_box(box) for box in state.placed_boxes],
        "current_box": None if state.current_box is None else serialize_current_box(state.current_box),
        "boxes_remaining": state.boxes_remaining,
        "density": float(state.density),
        "game_status": state.game_status if state.game_status == "in_progress" else "completed",
        "termination_reason": state.termination_reason,
    }


def serialize_challenge_status_response(state: GameState) -> dict[str, Any]:
    return {
        "game_id": state.game_id,
        "game_status": state.game_status if state.game_status == "in_progress" else "completed",
        "mode": state.mode,
        "boxes_placed": len(state.placed_boxes),
        "boxes_remaining": state.boxes_remaining,
        "density": float(state.density),
        "placed_boxes": [serialize_placed_box(box) for box in state.placed_boxes],
        "current_box": None if state.current_box is None else serialize_current_box(state.current_box),
    }


def serialize_local_state(state: GameState, config: EngineConfig) -> dict[str, Any]:
    loading_guide_x = state.metadata.get("loading_guide_x")
    if loading_guide_x is None and state.metadata.get("api_variant") == "local":
        loading_guide_x = config.local_loading_guide_x
    return {
        "game_id": state.game_id,
        "status": "terminated" if state.game_status != "in_progress" else "ok",
        "truck": serialize_truck(state),
        "placed_boxes": [serialize_placed_box(box) for box in state.placed_boxes],
        "current_box": None if state.current_box is None else serialize_current_box(state.current_box),
        "boxes_remaining": state.boxes_remaining,
        "density": float(state.density),
        "game_status": state.game_status,
        "termination_reason": state.termination_reason,
        "mode": state.mode,
        "created_at": state.created_at.isoformat(),
        "current_box_started_at": None if state.current_box_started_at is None else state.current_box_started_at.isoformat(),
        "current_box_deadline": None if state.current_box_deadline is None else state.current_box_deadline.isoformat(),
        "timeout_seconds": float(state.timeout_seconds),
        "loading_guide_x": None if loading_guide_x is None else float(loading_guide_x),
    }
