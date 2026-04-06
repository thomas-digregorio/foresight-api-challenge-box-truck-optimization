from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.serializers import serialize_action, serialize_box, serialize_state, serialize_truck
from app.models.api import (
    GameStateResponse,
    HealthResponse,
    PlaceRequest,
    PreviewRequest,
    PreviewResponse,
    StartRequest,
    StartResponse,
    StopRequest,
)
from app.models.entities import PlacementAction, PreviewAction

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


@router.get("/challenge/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.post("/challenge/api/start", response_model=StartResponse)
async def start_game(payload: StartRequest, request: Request) -> StartResponse:
    state = _services(request).episode_service.start_episode(mode=payload.mode, seed=payload.seed)
    return StartResponse(
        game_id=state.game_id,
        truck=serialize_truck(state),
        current_box=None if state.current_box is None else serialize_box(state.current_box),
        boxes_remaining=state.boxes_remaining,
        mode=state.mode,
    )


@router.post("/challenge/api/place", response_model=GameStateResponse)
async def place_box(payload: PlaceRequest, request: Request) -> GameStateResponse:
    action = PlacementAction(
        box_id=payload.box_id,
        position=payload.position,
        orientation_wxyz=payload.orientation_wxyz,
    )
    state = _services(request).episode_service.place_box(payload.game_id, action)
    return GameStateResponse(**serialize_state(state))


@router.get("/challenge/api/status/{game_id}", response_model=GameStateResponse)
async def status(game_id: str, request: Request) -> GameStateResponse:
    state = _services(request).episode_service.get_state(game_id)
    return GameStateResponse(**serialize_state(state))


@router.post("/challenge/api/stop", response_model=GameStateResponse)
async def stop(payload: StopRequest, request: Request) -> GameStateResponse:
    state = _services(request).episode_service.stop_episode(payload.game_id)
    return GameStateResponse(**serialize_state(state))


@router.post("/local/api/preview", response_model=PreviewResponse)
async def preview(payload: PreviewRequest, request: Request) -> PreviewResponse:
    validation, support_aligned, nearby_valid, any_valid = _services(request).preview_service.update_preview(
        payload.game_id,
        PreviewAction(
            box_id=payload.box_id,
            position=payload.position,
            orientation_wxyz=payload.orientation_wxyz,
        ),
    )
    state = _services(request).episode_service.get_state(payload.game_id)
    return PreviewResponse(
        is_valid=validation.is_valid,
        message=validation.message,
        category=validation.category,
        details=validation.details,
        support_ratio=validation.support_ratio,
        normalized_position=None if validation.normalized_action is None else validation.normalized_action.position,
        normalized_orientation_wxyz=None
        if validation.normalized_action is None
        else validation.normalized_action.orientation_wxyz,
        latest_valid_preview_action=serialize_action(state.latest_valid_preview_action),
        snap_suggestions={},
        repair_suggestions={
            "support_aligned": serialize_action(support_aligned),
            "nearby_valid": serialize_action(nearby_valid),
            "any_valid": serialize_action(any_valid),
        },
        game_status=state.game_status,
        termination_reason=state.termination_reason,
        current_box_deadline=None if state.current_box_deadline is None else state.current_box_deadline.isoformat(),
        density=state.density,
    )
