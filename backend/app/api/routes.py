from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.serializers import (
    serialize_action,
    serialize_challenge_place_response,
    serialize_challenge_start,
    serialize_challenge_status_response,
    serialize_local_state,
)
from app.models.api import (
    ChallengePlaceResponse,
    ChallengeStartRequest,
    ChallengeStartResponse,
    ChallengeStatusResponse,
    HealthResponse,
    LocalGameStateResponse,
    LocalStartRequest,
    LocalStartResponse,
    PlaceRequest,
    PreviewRequest,
    PreviewResponse,
    StopRequest,
)
from app.models.entities import PlacementAction, PreviewAction

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


@router.get("/challenge/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.post("/challenge/api/start", response_model=ChallengeStartResponse)
async def challenge_start_game(payload: ChallengeStartRequest, request: Request) -> ChallengeStartResponse:
    state = _services(request).episode_service.start_episode(
        mode=payload.mode,
        seed=None,
        api_key=payload.api_key,
        api_variant="challenge",
    )
    return ChallengeStartResponse(**serialize_challenge_start(state))


@router.post("/challenge/api/place", response_model=ChallengePlaceResponse)
async def challenge_place_box(payload: PlaceRequest, request: Request) -> ChallengePlaceResponse:
    action = PlacementAction(
        box_id=payload.box_id,
        position=payload.position,
        orientation_wxyz=payload.orientation_wxyz,
    )
    state = _services(request).episode_service.place_box(
        payload.game_id,
        action,
        expected_api_variant="challenge",
    )
    return ChallengePlaceResponse(**serialize_challenge_place_response(state))


@router.get("/challenge/api/status/{game_id}", response_model=ChallengeStatusResponse)
async def challenge_status(game_id: str, request: Request) -> ChallengeStatusResponse:
    state = _services(request).episode_service.get_state(game_id, expected_api_variant="challenge")
    return ChallengeStatusResponse(**serialize_challenge_status_response(state))


@router.post("/challenge/api/stop", response_model=ChallengePlaceResponse)
async def challenge_stop(payload: StopRequest, request: Request) -> ChallengePlaceResponse:
    state = _services(request).episode_service.stop_episode(payload.game_id, expected_api_variant="challenge")
    return ChallengePlaceResponse(**serialize_challenge_place_response(state))


@router.post("/local/api/start", response_model=LocalStartResponse)
async def local_start_game(payload: LocalStartRequest, request: Request) -> LocalStartResponse:
    state = _services(request).episode_service.start_episode(
        mode=payload.mode,
        seed=payload.seed,
        api_key=payload.api_key,
        api_variant="local",
        enable_local_timeout=True,
        auto_terminate_on_no_feasible_placement=True,
    )
    return LocalStartResponse(**serialize_challenge_start(state))


@router.post("/local/api/place", response_model=LocalGameStateResponse)
async def local_place_box(payload: PlaceRequest, request: Request) -> LocalGameStateResponse:
    action = PlacementAction(
        box_id=payload.box_id,
        position=payload.position,
        orientation_wxyz=payload.orientation_wxyz,
    )
    state = _services(request).episode_service.place_box(
        payload.game_id,
        action,
        expected_api_variant="local",
    )
    return LocalGameStateResponse(**serialize_local_state(state, request.app.state.services.engine.config))


@router.get("/local/api/status/{game_id}", response_model=LocalGameStateResponse)
async def local_status(game_id: str, request: Request) -> LocalGameStateResponse:
    state = _services(request).episode_service.get_state(game_id, expected_api_variant="local")
    return LocalGameStateResponse(**serialize_local_state(state, request.app.state.services.engine.config))


@router.post("/local/api/stop", response_model=LocalGameStateResponse)
async def local_stop(payload: StopRequest, request: Request) -> LocalGameStateResponse:
    state = _services(request).episode_service.stop_episode(payload.game_id, expected_api_variant="local")
    return LocalGameStateResponse(**serialize_local_state(state, request.app.state.services.engine.config))


@router.post("/local/api/preview", response_model=PreviewResponse)
async def preview(payload: PreviewRequest, request: Request) -> PreviewResponse:
    validation, support_aligned, nearby_valid, any_valid = _services(request).preview_service.update_preview(
        payload.game_id,
        PreviewAction(
            box_id=payload.box_id,
            position=payload.position,
            orientation_wxyz=payload.orientation_wxyz,
        ),
        expected_api_variant="local",
    )
    state = _services(request).episode_service.get_state(payload.game_id, expected_api_variant="local")
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
