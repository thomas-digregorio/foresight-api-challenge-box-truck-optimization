from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from app.api.app import build_error_response, create_app
from app.api.routes import (
    challenge_place_box,
    challenge_start_game,
    challenge_status,
    local_start_game,
    local_status,
    preview,
)
from app.models.api import ChallengeStartRequest, LocalStartRequest, PlaceRequest, PreviewRequest
from app.models.entities import CurrentBox, PlacedBox


def make_request(app):
    return SimpleNamespace(app=app)


def test_challenge_start_defaults_to_public_compete_mode() -> None:
    app = create_app()

    response = asyncio.run(challenge_start_game(ChallengeStartRequest(api_key="local"), make_request(app)))

    assert response.mode == "compete"
    assert response.truck.model_dump() == {"depth": 2.0, "width": 2.6, "height": 2.75}


def test_challenge_place_response_uses_canonical_shape_without_local_fields() -> None:
    app = create_app()
    request = make_request(app)

    start = asyncio.run(challenge_start_game(ChallengeStartRequest(api_key="local", mode="dev"), request))
    state = app.state.services.registry.get(start.game_id)
    state.current_box = CurrentBox(id="manual-box", dimensions=(0.5, 0.5, 0.5), weight=5.0)
    state.remaining_boxes = []
    app.state.services.registry.update(state)

    response = asyncio.run(
        challenge_place_box(
            PlaceRequest(
                game_id=start.game_id,
                box_id="manual-box",
                position=(1.5, 1.3, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            request,
        )
    )

    payload = response.model_dump()
    assert set(payload) == {
        "status",
        "placed_boxes",
        "current_box",
        "boxes_remaining",
        "density",
        "game_status",
        "termination_reason",
    }
    assert payload["status"] == "terminated"
    assert payload["game_status"] == "completed"
    assert payload["placed_boxes"][0]["position"][0] > 0.78
    assert "weight" not in payload["placed_boxes"][0]


def test_challenge_status_response_excludes_local_helper_fields() -> None:
    app = create_app()
    request = make_request(app)

    start = asyncio.run(challenge_start_game(ChallengeStartRequest(api_key="local", mode="dev"), request))
    response = asyncio.run(challenge_status(start.game_id, request))

    payload = response.model_dump()
    assert set(payload) == {
        "game_id",
        "game_status",
        "mode",
        "boxes_placed",
        "boxes_remaining",
        "density",
        "placed_boxes",
        "current_box",
    }
    assert "truck" not in payload
    assert "status" not in payload
    assert "current_box_deadline" not in payload
    assert "timeout_seconds" not in payload


def test_local_status_keeps_local_timeout_and_loading_guide_fields() -> None:
    app = create_app()
    request = make_request(app)

    start = asyncio.run(local_start_game(LocalStartRequest(api_key="local-dev", mode="dev", seed=7), request))
    response = asyncio.run(local_status(start.game_id, request))

    payload = response.model_dump()
    assert payload["current_box_deadline"] is not None
    assert payload["timeout_seconds"] == 10.0
    assert payload["loading_guide_x"] == 0.78


def test_challenge_invalid_box_id_maps_to_documented_error_code() -> None:
    app = create_app()
    request = make_request(app)

    start = asyncio.run(challenge_start_game(ChallengeStartRequest(api_key="local", mode="dev"), request))
    try:
        asyncio.run(
            challenge_place_box(
                PlaceRequest(
                    game_id=start.game_id,
                    box_id="wrong-box-id",
                    position=(0.5, 1.3, 0.5),
                    orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
                ),
                request,
            )
        )
    except Exception as exc:  # noqa: BLE001
        status_code, payload = build_error_response("/challenge/api/place", exc)
    else:  # pragma: no cover
        raise AssertionError("Expected challenge_place_box to raise for invalid_box_id")

    assert status_code == 400
    assert payload["error"] == "invalid_box_id"


def test_invalid_place_action_maps_to_canonical_422_for_challenge_routes() -> None:
    app = create_app()
    request = make_request(app)

    start = asyncio.run(challenge_start_game(ChallengeStartRequest(api_key="local", mode="dev"), request))
    state = app.state.services.registry.get(start.game_id)
    state.current_box = CurrentBox(id="manual-box", dimensions=(1.0, 1.0, 1.0), weight=5.0)
    state.remaining_boxes = []
    app.state.services.registry.update(state)

    try:
        asyncio.run(
            challenge_place_box(
                PlaceRequest(
                    game_id=start.game_id,
                    box_id="manual-box",
                    position=(0.5, 0.5, 0.5),
                    orientation_wxyz=(0.0, 0.0, 0.0, 0.0),
                ),
                request,
            )
        )
    except Exception as exc:  # noqa: BLE001
        status_code, payload = build_error_response("/challenge/api/place", exc)
    else:  # pragma: no cover
        raise AssertionError("Expected challenge_place_box to raise for invalid quaternion")

    assert status_code == 422
    assert payload["error"] == "validation"
    assert "details" not in payload


def test_preview_returns_repair_suggestion_for_overlapping_default_pose() -> None:
    app = create_app()
    request = make_request(app)

    start = asyncio.run(local_start_game(LocalStartRequest(api_key="local-dev", mode="dev", seed=9), request))
    state = app.state.services.registry.get(start.game_id)
    state.current_box = CurrentBox(id="top-box", dimensions=(0.5, 0.5, 0.5), weight=5.0)
    state.remaining_boxes = []
    state.placed_boxes = [
        PlacedBox(
            id="base-box",
            dimensions=(0.5, 0.5, 0.5),
            weight=5.0,
            position=(0.31, 1.3, 0.25),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
    ]
    app.state.services.registry.update(state)

    response = asyncio.run(
        preview(
            PreviewRequest(
                game_id=start.game_id,
                box_id="top-box",
                position=(0.31, 1.3, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            request,
        )
    )

    payload = response.model_dump()
    assert payload["is_valid"] is False
    assert payload["category"] == "overlap"
    assert payload["repair_suggestions"]["support_aligned"] is not None
    assert payload["repair_suggestions"]["any_valid"] is not None


def test_local_and_challenge_route_families_do_not_mix() -> None:
    app = create_app()
    request = make_request(app)

    local = asyncio.run(local_start_game(LocalStartRequest(api_key="local-dev", mode="dev", seed=1), request))
    try:
        asyncio.run(challenge_status(local.game_id, request))
    except Exception as exc:  # noqa: BLE001
        status_code, payload = build_error_response("/challenge/api/status/test", exc)
    else:  # pragma: no cover
        raise AssertionError("Expected challenge_status to reject local game ids")

    assert status_code == 404
    assert payload["error"] == "invalid_game_id"


def test_shared_examples_and_readme_match_current_contract() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    challenge_start_example = json.loads((repo_root / "shared" / "api_examples" / "start.json").read_text())
    local_start_example = json.loads((repo_root / "shared" / "api_examples" / "local_start.json").read_text())
    place_example = json.loads((repo_root / "shared" / "api_examples" / "place.json").read_text())

    assert ChallengeStartRequest.model_validate(challenge_start_example).mode == "dev"
    assert LocalStartRequest.model_validate(local_start_example).seed == 42
    assert PlaceRequest.model_validate(place_example).box_id == "108148"

    readme = (repo_root / "README.md").read_text()
    assert "local approximation of the Dexterity Foresight API Challenge" in readme
    assert "backend physics simulation" in readme
    assert "local-only" in readme
