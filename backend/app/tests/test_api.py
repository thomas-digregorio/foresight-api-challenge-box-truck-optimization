from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.models.entities import CurrentBox, PlacedBox


def test_invalid_place_action_returns_structured_422() -> None:
    app = create_app()
    client = TestClient(app)

    start_response = client.post(
        "/challenge/api/start",
        json={"api_key": "local", "mode": "dev", "seed": 7},
    )
    assert start_response.status_code == 200
    game_id = start_response.json()["game_id"]

    state = app.state.services.registry.get(game_id)
    state.current_box = CurrentBox(id="manual-box", dimensions=(1.0, 1.0, 1.0), weight=5.0)
    state.remaining_boxes = []
    app.state.services.registry.update(state)

    response = client.post(
        "/challenge/api/place",
        json={
            "game_id": game_id,
            "box_id": "manual-box",
            "position": [0.5, 0.5, 0.5],
            "orientation_wxyz": [0.0, 0.0, 0.0, 0.0],
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "validation_error"
    assert payload["details"]["category"] == "invalid_quaternion"


def test_preview_returns_repair_suggestion_for_overlapping_default_pose() -> None:
    app = create_app()
    client = TestClient(app)

    start_response = client.post(
        "/challenge/api/start",
        json={"api_key": "local", "mode": "dev", "seed": 9},
    )
    game_id = start_response.json()["game_id"]

    state = app.state.services.registry.get(game_id)
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

    response = client.post(
        "/local/api/preview",
        json={
            "game_id": game_id,
            "box_id": "top-box",
            "position": [0.31, 1.3, 0.25],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["is_valid"] is False
    assert payload["category"] == "overlap"
    assert payload["repair_suggestions"]["support_aligned"] is not None
    assert payload["repair_suggestions"]["any_valid"] is not None
