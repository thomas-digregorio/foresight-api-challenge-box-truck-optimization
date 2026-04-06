from __future__ import annotations

import pytest

from app.models.entities import CurrentBox, PlacementAction, PlacedBox, PreviewAction
from app.tests.conftest import make_state


def test_support_overlap_rule_uses_ninety_percent_threshold(engine) -> None:
    state = make_state(
        current_box=CurrentBox(id="top", dimensions=(0.5, 0.5, 0.5), weight=10.0),
        placed_boxes=[
            PlacedBox(
                id="base",
                dimensions=(0.5, 0.5, 0.5),
                weight=12.0,
                position=(0.25, 0.5, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        ],
    )
    valid = engine.validate_place_action(
        state,
        PlacementAction(
            box_id="top",
            position=(0.295, 0.5, 0.75),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    invalid = engine.validate_place_action(
        state,
        PlacementAction(
            box_id="top",
            position=(0.305, 0.5, 0.75),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    assert valid.is_valid is True
    assert invalid.is_valid is False
    assert invalid.category == "insufficient_support"
    assert invalid.support_ratio == pytest.approx(0.89)


def test_density_calculation_uses_max_x_reached(engine) -> None:
    state = make_state(
        current_box=None,
        placed_boxes=[
            PlacedBox(
                id="placed",
                dimensions=(1.0, 1.0, 1.0),
                weight=5.0,
                position=(0.5, 0.5, 0.5),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        ],
    )
    density = engine.compute_density(state)
    assert density == pytest.approx(1.0 / (1.0 * 2.6 * 2.75))


def test_preview_updates_keep_latest_valid_preview(engine) -> None:
    state = make_state(current_box=CurrentBox(id="active", dimensions=(0.5, 0.5, 0.5), weight=5.0))
    valid_preview = PreviewAction(
        box_id="active",
        position=(0.25, 0.5, 0.25),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    invalid_preview = PreviewAction(
        box_id="active",
        position=(-0.2, 0.5, 0.25),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    valid_result = engine.update_preview(state, valid_preview)
    invalid_result = engine.update_preview(state, invalid_preview)

    assert valid_result.is_valid is True
    assert invalid_result.is_valid is False
    assert state.latest_preview_action.position == pytest.approx((-0.2, 0.5, 0.25))
    assert state.latest_valid_preview_action is not None
    assert state.latest_valid_preview_action.position == pytest.approx((0.25, 0.5, 0.25))


def test_timeout_auto_places_latest_valid_preview(engine, clock) -> None:
    state = make_state(current_box=CurrentBox(id="active", dimensions=(0.5, 0.5, 0.5), weight=5.0), now=clock())
    preview = PreviewAction(
        box_id="active",
        position=(0.25, 0.5, 0.25),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
    )
    engine.update_preview(state, preview)
    clock.advance(11)

    timed_out = engine.handle_timeout_if_needed(state)

    assert timed_out is True
    assert len(state.placed_boxes) == 1
    assert state.placed_boxes[0].source == "timeout_auto_place"
    assert state.game_status == "completed"


def test_action_repair_recovers_nearby_valid_action(engine) -> None:
    state = make_state(
        current_box=CurrentBox(id="top", dimensions=(0.5, 0.5, 0.5), weight=5.0),
        placed_boxes=[
            PlacedBox(
                id="base",
                dimensions=(0.5, 0.5, 0.5),
                weight=5.0,
                position=(0.25, 0.5, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        ],
    )
    repaired = engine.find_valid_action_near(
        state,
        PlacementAction(
            box_id="top",
            position=(0.38, 0.5, 0.75),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    assert repaired is not None
    assert engine.validate_place_action(state, repaired).is_valid is True


def test_support_projection_prefers_box_tops_over_floor_at_same_xy(engine) -> None:
    state = make_state(
        current_box=CurrentBox(id="active", dimensions=(0.5, 0.5, 0.5), weight=5.0),
        placed_boxes=[
            PlacedBox(
                id="base",
                dimensions=(0.5, 0.5, 0.5),
                weight=5.0,
                position=(0.25, 0.6, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            PlacedBox(
                id="tall",
                dimensions=(0.5, 0.5, 0.7),
                weight=6.0,
                position=(0.25, 1.4, 0.35),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
        ],
    )

    left_support = engine.find_valid_action_at_current_xy(
        state,
        PlacementAction(
            box_id="active",
            position=(0.25, 0.6, 0.25),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    right_support = engine.find_valid_action_at_current_xy(
        state,
        PlacementAction(
            box_id="active",
            position=(0.25, 1.4, 0.25),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    assert left_support is not None
    assert left_support.position == pytest.approx((0.25, 0.6, 0.75))
    assert right_support is not None
    assert right_support.position == pytest.approx((0.25, 1.4, 0.95))


def test_loading_line_rejects_boxes_that_cross_frontier(engine) -> None:
    state = make_state(current_box=CurrentBox(id="active", dimensions=(0.5, 0.5, 0.5), weight=5.0))
    valid = engine.validate_place_action(
        state,
        PlacementAction(
            box_id="active",
            position=(0.52, 1.3, 0.25),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    invalid = engine.validate_place_action(
        state,
        PlacementAction(
            box_id="active",
            position=(0.56, 1.3, 0.25),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    assert valid.is_valid is True
    assert invalid.is_valid is False
    assert invalid.category == "loading_line_crossed"
    assert invalid.details["loading_line_x"] == pytest.approx(engine.config.loading_line_x)
