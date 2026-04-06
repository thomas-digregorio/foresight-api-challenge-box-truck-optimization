from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.engine.truck_packing_engine import TruckPackingEngine
from app.models.entities import CurrentBox, EngineConfig, EpisodeTimerState, GameState, PlacedBox, Truck


class MutableClock:
    def __init__(self, now: datetime | None = None) -> None:
        self.now = now or datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock() -> MutableClock:
    return MutableClock()


@pytest.fixture
def engine(clock: MutableClock) -> TruckPackingEngine:
    return TruckPackingEngine(config=EngineConfig(default_queue_length=4), clock=clock)


def make_state(
    *,
    current_box: CurrentBox | None = None,
    placed_boxes: list[PlacedBox] | None = None,
    remaining_boxes: list[CurrentBox] | None = None,
    now: datetime | None = None,
) -> GameState:
    current_time = now or datetime(2026, 1, 1, tzinfo=UTC)
    timer = None
    if current_box is not None:
        timer = EpisodeTimerState(
            current_box_started_at=current_time,
            current_box_deadline=current_time + timedelta(seconds=10),
            timeout_seconds=10.0,
        )
    return GameState(
        game_id="test-game",
        mode="dev",
        truck=Truck(),
        placed_boxes=placed_boxes or [],
        remaining_boxes=remaining_boxes or [],
        current_box=current_box,
        density=0.0,
        game_status="in_progress",
        created_at=current_time,
        timer_state=timer,
        latest_preview_action=None,
        latest_valid_preview_action=None,
    )

