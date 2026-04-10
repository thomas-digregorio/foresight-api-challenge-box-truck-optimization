from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from app.api.serializers import serialize_local_state
from app.engine.truck_packing_engine import TruckPackingEngine
from app.models.entities import PlacementAction
from app.services.episode_registry import EpisodeRegistry
from app.services.episode_service import EpisodeService

from app.agents.extreme_point.agent import GreedyExtremePointAgent


@dataclass(slots=True)
class LocalEpisodeResult:
    game_id: str
    density: float
    boxes_placed: int
    game_status: str
    termination_reason: str | None
    invalid_submission_count: int
    fallback_count: int
    move_count: int
    decision_latencies_ms: list[float]
    candidates_generated_per_move: list[int]
    candidates_validated_per_move: list[int]


def run_local_episode(
    *,
    agent: GreedyExtremePointAgent,
    seed: int | None = None,
    mode: str = "dev",
    engine: TruckPackingEngine | None = None,
) -> LocalEpisodeResult:
    engine = engine or agent.engine
    registry = EpisodeRegistry()
    service = EpisodeService(registry, engine)
    state = service.start_episode(
        mode=mode,
        seed=seed,
        api_key="local-agent",
        api_variant="local",
        enable_local_timeout=False,
        auto_terminate_on_no_feasible_placement=True,
    )

    invalid_submission_count = 0
    fallback_count = 0
    move_count = 0
    decision_latencies_ms: list[float] = []
    candidates_generated_per_move: list[int] = []
    candidates_validated_per_move: list[int] = []

    while state.game_status == "in_progress" and state.current_box is not None:
        raw_state = serialize_local_state(state, engine.config)
        started = perf_counter()
        action = agent.select_action(raw_state)
        decision_latencies_ms.append((perf_counter() - started) * 1000.0)
        explanation = agent.explain_last_choice()
        candidates_generated_per_move.append(int(explanation.get("candidates_generated", 0)))
        candidates_validated_per_move.append(int(explanation.get("candidates_validated", 0)))
        fallback_count += int(bool(explanation.get("fallback_used", False)))
        if action is None:
            state = service.stop_episode(state.game_id, expected_api_variant="local")
            break
        try:
            state = service.place_box(
                state.game_id,
                PlacementAction(
                    box_id=str(action["box_id"]),
                    position=tuple(float(value) for value in action["position"]),
                    orientation_wxyz=tuple(float(value) for value in action["orientation_wxyz"]),
                ),
                expected_api_variant="local",
            )
        except Exception:  # noqa: BLE001
            invalid_submission_count += 1
            state = service.stop_episode(state.game_id, expected_api_variant="local")
            break
        move_count += 1

    return LocalEpisodeResult(
        game_id=state.game_id,
        density=float(state.density),
        boxes_placed=len(state.placed_boxes),
        game_status=state.game_status,
        termination_reason=state.termination_reason,
        invalid_submission_count=invalid_submission_count,
        fallback_count=fallback_count,
        move_count=move_count,
        decision_latencies_ms=decision_latencies_ms,
        candidates_generated_per_move=candidates_generated_per_move,
        candidates_validated_per_move=candidates_validated_per_move,
    )
