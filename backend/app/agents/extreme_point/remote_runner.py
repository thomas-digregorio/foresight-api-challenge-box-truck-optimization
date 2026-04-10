from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from app.agents.extreme_point.agent import GreedyExtremePointAgent
from app.agents.extreme_point.http_client import ChallengeLikeHttpClient


@dataclass(slots=True)
class RemoteEpisodeResult:
    game_id: str
    density: float
    boxes_placed: int
    game_status: str
    termination_reason: str | None
    fallback_count: int
    move_count: int
    decision_latencies_ms: list[float]
    candidates_generated_per_move: list[int]
    candidates_validated_per_move: list[int]


def run_remote_episode(
    *,
    client: ChallengeLikeHttpClient,
    agent: GreedyExtremePointAgent,
    mode: str = "dev",
) -> RemoteEpisodeResult:
    state = client.start_game(mode=mode)
    game_id = str(state["game_id"])
    truck = state.get("truck")
    fallback_count = 0
    move_count = 0
    decision_latencies_ms: list[float] = []
    candidates_generated_per_move: list[int] = []
    candidates_validated_per_move: list[int] = []

    while state.get("game_status", "in_progress") == "in_progress" and state.get("current_box") is not None:
        if truck is not None and state.get("truck") is None:
            state = {**state, "truck": truck}
        started = perf_counter()
        action = agent.select_action(state)
        decision_latencies_ms.append((perf_counter() - started) * 1000.0)
        explanation = agent.explain_last_choice()
        candidates_generated_per_move.append(int(explanation.get("candidates_generated", 0)))
        candidates_validated_per_move.append(int(explanation.get("candidates_validated", 0)))
        fallback_count += int(bool(explanation.get("fallback_used", False)))
        if action is None:
            state = client.stop_game(game_id)
            if truck is not None and state.get("truck") is None:
                state["truck"] = truck
            break
        state = client.place_box(game_id, action)
        if truck is not None and state.get("truck") is None:
            state["truck"] = truck
        move_count += 1

    density = float(state.get("density", 0.0))
    placed_boxes = state.get("placed_boxes", [])
    return RemoteEpisodeResult(
        game_id=game_id,
        density=density,
        boxes_placed=len(placed_boxes),
        game_status=str(state.get("game_status", "completed")),
        termination_reason=state.get("termination_reason"),
        fallback_count=fallback_count,
        move_count=move_count,
        decision_latencies_ms=decision_latencies_ms,
        candidates_generated_per_move=candidates_generated_per_move,
        candidates_validated_per_move=candidates_validated_per_move,
    )
