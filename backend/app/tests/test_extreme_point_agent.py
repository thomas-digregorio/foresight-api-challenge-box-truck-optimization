from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import pytest

from app.agents.extreme_point.agent import GreedyExtremePointAgent
from app.agents.extreme_point.candidate_generation import generate_candidate_groups, generate_candidate_groups_limited
from app.agents.extreme_point.evaluator import evaluate_candidate_groups
from app.agents.extreme_point.local_runner import run_local_episode
from app.agents.extreme_point.orientations import get_orthogonal_orientation_options, get_orthogonal_orientations_wxyz
from app.agents.extreme_point.remote_runner import run_remote_episode
from app.agents.extreme_point.scoring import compute_score_breakdown
from app.agents.extreme_point.state_view import DecisionStateView
from app.agents.extreme_point.support_planes import extract_support_planes
from app.agents.extreme_point.types import EvaluationSummary, ProxyCandidate, ScoreWeights
from app.api.serializers import serialize_local_state
from app.engine.geometry import normalize_quaternion
from app.engine.truck_packing_engine import TruckPackingEngine
from app.models.entities import CurrentBox, EngineConfig, PlacedBox, ValidationResult
from app.tests.conftest import make_state


def build_raw_state(engine, state):
    return serialize_local_state(state, engine.config)


def test_orthogonal_orientation_enumeration_is_deterministic_and_unique() -> None:
    orientations_a = get_orthogonal_orientations_wxyz()
    orientations_b = get_orthogonal_orientations_wxyz()

    assert len(orientations_a) == 6
    assert orientations_a == orientations_b
    assert len({tuple(round(value, 6) for value in quaternion) for quaternion in orientations_a}) == 6
    for quaternion in orientations_a:
        assert normalize_quaternion(quaternion) == pytest.approx(quaternion)


def test_candidate_generation_on_empty_truck_is_front_left_and_deduplicated(engine) -> None:
    state = make_state(current_box=CurrentBox(id="box", dimensions=(0.4, 0.3, 0.2), weight=5.0))
    view = DecisionStateView.from_raw_state(build_raw_state(engine, state), config=engine.config)

    support_planes = extract_support_planes(view)
    orientations = get_orthogonal_orientation_options(
        view.current_box.dimensions,
        vertical_axis_cos_tolerance=engine.config.vertical_axis_cos_tolerance,
    )
    groups = generate_candidate_groups(view, support_planes, orientations)
    candidates = [candidate for group in groups for candidate in group.candidates]

    assert len(support_planes) == 1
    assert candidates
    assert len({candidate.dedup_key for candidate in candidates}) == len(candidates)
    assert any(candidate.position == pytest.approx((0.2, 0.15, 0.1)) for candidate in candidates)


def test_generate_candidate_groups_limited_respects_generation_cap(engine) -> None:
    state = make_state(current_box=CurrentBox(id="box", dimensions=(0.4, 0.3, 0.2), weight=5.0))
    view = DecisionStateView.from_raw_state(build_raw_state(engine, state), config=engine.config)
    support_planes = extract_support_planes(view)
    orientations = get_orthogonal_orientation_options(
        view.current_box.dimensions,
        vertical_axis_cos_tolerance=engine.config.vertical_axis_cos_tolerance,
    )

    groups, generated_count, _, _ = generate_candidate_groups_limited(
        view,
        support_planes,
        orientations,
        max_generated_candidates=12,
    )

    assert groups
    assert generated_count <= 12
    assert sum(len(group.candidates) for group in groups) == generated_count


def test_candidate_generation_above_supported_box_includes_top_surface(engine) -> None:
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
    view = DecisionStateView.from_raw_state(build_raw_state(engine, state), config=engine.config)

    support_planes = extract_support_planes(view)
    orientations = get_orthogonal_orientation_options(
        view.current_box.dimensions,
        vertical_axis_cos_tolerance=engine.config.vertical_axis_cos_tolerance,
    )
    groups = generate_candidate_groups(view, support_planes, orientations)
    candidates = [candidate for group in groups for candidate in group.candidates]

    assert any(plane.supporting_box_ids == ("base",) for plane in support_planes)
    assert any(candidate.position == pytest.approx((0.25, 0.5, 0.75)) for candidate in candidates)


def test_scoring_favors_smaller_delta_x(engine) -> None:
    state = make_state(current_box=CurrentBox(id="box", dimensions=(0.5, 0.5, 0.5), weight=5.0))
    view = DecisionStateView.from_raw_state(build_raw_state(engine, state), config=engine.config)
    orientation = get_orthogonal_orientation_options(
        view.current_box.dimensions,
        vertical_axis_cos_tolerance=engine.config.vertical_axis_cos_tolerance,
    )[0]
    near_front = generate_candidate_groups(view, extract_support_planes(view), [orientation])[0].candidates[0]
    farther_back = replace(near_front, position=(0.75, near_front.position[1], near_front.position[2]))

    near_score = compute_score_breakdown(view, near_front, orientation, support_ratio=1.0, weights=ScoreWeights())
    far_score = compute_score_breakdown(view, farther_back, orientation, support_ratio=1.0, weights=ScoreWeights())

    assert near_score.delta_x < far_score.delta_x
    assert near_score.total_score > far_score.total_score


def test_scoring_penalizes_front_gaps_without_increasing_delta_x(engine) -> None:
    state = make_state(
        current_box=CurrentBox(id="box", dimensions=(0.5, 0.5, 0.5), weight=5.0),
        placed_boxes=[
            PlacedBox(
                id="frontier-marker",
                dimensions=(0.2, 0.2, 0.2),
                weight=1.0,
                position=(1.7, 2.3, 0.1),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        ],
    )
    view = DecisionStateView.from_raw_state(build_raw_state(engine, state), config=engine.config)
    orientation = get_orthogonal_orientation_options(
        view.current_box.dimensions,
        vertical_axis_cos_tolerance=engine.config.vertical_axis_cos_tolerance,
    )[0]
    tight_candidate = generate_candidate_groups(view, extract_support_planes(view), [orientation])[0].candidates[0]
    gappy_candidate = replace(tight_candidate, position=(0.55, tight_candidate.position[1], tight_candidate.position[2]))

    tight_score = compute_score_breakdown(view, tight_candidate, orientation, support_ratio=1.0, weights=ScoreWeights())
    gappy_score = compute_score_breakdown(view, gappy_candidate, orientation, support_ratio=1.0, weights=ScoreWeights())

    assert tight_score.delta_x == pytest.approx(gappy_score.delta_x)
    assert tight_score.gap_penalty < gappy_score.gap_penalty
    assert tight_score.total_score > gappy_score.total_score


def test_scoring_penalizes_future_slivers_on_support_components(engine) -> None:
    state = make_state(
        current_box=CurrentBox(id="box", dimensions=(0.5, 0.5, 0.5), weight=5.0),
        placed_boxes=[
            PlacedBox(
                id="base",
                dimensions=(1.0, 1.0, 0.5),
                weight=5.0,
                position=(0.5, 0.5, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        ],
    )
    view = DecisionStateView.from_raw_state(build_raw_state(engine, state), config=engine.config)
    orientation = get_orthogonal_orientation_options(
        view.current_box.dimensions,
        vertical_axis_cos_tolerance=engine.config.vertical_axis_cos_tolerance,
    )[0]
    top_groups = [
        group
        for group in generate_candidate_groups(view, extract_support_planes(view), [orientation])
        if group.support_plane.z_support > 0.0
    ]
    assert top_groups
    flush_candidate = next(candidate for candidate in top_groups[0].candidates if candidate.position == pytest.approx((0.25, 0.25, 0.75)))
    centered_candidate = replace(flush_candidate, position=(0.25, 0.5, 0.75))

    flush_score = compute_score_breakdown(view, flush_candidate, orientation, support_ratio=1.0, weights=ScoreWeights())
    centered_score = compute_score_breakdown(view, centered_candidate, orientation, support_ratio=1.0, weights=ScoreWeights())

    assert centered_score.future_sliver_penalty > flush_score.future_sliver_penalty
    assert centered_score.total_score < flush_score.total_score


def test_serial_and_parallel_evaluation_choose_same_candidate(engine) -> None:
    state = make_state(
        current_box=CurrentBox(id="top", dimensions=(0.5, 0.5, 0.5), weight=5.0),
        placed_boxes=[
            PlacedBox(
                id="base-left",
                dimensions=(0.5, 0.5, 0.5),
                weight=5.0,
                position=(0.25, 0.5, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            PlacedBox(
                id="base-right",
                dimensions=(0.5, 0.5, 0.5),
                weight=5.0,
                position=(0.25, 1.1, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
        ],
    )
    view = DecisionStateView.from_raw_state(build_raw_state(engine, state), config=engine.config)
    orientations = get_orthogonal_orientation_options(
        view.current_box.dimensions,
        vertical_axis_cos_tolerance=engine.config.vertical_axis_cos_tolerance,
    )
    groups = generate_candidate_groups(view, extract_support_planes(view), orientations)

    serial = evaluate_candidate_groups(
        view,
        groups,
        engine=engine,
        weights=ScoreWeights(),
        parallel=False,
        max_workers=None,
        parallel_candidate_threshold=1,
    )
    parallel = evaluate_candidate_groups(
        view,
        groups,
        engine=engine,
        weights=ScoreWeights(),
        parallel=True,
        max_workers=2,
        parallel_candidate_threshold=1,
    )

    assert serial.ranked_candidates
    assert parallel.ranked_candidates
    assert serial.ranked_candidates[0].action.position == pytest.approx(parallel.ranked_candidates[0].action.position)
    assert serial.ranked_candidates[0].action.orientation_wxyz == pytest.approx(parallel.ranked_candidates[0].action.orientation_wxyz)
    assert [candidate.action.position for candidate in serial.ranked_candidates[:5]] == pytest.approx(
        [candidate.action.position for candidate in parallel.ranked_candidates[:5]]
    )


def test_pruning_reduces_validations_without_changing_best_candidate(engine) -> None:
    state = make_state(
        current_box=CurrentBox(id="top", dimensions=(0.5, 0.5, 0.5), weight=5.0),
        placed_boxes=[
            PlacedBox(
                id="base-left",
                dimensions=(0.5, 0.5, 0.5),
                weight=5.0,
                position=(0.25, 0.5, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            PlacedBox(
                id="base-right",
                dimensions=(0.5, 0.5, 0.5),
                weight=5.0,
                position=(0.25, 1.1, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
        ],
    )
    view = DecisionStateView.from_raw_state(build_raw_state(engine, state), config=engine.config)
    orientations = get_orthogonal_orientation_options(
        view.current_box.dimensions,
        vertical_axis_cos_tolerance=engine.config.vertical_axis_cos_tolerance,
    )
    groups = generate_candidate_groups(view, extract_support_planes(view), orientations)

    full = evaluate_candidate_groups(
        view,
        groups,
        engine=engine,
        weights=ScoreWeights(),
        parallel=False,
        max_workers=None,
        parallel_candidate_threshold=1,
        max_candidates_per_group=None,
        frontier_band_delta_x=None,
        enable_branch_and_bound=False,
        enable_dominance_pruning=False,
    )
    pruned = evaluate_candidate_groups(
        view,
        groups,
        engine=engine,
        weights=ScoreWeights(),
        parallel=False,
        max_workers=None,
        parallel_candidate_threshold=1,
        max_candidates_per_group=8,
        frontier_band_delta_x=0.2,
        enable_branch_and_bound=True,
    )

    assert full.ranked_candidates
    assert pruned.ranked_candidates
    assert pruned.validated_count < full.validated_count
    assert pruned.pruned_count > 0 or pruned.skipped_by_bound_count > 0
    assert pruned.ranked_candidates[0].score.total_score == pytest.approx(
        sorted(candidate.score.total_score for candidate in pruned.ranked_candidates)[-1]
    )


def test_evaluation_respects_total_validation_budget(engine) -> None:
    state = make_state(
        current_box=CurrentBox(id="top", dimensions=(0.5, 0.5, 0.5), weight=5.0),
        placed_boxes=[
            PlacedBox(
                id="base-left",
                dimensions=(0.5, 0.5, 0.5),
                weight=5.0,
                position=(0.25, 0.5, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            PlacedBox(
                id="base-right",
                dimensions=(0.5, 0.5, 0.5),
                weight=5.0,
                position=(0.25, 1.1, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
        ],
    )
    view = DecisionStateView.from_raw_state(build_raw_state(engine, state), config=engine.config)
    orientations = get_orthogonal_orientation_options(
        view.current_box.dimensions,
        vertical_axis_cos_tolerance=engine.config.vertical_axis_cos_tolerance,
    )
    groups = generate_candidate_groups(view, extract_support_planes(view), orientations)

    summary = evaluate_candidate_groups(
        view,
        groups,
        engine=engine,
        weights=ScoreWeights(),
        parallel=False,
        max_workers=None,
        parallel_candidate_threshold=1,
        max_candidates_per_group=None,
        frontier_band_delta_x=None,
        enable_branch_and_bound=False,
        enable_dominance_pruning=False,
        max_total_validations=5,
    )

    assert summary.validated_count == 5
    assert summary.validation_budget_hit is True


def test_evaluation_respects_deadline_during_preparation(engine) -> None:
    state = make_state(
        current_box=CurrentBox(id="top", dimensions=(0.5, 0.5, 0.5), weight=5.0),
        placed_boxes=[
            PlacedBox(
                id="base-left",
                dimensions=(0.5, 0.5, 0.5),
                weight=5.0,
                position=(0.25, 0.5, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            PlacedBox(
                id="base-right",
                dimensions=(0.5, 0.5, 0.5),
                weight=5.0,
                position=(0.25, 1.1, 0.25),
                orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
        ],
    )
    view = DecisionStateView.from_raw_state(build_raw_state(engine, state), config=engine.config)
    orientations = get_orthogonal_orientation_options(
        view.current_box.dimensions,
        vertical_axis_cos_tolerance=engine.config.vertical_axis_cos_tolerance,
    )
    groups = generate_candidate_groups(view, extract_support_planes(view), orientations)

    summary = evaluate_candidate_groups(
        view,
        groups,
        engine=engine,
        weights=ScoreWeights(),
        parallel=False,
        max_workers=None,
        parallel_candidate_threshold=1,
        deadline_monotonic=perf_counter() - 1.0,
    )

    assert summary.deadline_hit is True
    assert summary.validated_count == 0


def test_fallback_returns_engine_action_when_extreme_points_are_empty(engine, monkeypatch) -> None:
    state = make_state(current_box=CurrentBox(id="box", dimensions=(0.5, 0.5, 0.5), weight=5.0))
    raw_state = build_raw_state(engine, state)
    agent = GreedyExtremePointAgent(engine=engine, parallel=False)

    monkeypatch.setattr(
        "app.agents.extreme_point.agent.generate_candidate_groups_limited",
        lambda *args, **kwargs: ([], 0, kwargs.get("start_group_index", 0), kwargs.get("seen_dedup_keys", set())),
    )

    action = agent.select_action(raw_state)

    assert action is not None
    explanation = agent.explain_last_choice()
    assert explanation["fallback_used"] is True


def test_budgeted_repair_fallback_avoids_exhaustive_scan(engine, monkeypatch) -> None:
    state = make_state(current_box=CurrentBox(id="box", dimensions=(0.5, 0.5, 0.5), weight=5.0))
    raw_state = build_raw_state(engine, state)
    view = DecisionStateView.from_raw_state(raw_state, config=engine.config)
    orientations = get_orthogonal_orientation_options(
        view.current_box.dimensions,
        vertical_axis_cos_tolerance=engine.config.vertical_axis_cos_tolerance,
    )
    groups = generate_candidate_groups(view, extract_support_planes(view), orientations)
    proxy_candidate = groups[0].candidates[0]
    orientation = next(option for option in orientations if option.index == proxy_candidate.orientation_index)
    proxy_score = compute_score_breakdown(view, proxy_candidate, orientation, support_ratio=1.0, weights=ScoreWeights())
    repaired_action = proxy_candidate.as_action("box")

    monkeypatch.setattr(
        "app.agents.extreme_point.agent.evaluate_candidate_groups",
        lambda *args, **kwargs: EvaluationSummary(
            ranked_candidates=[],
            generated_count=1,
            validated_count=0,
            valid_count=0,
            parallel_used=False,
            group_count=1,
            best_proxy_candidate=proxy_candidate,
            best_proxy_score=proxy_score,
        ),
    )

    def validate_place_action(_state, action):
        if tuple(round(value, 6) for value in action.position) == tuple(round(value, 6) for value in repaired_action.position):
            return ValidationResult(
                is_valid=True,
                message="ok",
                normalized_action=action,
                support_ratio=1.0,
            )
        return ValidationResult(
            is_valid=False,
            message="invalid",
            normalized_action=None,
            support_ratio=0.0,
        )

    monkeypatch.setattr(engine, "validate_place_action", validate_place_action)
    monkeypatch.setattr(engine, "find_valid_action_at_current_xy", lambda _state, _action: repaired_action)
    monkeypatch.setattr(
        engine,
        "find_valid_action_near",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("near fallback should not run")),
    )
    monkeypatch.setattr(
        engine,
        "find_any_valid_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("exhaustive fallback should not run")),
    )

    agent = GreedyExtremePointAgent(engine=engine, parallel=False)
    action = agent.select_action(raw_state)

    assert action is not None
    assert action["position"] == pytest.approx(repaired_action.position)
    explanation = agent.explain_last_choice()
    assert explanation["fallback_used"] is True


def test_budgeted_fallback_ranks_multiple_repaired_proxy_candidates(engine, monkeypatch) -> None:
    state = make_state(current_box=CurrentBox(id="box", dimensions=(0.5, 0.5, 0.5), weight=5.0))
    raw_state = build_raw_state(engine, state)
    view = DecisionStateView.from_raw_state(raw_state, config=engine.config)
    orientation = get_orthogonal_orientation_options(
        view.current_box.dimensions,
        vertical_axis_cos_tolerance=engine.config.vertical_axis_cos_tolerance,
    )[0]
    groups = generate_candidate_groups(view, extract_support_planes(view), [orientation])
    proxy_a = groups[0].candidates[0]
    proxy_b = replace(proxy_a, position=(0.75, proxy_a.position[1], proxy_a.position[2]))
    proxy_a_score = compute_score_breakdown(view, proxy_a, orientation, support_ratio=1.0, weights=ScoreWeights())
    proxy_b_score = compute_score_breakdown(view, proxy_b, orientation, support_ratio=1.0, weights=ScoreWeights())

    repaired_a = proxy_b.as_action("box")
    repaired_b = proxy_a.as_action("box")

    monkeypatch.setattr(
        "app.agents.extreme_point.agent.evaluate_candidate_groups",
        lambda *args, **kwargs: EvaluationSummary(
            ranked_candidates=[],
            generated_count=2,
            validated_count=0,
            valid_count=0,
            parallel_used=False,
            group_count=1,
            best_proxy_candidate=proxy_a,
            best_proxy_score=proxy_a_score,
            top_proxy_candidates=[
                ProxyCandidate(candidate=proxy_a, score=proxy_a_score),
                ProxyCandidate(candidate=proxy_b, score=proxy_b_score),
            ],
        ),
    )

    def validate_place_action(_state, action):
        rounded_position = tuple(round(value, 6) for value in action.position)
        if rounded_position in {
            tuple(round(value, 6) for value in repaired_a.position),
            tuple(round(value, 6) for value in repaired_b.position),
        }:
            return ValidationResult(
                is_valid=True,
                message="ok",
                normalized_action=action,
                support_ratio=1.0,
            )
        return ValidationResult(
            is_valid=False,
            message="invalid",
            normalized_action=None,
            support_ratio=0.0,
        )

    monkeypatch.setattr(engine, "validate_place_action", validate_place_action)
    monkeypatch.setattr(
        engine,
        "find_valid_action_at_current_xy",
        lambda _state, action: repaired_a if tuple(round(value, 6) for value in action.position) == tuple(round(value, 6) for value in proxy_a.position) else repaired_b,
    )
    monkeypatch.setattr(
        engine,
        "find_valid_action_near",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("near fallback should not run when aligned repairs succeed")),
    )
    monkeypatch.setattr(
        engine,
        "find_any_valid_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("exhaustive fallback should not run when repairs succeed")),
    )

    agent = GreedyExtremePointAgent(engine=engine, parallel=False)
    action = agent.select_action(raw_state)

    assert action is not None
    assert action["position"] == pytest.approx(repaired_b.position)
    explanation = agent.explain_last_choice()
    assert explanation["fallback_used"] is True


def test_budgeted_fallback_skips_exhaustive_scan_when_deadline_is_gone(engine, monkeypatch) -> None:
    state = make_state(current_box=CurrentBox(id="box", dimensions=(0.5, 0.5, 0.5), weight=5.0))
    raw_state = build_raw_state(engine, state)
    agent = GreedyExtremePointAgent(
        engine=engine,
        parallel=False,
        decision_time_budget_seconds=0.0,
    )

    monkeypatch.setattr(
        engine,
        "find_any_valid_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("exhaustive fallback should not run")),
    )

    action = agent.select_action(raw_state)

    assert action is None
    explanation = agent.explain_last_choice()
    assert explanation["deadline_hit"] is True


def test_explain_last_choice_contains_required_fields(engine) -> None:
    state = make_state(current_box=CurrentBox(id="box", dimensions=(0.5, 0.5, 0.5), weight=5.0))
    raw_state = build_raw_state(engine, state)
    agent = GreedyExtremePointAgent(engine=engine, parallel=False)

    action = agent.select_action(raw_state)

    assert action is not None
    explanation = agent.explain_last_choice()
    assert explanation["chosen_action"]["box_id"] == "box"
    for key in (
        "total_score",
        "delta_x",
        "front_gap",
        "frontier_slack",
        "future_sliver_penalty",
        "left_gap",
        "right_gap",
        "support_reward",
        "contact_reward",
        "frontier_reach_reward",
        "fallback_used",
        "candidates_generated",
        "candidates_validated",
        "candidates_pruned",
        "candidates_skipped_by_bound",
        "support_plane_extraction_ms",
        "candidate_generation_ms",
        "evaluation_preparation_ms",
        "evaluation_validation_ms",
        "fallback_ms",
        "decision_time_ms",
    ):
        assert key in explanation


def test_end_to_end_local_runner_completes_multiple_episodes() -> None:
    densities: list[float] = []
    for seed in (1, 2):
        engine = TruckPackingEngine(config=EngineConfig(default_queue_length=4))
        agent = GreedyExtremePointAgent(engine=engine, parallel=True, max_workers=2)
        result = run_local_episode(agent=agent, seed=seed, engine=engine)
        densities.append(result.density)
        assert result.move_count > 0
        assert result.invalid_submission_count == 0
        assert result.boxes_placed > 0
    assert all(density >= 0.0 for density in densities)


def test_remote_runner_preserves_game_id_and_submits_it_with_actions() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.actions: list[dict[str, object]] = []

        def start_game(self, *, mode: str) -> dict[str, object]:
            assert mode == "dev"
            return {
                "game_id": "game-123",
                "truck": {"depth": 2.0, "width": 2.6, "height": 2.75},
                "placed_boxes": [],
                "current_box": {"id": "box-1", "dimensions": (0.5, 0.5, 0.5), "weight": 5.0},
                "boxes_remaining": 0,
                "game_status": "in_progress",
            }

        def place_box(self, game_id: str, action: dict[str, object]) -> dict[str, object]:
            assert game_id == "game-123"
            self.actions.append({"game_id": game_id, **action})
            return {
                "status": "terminated",
                "placed_boxes": [
                    {
                        "id": "box-1",
                        "dimensions": (0.5, 0.5, 0.5),
                        "position": (0.25, 0.25, 0.25),
                        "orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
                    }
                ],
                "current_box": None,
                "boxes_remaining": 0,
                "density": 0.1,
                "game_status": "completed",
                "termination_reason": None,
            }

        def stop_game(self, game_id: str) -> dict[str, object]:
            raise AssertionError(f"stop_game should not be called for {game_id}")

    class FakeAgent:
        def __init__(self) -> None:
            self._last_choice = {
                "fallback_used": False,
                "candidates_generated": 4,
                "candidates_validated": 2,
            }

        def select_action(self, raw_state: dict[str, object]) -> dict[str, object] | None:
            assert raw_state["truck"] == {"depth": 2.0, "width": 2.6, "height": 2.75}
            return {
                "box_id": "box-1",
                "position": [0.25, 0.25, 0.25],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            }

        def explain_last_choice(self) -> dict[str, object]:
            return dict(self._last_choice)

    client = FakeClient()
    result = run_remote_episode(client=client, agent=FakeAgent(), mode="dev")

    assert result.game_id == "game-123"
    assert client.actions == [
        {
            "game_id": "game-123",
            "box_id": "box-1",
            "position": [0.25, 0.25, 0.25],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
    ]
    assert result.candidates_generated_per_move == [4]
    assert result.candidates_validated_per_move == [2]
