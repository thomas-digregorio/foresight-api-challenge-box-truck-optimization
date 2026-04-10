from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any

from app.engine.truck_packing_engine import TruckPackingEngine
from app.models.entities import PlacementAction, Truck

from app.agents.extreme_point.candidate_generation import generate_candidate_groups_limited
from app.agents.extreme_point.evaluator import evaluate_candidate_groups
from app.agents.extreme_point.orientations import (
    build_orientation_option_for_quaternion,
    canonicalize_quaternion_sign,
    get_orthogonal_orientation_options,
)
from app.agents.extreme_point.scoring import compute_score_breakdown
from app.agents.extreme_point.state_view import DecisionStateView
from app.agents.extreme_point.support_planes import extract_support_planes
from app.agents.extreme_point.types import CandidatePlacement, EvaluationSummary, ProxyCandidate, RankedCandidate, ScoreWeights, SupportPlane


class GreedyExtremePointAgent:
    def __init__(
        self,
        *,
        engine: TruckPackingEngine | None = None,
        max_workers: int | None = None,
        parallel: bool = True,
        parallel_candidate_threshold: int = 64,
        fallback_to_engine: bool = True,
        score_weights: ScoreWeights | None = None,
        max_candidates_per_group: int | None = 12,
        frontier_band_delta_x: float | None = 0.15,
        enable_branch_and_bound: bool = True,
        enable_dominance_pruning: bool = True,
        decision_time_budget_seconds: float = 8.0,
        incumbent_return_margin_seconds: float = 1.25,
        fallback_buffer_seconds: float = 2.0,
        max_generated_candidates_per_move: int = 15000,
        max_validated_candidates_per_move: int = 600,
    ) -> None:
        self.engine = engine or TruckPackingEngine()
        self.max_workers = max_workers
        self.parallel = parallel
        self.parallel_candidate_threshold = parallel_candidate_threshold
        self.fallback_to_engine = fallback_to_engine
        self.score_weights = score_weights or ScoreWeights()
        self.max_candidates_per_group = max_candidates_per_group
        self.frontier_band_delta_x = frontier_band_delta_x
        self.enable_branch_and_bound = enable_branch_and_bound
        self.enable_dominance_pruning = enable_dominance_pruning
        self.decision_time_budget_seconds = decision_time_budget_seconds
        self.incumbent_return_margin_seconds = incumbent_return_margin_seconds
        self.fallback_buffer_seconds = fallback_buffer_seconds
        self.max_generated_candidates_per_move = max_generated_candidates_per_move
        self.max_validated_candidates_per_move = max_validated_candidates_per_move
        self._resolved_max_workers = max(1, max_workers or min(8, os.cpu_count() or 1))
        self._executor: ThreadPoolExecutor | None = None
        self._last_choice: dict[str, Any] = {}
        self._last_truck = Truck(
            depth=self.engine.config.truck.depth,
            width=self.engine.config.truck.width,
            height=self.engine.config.truck.height,
        )

    def select_action(self, raw_state: dict[str, Any]) -> dict[str, Any] | None:
        if raw_state.get("truck") is not None:
            truck_payload = raw_state["truck"]
            self._last_truck = Truck(
                depth=float(truck_payload["depth"]),
                width=float(truck_payload["width"]),
                height=float(truck_payload["height"]),
            )
        view = DecisionStateView.from_raw_state(
            raw_state,
            config=self.engine.config,
            fallback_truck=self._last_truck,
        )
        if view.current_box is None or view.game_state.game_status != "in_progress":
            self._last_choice = {
                "chosen_action": None,
                "fallback_used": False,
                "reason": "No current box is available.",
                "candidates_generated": 0,
                "candidates_validated": 0,
                "candidates_pruned": 0,
                "candidates_skipped_by_bound": 0,
                "support_plane_extraction_ms": 0.0,
                "candidate_generation_ms": 0.0,
                "evaluation_preparation_ms": 0.0,
                "evaluation_validation_ms": 0.0,
                "fallback_ms": 0.0,
                "decision_time_ms": 0.0,
                "invalid_reason_counts": {},
                "valid_counts_by_group": {},
            }
            return None

        started_at = perf_counter()
        deadline_monotonic = started_at + self.decision_time_budget_seconds
        orientations = get_orthogonal_orientation_options(
            view.current_box.dimensions,
            vertical_axis_cos_tolerance=self.engine.config.vertical_axis_cos_tolerance,
        )
        support_plane_started = perf_counter()
        support_plane_batches = self._support_plane_batches(extract_support_planes(view))
        support_plane_extraction_ms = (perf_counter() - support_plane_started) * 1000.0
        incumbent: RankedCandidate | None = None
        seen_dedup_keys: set[tuple[float, ...]] = set()
        next_group_index = 0
        aggregate_generated = 0
        aggregate_validated = 0
        aggregate_valid_count = 0
        aggregate_group_count = 0
        aggregate_pruned = 0
        aggregate_skipped = 0
        parallel_used = False
        deadline_hit = False
        validation_budget_hit = False
        best_proxy_candidate: CandidatePlacement | None = None
        best_proxy_score = None
        top_proxy_candidates: list[ProxyCandidate] = []
        candidate_generation_ms = 0.0
        evaluation_preparation_ms = 0.0
        evaluation_validation_ms = 0.0
        hard_state_mode = False
        aggregate_invalid_reason_counts: dict[str, int] = {}
        aggregate_valid_counts_by_group: dict[str, int] = {}

        for phase_index, support_plane_batch in enumerate(support_plane_batches):
            remaining_seconds = deadline_monotonic - perf_counter()
            if remaining_seconds <= 0.0:
                deadline_hit = True
                break
            if incumbent is not None and remaining_seconds <= self.incumbent_return_margin_seconds:
                break
            if incumbent is None and phase_index >= 1 and remaining_seconds <= self.fallback_buffer_seconds:
                break
            if aggregate_generated >= self.max_generated_candidates_per_move:
                break
            if aggregate_validated >= self.max_validated_candidates_per_move:
                validation_budget_hit = True
                break

            adaptive_max_candidates, adaptive_frontier_band = self._adaptive_search_limits(
                remaining_seconds,
                hard_state_mode=hard_state_mode,
            )
            generated_budget_remaining = self.max_generated_candidates_per_move - aggregate_generated
            generation_started = perf_counter()
            groups, generated_count, next_group_index, seen_dedup_keys = generate_candidate_groups_limited(
                view,
                support_plane_batch,
                orientations,
                start_group_index=next_group_index,
                seen_dedup_keys=seen_dedup_keys,
                max_generated_candidates=generated_budget_remaining,
                deadline_monotonic=deadline_monotonic,
            )
            candidate_generation_ms += (perf_counter() - generation_started) * 1000.0
            aggregate_generated += generated_count
            if not groups:
                if perf_counter() >= deadline_monotonic:
                    deadline_hit = True
                    break
                continue

            summary = evaluate_candidate_groups(
                view,
                groups,
                engine=self.engine,
                weights=self.score_weights,
                parallel=self.parallel,
                max_workers=self._resolved_max_workers,
                parallel_candidate_threshold=self.parallel_candidate_threshold,
                max_candidates_per_group=adaptive_max_candidates,
                frontier_band_delta_x=adaptive_frontier_band,
                enable_branch_and_bound=self.enable_branch_and_bound,
                enable_dominance_pruning=self.enable_dominance_pruning,
                deadline_monotonic=deadline_monotonic,
                max_total_validations=self.max_validated_candidates_per_move - aggregate_validated,
                incumbent_score=(-float("inf") if incumbent is None else incumbent.score.total_score),
                executor=self._ensure_executor(),
            )
            aggregate_validated += summary.validated_count
            aggregate_valid_count += summary.valid_count
            aggregate_group_count += summary.group_count
            aggregate_pruned += summary.pruned_count
            aggregate_skipped += summary.skipped_by_bound_count
            evaluation_preparation_ms += summary.preparation_time_ms
            evaluation_validation_ms += summary.validation_time_ms
            for reason, count in summary.invalid_reason_counts.items():
                aggregate_invalid_reason_counts[reason] = aggregate_invalid_reason_counts.get(reason, 0) + count
            for group_key, count in summary.valid_counts_by_group.items():
                aggregate_valid_counts_by_group[group_key] = aggregate_valid_counts_by_group.get(group_key, 0) + count
            parallel_used = parallel_used or summary.parallel_used
            deadline_hit = deadline_hit or summary.deadline_hit
            validation_budget_hit = validation_budget_hit or summary.validation_budget_hit
            top_proxy_candidates = self._merge_proxy_candidates(top_proxy_candidates, summary.top_proxy_candidates)
            if (
                not summary.top_proxy_candidates
                and summary.best_proxy_candidate is not None
                and summary.best_proxy_score is not None
            ):
                top_proxy_candidates = self._merge_proxy_candidates(
                    top_proxy_candidates,
                    [ProxyCandidate(candidate=summary.best_proxy_candidate, score=summary.best_proxy_score)],
                )
            if summary.best_proxy_candidate is not None and summary.best_proxy_score is not None:
                if best_proxy_candidate is None or self._proxy_ranking_key(
                    summary.best_proxy_candidate,
                    summary.best_proxy_score,
                ) < self._proxy_ranking_key(best_proxy_candidate, best_proxy_score):
                    best_proxy_candidate = summary.best_proxy_candidate
                    best_proxy_score = summary.best_proxy_score

            if summary.ranked_candidates:
                phase_best = summary.ranked_candidates[0]
                if incumbent is None or phase_best.ranking_key() < incumbent.ranking_key():
                    incumbent = phase_best
            hard_state_mode = hard_state_mode or summary.valid_count <= 1

            remaining_seconds = deadline_monotonic - perf_counter()
            if deadline_hit or validation_budget_hit:
                break
            if incumbent is not None and remaining_seconds <= self.incumbent_return_margin_seconds:
                break
            if incumbent is not None and phase_index >= 1 and remaining_seconds <= self.fallback_buffer_seconds:
                break

        summary = EvaluationSummary(
            ranked_candidates=[] if incumbent is None else [incumbent],
            generated_count=aggregate_generated,
            validated_count=aggregate_validated,
            valid_count=aggregate_valid_count,
            parallel_used=parallel_used,
            group_count=aggregate_group_count,
            pruned_count=aggregate_pruned,
            skipped_by_bound_count=aggregate_skipped,
            deadline_hit=deadline_hit,
            validation_budget_hit=validation_budget_hit,
            best_proxy_candidate=best_proxy_candidate,
            best_proxy_score=best_proxy_score,
            top_proxy_candidates=top_proxy_candidates,
            preparation_time_ms=evaluation_preparation_ms,
            validation_time_ms=evaluation_validation_ms,
            invalid_reason_counts=aggregate_invalid_reason_counts,
            valid_counts_by_group=aggregate_valid_counts_by_group,
        )
        if summary.best_proxy_candidate is None and summary.top_proxy_candidates:
            summary.best_proxy_candidate = summary.top_proxy_candidates[0].candidate
            summary.best_proxy_score = summary.top_proxy_candidates[0].score
        if incumbent is not None:
            self._remember_choice(
                incumbent,
                summary=summary,
                fallback_used=False,
                support_plane_extraction_ms=support_plane_extraction_ms,
                candidate_generation_ms=candidate_generation_ms,
                fallback_ms=0.0,
                decision_time_ms=(perf_counter() - started_at) * 1000.0,
            )
            return {
                "box_id": incumbent.action.box_id,
                "position": list(incumbent.action.position),
                "orientation_wxyz": list(incumbent.action.orientation_wxyz),
            }

        if not self.fallback_to_engine:
            self._last_choice = {
                "chosen_action": None,
                "fallback_used": False,
                "reason": "No valid extreme-point candidate found.",
                "candidates_generated": summary.generated_count,
                "candidates_validated": summary.validated_count,
                "candidates_pruned": summary.pruned_count,
                "candidates_skipped_by_bound": summary.skipped_by_bound_count,
                "deadline_hit": summary.deadline_hit,
                "validation_budget_hit": summary.validation_budget_hit,
                "support_plane_extraction_ms": support_plane_extraction_ms,
                "candidate_generation_ms": candidate_generation_ms,
                "evaluation_preparation_ms": evaluation_preparation_ms,
                "evaluation_validation_ms": evaluation_validation_ms,
                "fallback_ms": 0.0,
                "decision_time_ms": (perf_counter() - started_at) * 1000.0,
                "invalid_reason_counts": dict(summary.invalid_reason_counts),
                "valid_counts_by_group": dict(summary.valid_counts_by_group),
            }
            return None

        fallback_started = perf_counter()
        fallback_ranked = self._select_budgeted_fallback(
            view,
            top_proxy_candidates=top_proxy_candidates,
            deadline_monotonic=deadline_monotonic,
        )
        fallback_ms = (perf_counter() - fallback_started) * 1000.0
        if fallback_ranked is None:
            remaining_seconds = self._remaining_seconds(deadline_monotonic)
            deadline_expired = remaining_seconds <= 0.0
            self._last_choice = {
                "chosen_action": None,
                "fallback_used": False,
                "reason": (
                    "No exact incumbent found before the move deadline expired."
                    if deadline_expired
                    else "No valid candidate or budget-safe fallback action found."
                ),
                "candidates_generated": summary.generated_count,
                "candidates_validated": summary.validated_count,
                "candidates_pruned": summary.pruned_count,
                "candidates_skipped_by_bound": summary.skipped_by_bound_count,
                "deadline_hit": summary.deadline_hit or deadline_expired,
                "validation_budget_hit": summary.validation_budget_hit,
                "support_plane_extraction_ms": support_plane_extraction_ms,
                "candidate_generation_ms": candidate_generation_ms,
                "evaluation_preparation_ms": evaluation_preparation_ms,
                "evaluation_validation_ms": evaluation_validation_ms,
                "fallback_ms": fallback_ms,
                "decision_time_ms": (perf_counter() - started_at) * 1000.0,
                "invalid_reason_counts": dict(summary.invalid_reason_counts),
                "valid_counts_by_group": dict(summary.valid_counts_by_group),
            }
            return None
        self._remember_choice(
            fallback_ranked,
            summary=summary,
            fallback_used=True,
            support_plane_extraction_ms=support_plane_extraction_ms,
            candidate_generation_ms=candidate_generation_ms,
            fallback_ms=fallback_ms,
            decision_time_ms=(perf_counter() - started_at) * 1000.0,
        )
        return {
            "box_id": fallback_ranked.action.box_id,
            "position": list(fallback_ranked.action.position),
            "orientation_wxyz": list(fallback_ranked.action.orientation_wxyz),
        }

    def explain_last_choice(self) -> dict[str, Any]:
        return dict(self._last_choice)

    def close(self) -> None:
        if self._executor is None:
            return
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = None

    def _ensure_executor(self) -> ThreadPoolExecutor | None:
        if not self.parallel or self._resolved_max_workers <= 1:
            return None
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._resolved_max_workers,
                thread_name_prefix="extreme-point",
            )
        return self._executor

    def _support_plane_batches(self, support_planes: list[SupportPlane]) -> list[list[SupportPlane]]:
        if not support_planes:
            return []
        phase_limits = sorted({1, 2, 4, 8, len(support_planes)})
        batches: list[list[SupportPlane]] = []
        previous_limit = 0
        for limit in phase_limits:
            if limit <= previous_limit:
                continue
            batches.append(support_planes[previous_limit:limit])
            previous_limit = limit
        return batches

    def _adaptive_search_limits(
        self,
        remaining_seconds: float,
        *,
        hard_state_mode: bool,
    ) -> tuple[int | None, float | None]:
        base_max_candidates = self.max_candidates_per_group
        base_frontier_band = self.frontier_band_delta_x
        if base_max_candidates is None or base_frontier_band is None:
            return base_max_candidates, base_frontier_band
        if hard_state_mode and remaining_seconds > max(2.5, self.fallback_buffer_seconds + 0.5):
            return max(base_max_candidates, 20), max(base_frontier_band, 0.25)
        if remaining_seconds <= 1.0:
            return max(4, min(base_max_candidates, 4)), min(base_frontier_band, 0.05)
        if remaining_seconds <= 2.0:
            return max(6, min(base_max_candidates, 8)), min(base_frontier_band, 0.08)
        if remaining_seconds <= 4.0:
            return max(8, min(base_max_candidates, 12)), min(base_frontier_band, 0.12)
        return base_max_candidates, base_frontier_band

    def _remaining_seconds(self, deadline_monotonic: float) -> float:
        return max(0.0, deadline_monotonic - perf_counter())

    def _proxy_ranking_key(self, candidate: CandidatePlacement, score: Any) -> tuple[object, ...]:
        position = tuple(round(value, 12) for value in candidate.position)
        return (
            -round(score.total_score, 12),
            round(score.delta_x, 12),
            round(score.gap_penalty, 12),
            round(candidate.position[2], 12),
            -round(score.support_reward, 12),
            -round(score.contact_reward, 12),
            round(candidate.position[1], 12),
            candidate.orientation_index,
            position,
        )

    def _merge_proxy_candidates(
        self,
        existing: list[ProxyCandidate],
        incoming: list[ProxyCandidate],
        *,
        limit: int = 6,
    ) -> list[ProxyCandidate]:
        merged: dict[tuple[float, ...], ProxyCandidate] = {}
        for proxy in [*existing, *incoming]:
            key = proxy.candidate.dedup_key
            current = merged.get(key)
            if current is None or proxy.ranking_key() < current.ranking_key():
                merged[key] = proxy
        return sorted(merged.values(), key=lambda item: item.ranking_key())[:limit]

    def _select_budgeted_fallback(
        self,
        view: DecisionStateView,
        *,
        top_proxy_candidates: list[ProxyCandidate],
        deadline_monotonic: float,
    ) -> RankedCandidate | None:
        if view.current_box is None:
            return None

        ranked_fallbacks: list[RankedCandidate] = []
        seen_actions: set[tuple[float, ...]] = set()

        for proxy in top_proxy_candidates[:4]:
            proxy_action = proxy.candidate.as_action(view.current_box.id)
            exact_ranked = self._rank_fallback(view, proxy_action)
            if exact_ranked is not None:
                self._append_ranked_fallback(ranked_fallbacks, exact_ranked, seen_actions)
                continue
            if self._remaining_seconds(deadline_monotonic) <= 0.0:
                break
            aligned_action = self.engine.find_valid_action_at_current_xy(view.game_state, proxy_action)
            aligned_ranked = self._rank_fallback(view, aligned_action) if aligned_action is not None else None
            if aligned_ranked is not None:
                self._append_ranked_fallback(ranked_fallbacks, aligned_ranked, seen_actions)
                if ranked_fallbacks and self._remaining_seconds(deadline_monotonic) <= self.fallback_buffer_seconds:
                    break
                continue
            if self._remaining_seconds(deadline_monotonic) <= 0.0:
                break
            nearby_action = self.engine.find_valid_action_near(view.game_state, proxy_action)
            nearby_ranked = self._rank_fallback(view, nearby_action) if nearby_action is not None else None
            if nearby_ranked is not None:
                self._append_ranked_fallback(ranked_fallbacks, nearby_ranked, seen_actions)
            if ranked_fallbacks and self._remaining_seconds(deadline_monotonic) <= self.fallback_buffer_seconds:
                break

        if not ranked_fallbacks and self._remaining_seconds(deadline_monotonic) > self.fallback_buffer_seconds:
            any_valid_action = self.engine.find_any_valid_action(view.game_state)
            any_valid_ranked = self._rank_fallback(view, any_valid_action) if any_valid_action is not None else None
            if any_valid_ranked is not None:
                self._append_ranked_fallback(ranked_fallbacks, any_valid_ranked, seen_actions)

        if not ranked_fallbacks:
            return None
        ranked_fallbacks.sort(key=lambda item: item.ranking_key())
        return ranked_fallbacks[0]

    def _append_ranked_fallback(
        self,
        ranked_fallbacks: list[RankedCandidate],
        ranked_candidate: RankedCandidate,
        seen_actions: set[tuple[float, ...]],
    ) -> None:
        action = ranked_candidate.action
        key = tuple(round(value, 6) for value in (*action.position, *canonicalize_quaternion_sign(action.orientation_wxyz)))
        if key in seen_actions:
            return
        seen_actions.add(key)
        ranked_fallbacks.append(ranked_candidate)

    def _rank_fallback(self, view: DecisionStateView, fallback_action: PlacementAction | None) -> RankedCandidate | None:
        if fallback_action is None:
            return None
        validation = self.engine.validate_place_action(view.game_state, fallback_action)
        if not validation.is_valid or validation.normalized_action is None:
            return None
        normalized_action = validation.normalized_action
        orientations = get_orthogonal_orientation_options(
            view.current_box.dimensions,
            vertical_axis_cos_tolerance=self.engine.config.vertical_axis_cos_tolerance,
        )
        canonical_orientation = canonicalize_quaternion_sign(normalized_action.orientation_wxyz)
        orientation = next(
            (
                option
                for option in orientations
                if canonicalize_quaternion_sign(option.orientation_wxyz) == canonical_orientation
            ),
            build_orientation_option_for_quaternion(
                view.current_box.dimensions,
                normalized_action.orientation_wxyz,
                index=len(orientations),
                vertical_axis_cos_tolerance=self.engine.config.vertical_axis_cos_tolerance,
            ),
        )
        candidate = CandidatePlacement(
            group_index=-1,
            support_plane_id=f"fallback:{normalized_action.position[2] + orientation.bottom_z:.6f}",
            support_plane_index=-1,
            orientation_index=orientation.index,
            anchor_style="fallback",
            position=normalized_action.position,
            orientation_wxyz=normalized_action.orientation_wxyz,
            support_component_bounds=None,
            dedup_key=(),
            sort_key=(),
        )
        score = compute_score_breakdown(
            view,
            candidate,
            orientation,
            support_ratio=validation.support_ratio,
            weights=self.score_weights,
        )
        return RankedCandidate(
            candidate=candidate,
            action=normalized_action,
            score=score,
            support_ratio=float(validation.support_ratio or 0.0),
            validation_message=validation.message,
            validation_category=validation.category,
        )

    def _remember_choice(
        self,
        ranked_candidate: RankedCandidate,
        *,
        summary: EvaluationSummary,
        fallback_used: bool,
        support_plane_extraction_ms: float,
        candidate_generation_ms: float,
        fallback_ms: float,
        decision_time_ms: float,
    ) -> None:
        score = ranked_candidate.score
        self._last_choice = {
            "chosen_action": {
                "box_id": ranked_candidate.action.box_id,
                "position": list(ranked_candidate.action.position),
                "orientation_wxyz": list(ranked_candidate.action.orientation_wxyz),
            },
            "total_score": score.total_score,
            "delta_x": score.delta_x,
            "gap_penalty": score.gap_penalty,
            "front_gap": score.front_gap,
            "frontier_slack": score.frontier_slack,
            "future_sliver_penalty": score.future_sliver_penalty,
            "fragmentation_penalty": score.fragmentation_penalty,
            "left_gap": score.left_gap,
            "right_gap": score.right_gap,
            "support_reward": score.support_reward,
            "contact_reward": score.contact_reward,
            "frontier_reach_reward": score.frontier_reach_reward,
            "future_usable_area_reward": score.future_usable_area_reward,
            "shelf_completion_reward": score.shelf_completion_reward,
            "support_ratio": ranked_candidate.support_ratio,
            "fallback_used": fallback_used,
            "parallel_used": summary.parallel_used,
            "candidates_generated": summary.generated_count,
            "candidates_validated": summary.validated_count,
            "candidates_pruned": summary.pruned_count,
            "candidates_skipped_by_bound": summary.skipped_by_bound_count,
            "deadline_hit": summary.deadline_hit,
            "validation_budget_hit": summary.validation_budget_hit,
            "valid_candidates": summary.valid_count,
            "group_count": summary.group_count,
            "support_plane_extraction_ms": support_plane_extraction_ms,
            "candidate_generation_ms": candidate_generation_ms,
            "evaluation_preparation_ms": summary.preparation_time_ms,
            "evaluation_validation_ms": summary.validation_time_ms,
            "fallback_ms": fallback_ms,
            "decision_time_ms": decision_time_ms,
            "invalid_reason_counts": dict(summary.invalid_reason_counts),
            "valid_counts_by_group": dict(summary.valid_counts_by_group),
            "anchor_style": ranked_candidate.candidate.anchor_style,
            "support_plane_id": ranked_candidate.candidate.support_plane_id,
            "orientation_index": ranked_candidate.candidate.orientation_index,
        }

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass
