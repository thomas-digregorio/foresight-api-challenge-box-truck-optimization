from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from math import inf
from threading import Lock
from time import perf_counter

from app.engine.truck_packing_engine import TruckPackingEngine

from app.agents.extreme_point.scoring import compute_score_breakdown, score_breakdown_with_support_ratio
from app.agents.extreme_point.state_view import DecisionStateView
from app.agents.extreme_point.types import (
    CandidateGroup,
    CandidatePlacement,
    EvaluationSummary,
    ProxyCandidate,
    RankedCandidate,
    ScoreBreakdown,
    ScoreWeights,
)


BOUND_EPSILON = 1e-12
DOMINANCE_EPSILON = 1e-9


@dataclass(slots=True)
class CandidateEstimate:
    candidate: CandidatePlacement
    proxy_score: ScoreBreakdown

    def ranking_key(self) -> tuple[object, ...]:
        position = tuple(round(value, 12) for value in self.candidate.position)
        return (
            -round(self.proxy_score.total_score, 12),
            round(self.proxy_score.delta_x, 12),
            round(self.proxy_score.gap_penalty, 12),
            round(self.candidate.position[2], 12),
            -round(self.proxy_score.support_reward, 12),
            -round(self.proxy_score.contact_reward, 12),
            round(self.candidate.position[1], 12),
            self.candidate.orientation_index,
            position,
        )


@dataclass(slots=True)
class PreparedCandidateGroup:
    group: CandidateGroup
    estimates: list[CandidateEstimate]
    optimistic_upper_bound: float
    pruned_count: int


@dataclass(slots=True)
class SharedEvaluationState:
    deadline_monotonic: float | None
    max_total_validations: int | None
    best_score: float
    validation_count: int = 0
    deadline_hit: bool = False
    validation_budget_hit: bool = False
    lock: Lock = field(default_factory=Lock)

    def try_begin_validation(
        self,
        *,
        optimistic_upper_bound: float,
        local_best_score: float,
        enable_branch_and_bound: bool,
    ) -> str:
        with self.lock:
            if self.deadline_monotonic is not None and perf_counter() >= self.deadline_monotonic:
                self.deadline_hit = True
                return "deadline"
            incumbent_score = max(self.best_score, local_best_score)
            if enable_branch_and_bound and optimistic_upper_bound < incumbent_score - BOUND_EPSILON:
                return "bound"
            if self.max_total_validations is not None and self.validation_count >= self.max_total_validations:
                self.validation_budget_hit = True
                return "budget"
            self.validation_count += 1
            return "validate"

    def update_best_score(self, score: float) -> None:
        with self.lock:
            if score > self.best_score:
                self.best_score = score


def _estimate_candidate(
    view: DecisionStateView,
    group: CandidateGroup,
    *,
    weights: ScoreWeights,
    deadline_monotonic: float | None = None,
) -> list[CandidateEstimate]:
    estimates: list[CandidateEstimate] = []
    for candidate in group.candidates:
        if deadline_monotonic is not None and perf_counter() >= deadline_monotonic:
            break
        estimates.append(
            CandidateEstimate(
                candidate=candidate,
                proxy_score=compute_score_breakdown(
                    view,
                    candidate,
                    group.orientation,
                    support_ratio=1.0,
                    weights=weights,
                ),
            )
        )
    return estimates


def _apply_frontier_band(
    estimates: list[CandidateEstimate],
    *,
    frontier_band_delta_x: float | None,
) -> list[CandidateEstimate]:
    if frontier_band_delta_x is None or not estimates:
        return list(estimates)
    min_delta_x = min(estimate.proxy_score.delta_x for estimate in estimates)
    limit = min_delta_x + frontier_band_delta_x + DOMINANCE_EPSILON
    return [estimate for estimate in estimates if estimate.proxy_score.delta_x <= limit]


def _dominates(lhs: CandidateEstimate, rhs: CandidateEstimate) -> bool:
    no_worse = (
        lhs.proxy_score.delta_x <= rhs.proxy_score.delta_x + DOMINANCE_EPSILON
        and lhs.proxy_score.front_gap <= rhs.proxy_score.front_gap + DOMINANCE_EPSILON
        and lhs.proxy_score.frontier_slack <= rhs.proxy_score.frontier_slack + DOMINANCE_EPSILON
        and lhs.proxy_score.left_gap <= rhs.proxy_score.left_gap + DOMINANCE_EPSILON
        and lhs.proxy_score.right_gap <= rhs.proxy_score.right_gap + DOMINANCE_EPSILON
        and lhs.proxy_score.contact_reward + DOMINANCE_EPSILON >= rhs.proxy_score.contact_reward
        and lhs.candidate.position[2] <= rhs.candidate.position[2] + DOMINANCE_EPSILON
    )
    strictly_better = (
        lhs.proxy_score.delta_x < rhs.proxy_score.delta_x - DOMINANCE_EPSILON
        or lhs.proxy_score.front_gap < rhs.proxy_score.front_gap - DOMINANCE_EPSILON
        or lhs.proxy_score.frontier_slack < rhs.proxy_score.frontier_slack - DOMINANCE_EPSILON
        or lhs.proxy_score.left_gap < rhs.proxy_score.left_gap - DOMINANCE_EPSILON
        or lhs.proxy_score.right_gap < rhs.proxy_score.right_gap - DOMINANCE_EPSILON
        or lhs.proxy_score.contact_reward > rhs.proxy_score.contact_reward + DOMINANCE_EPSILON
        or lhs.candidate.position[2] < rhs.candidate.position[2] - DOMINANCE_EPSILON
    )
    return bool(no_worse and strictly_better)


def _apply_dominance_pruning(estimates: list[CandidateEstimate]) -> list[CandidateEstimate]:
    ordered = sorted(estimates, key=lambda estimate: estimate.ranking_key())
    survivors: list[CandidateEstimate] = []
    for estimate in ordered:
        if any(_dominates(existing, estimate) for existing in survivors):
            continue
        survivors = [existing for existing in survivors if not _dominates(estimate, existing)]
        survivors.append(estimate)
    survivors.sort(key=lambda estimate: estimate.ranking_key())
    return survivors


def _prepare_group(
    view: DecisionStateView,
    group: CandidateGroup,
    *,
    weights: ScoreWeights,
    max_candidates_per_group: int | None,
    frontier_band_delta_x: float | None,
    enable_dominance_pruning: bool,
    deadline_monotonic: float | None,
) -> PreparedCandidateGroup:
    original_estimates = _estimate_candidate(
        view,
        group,
        weights=weights,
        deadline_monotonic=deadline_monotonic,
    )
    estimates = _apply_frontier_band(original_estimates, frontier_band_delta_x=frontier_band_delta_x)
    if enable_dominance_pruning:
        estimates = _apply_dominance_pruning(estimates)
    estimates.sort(key=lambda estimate: estimate.ranking_key())
    if max_candidates_per_group is not None:
        estimates = estimates[:max_candidates_per_group]
    return PreparedCandidateGroup(
        group=group,
        estimates=estimates,
        optimistic_upper_bound=(-inf if not estimates else estimates[0].proxy_score.total_score),
        pruned_count=max(0, len(original_estimates) - len(estimates)),
    )


def _evaluate_group(
    view: DecisionStateView,
    prepared_group: PreparedCandidateGroup,
    engine: TruckPackingEngine,
    weights: ScoreWeights,
    shared_state: SharedEvaluationState,
    enable_branch_and_bound: bool,
) -> tuple[list[RankedCandidate], int, dict[str, int], dict[str, int]]:
    ranked_candidates: list[RankedCandidate] = []
    skipped_by_bound_count = 0
    invalid_reason_counts: dict[str, int] = {}
    valid_counts_by_group: dict[str, int] = {}
    local_best_score = -inf
    group = prepared_group.group
    group_key = f"{group.support_plane.plane_id}|ori:{group.orientation.index}"
    for estimate_index, estimate in enumerate(prepared_group.estimates):
        optimistic_upper_bound = estimate.proxy_score.total_score
        candidate_state = shared_state.try_begin_validation(
            optimistic_upper_bound=optimistic_upper_bound,
            local_best_score=local_best_score,
            enable_branch_and_bound=enable_branch_and_bound,
        )
        if candidate_state == "bound":
            skipped_by_bound_count += len(prepared_group.estimates) - estimate_index
            break
        if candidate_state in {"deadline", "budget"}:
            break
        candidate = estimate.candidate
        validation = engine.validate_place_action(
            view.game_state,
            candidate.as_action(view.current_box.id if view.current_box is not None else ""),
        )
        if not validation.is_valid or validation.normalized_action is None:
            reason = validation.category or "invalid"
            invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + 1
            continue
        breakdown = score_breakdown_with_support_ratio(
            estimate.proxy_score,
            support_ratio=validation.support_ratio,
            weights=weights,
        )
        ranked_candidates.append(
            RankedCandidate(
                candidate=candidate,
                action=validation.normalized_action,
                score=breakdown,
                support_ratio=float(validation.support_ratio or 0.0),
                validation_message=validation.message,
                validation_category=validation.category,
            )
        )
        valid_counts_by_group[group_key] = valid_counts_by_group.get(group_key, 0) + 1
        if breakdown.total_score > local_best_score:
            local_best_score = breakdown.total_score
            shared_state.update_best_score(breakdown.total_score)
    ranked_candidates.sort(key=lambda item: item.ranking_key())
    return ranked_candidates, skipped_by_bound_count, invalid_reason_counts, valid_counts_by_group


def evaluate_candidate_groups(
    view: DecisionStateView,
    groups: list[CandidateGroup],
    *,
    engine: TruckPackingEngine,
    weights: ScoreWeights,
    parallel: bool,
    max_workers: int | None,
    parallel_candidate_threshold: int,
    max_candidates_per_group: int | None = 16,
    frontier_band_delta_x: float | None = 0.2,
    enable_branch_and_bound: bool = True,
    enable_dominance_pruning: bool = True,
    deadline_monotonic: float | None = None,
    max_total_validations: int | None = None,
    incumbent_score: float = -inf,
    executor: ThreadPoolExecutor | None = None,
) -> EvaluationSummary:
    generated_count = sum(len(group.candidates) for group in groups)
    if not groups:
        return EvaluationSummary(
            ranked_candidates=[],
            generated_count=0,
            validated_count=0,
            valid_count=0,
            parallel_used=False,
            group_count=0,
            pruned_count=0,
            skipped_by_bound_count=0,
            deadline_hit=False,
            validation_budget_hit=False,
            best_proxy_candidate=None,
            best_proxy_score=None,
            top_proxy_candidates=[],
            preparation_time_ms=0.0,
            validation_time_ms=0.0,
        )

    preparation_started = perf_counter()
    preparation_deadline_hit = False
    prepared_groups: list[PreparedCandidateGroup] = []
    for group in groups:
        if deadline_monotonic is not None and perf_counter() >= deadline_monotonic:
            preparation_deadline_hit = True
            break
        prepared_groups.append(
            _prepare_group(
                view,
                group,
                weights=weights,
                max_candidates_per_group=max_candidates_per_group,
                frontier_band_delta_x=frontier_band_delta_x,
                enable_dominance_pruning=enable_dominance_pruning,
                deadline_monotonic=deadline_monotonic,
            )
        )
        if deadline_monotonic is not None and perf_counter() >= deadline_monotonic:
            preparation_deadline_hit = True
            break
    pruned_count = sum(group.pruned_count for group in prepared_groups)
    preparation_time_ms = (perf_counter() - preparation_started) * 1000.0
    prepared_groups = [group for group in prepared_groups if group.estimates]
    if not prepared_groups:
        return EvaluationSummary(
            ranked_candidates=[],
            generated_count=generated_count,
            validated_count=0,
            valid_count=0,
            parallel_used=False,
            group_count=0,
            pruned_count=pruned_count,
            skipped_by_bound_count=0,
            deadline_hit=preparation_deadline_hit,
            validation_budget_hit=False,
            best_proxy_candidate=None,
            best_proxy_score=None,
            top_proxy_candidates=[],
            preparation_time_ms=preparation_time_ms,
            validation_time_ms=0.0,
        )
    prepared_groups.sort(
        key=lambda prepared: (
            -round(prepared.optimistic_upper_bound, 12),
            round(prepared.group.support_plane.z_support, 12),
            prepared.group.orientation.index,
            prepared.group.index,
        )
    )
    top_proxy_candidates = sorted(
        (
            ProxyCandidate(candidate=estimate.candidate, score=estimate.proxy_score)
            for prepared in prepared_groups
            for estimate in prepared.estimates
        ),
        key=lambda item: item.ranking_key(),
    )[:6]
    best_proxy_candidate = top_proxy_candidates[0].candidate
    best_proxy_score = top_proxy_candidates[0].score
    parallel_used = bool(parallel and generated_count >= parallel_candidate_threshold and len(prepared_groups) > 1)
    shared_state = SharedEvaluationState(
        deadline_monotonic=deadline_monotonic,
        max_total_validations=max_total_validations,
        best_score=incumbent_score,
    )
    results: list[tuple[list[RankedCandidate], int, dict[str, int], dict[str, int]]]
    validation_started = perf_counter()
    if parallel_used:
        if executor is None:
            with ThreadPoolExecutor(max_workers=max_workers) as ephemeral_executor:
                futures = [
                    ephemeral_executor.submit(
                        _evaluate_group,
                        view,
                        group,
                        engine,
                        weights,
                        shared_state,
                        enable_branch_and_bound,
                    )
                    for group in prepared_groups
                ]
                results = [future.result() for future in futures]
        else:
            futures = [
                executor.submit(
                    _evaluate_group,
                    view,
                    group,
                    engine,
                    weights,
                    shared_state,
                    enable_branch_and_bound,
                )
                for group in prepared_groups
            ]
            results = [future.result() for future in futures]
    else:
        results = []
        for group in prepared_groups:
            if enable_branch_and_bound and group.optimistic_upper_bound < shared_state.best_score - BOUND_EPSILON:
                results.append(([], len(group.estimates), {}, {}))
                continue
            group_results = _evaluate_group(
                view,
                group,
                engine,
                weights,
                shared_state,
                enable_branch_and_bound,
            )
            results.append(group_results)
            if shared_state.deadline_hit or shared_state.validation_budget_hit:
                break

    ranked_candidates: list[RankedCandidate] = []
    skipped_by_bound_count = 0
    invalid_reason_counts: dict[str, int] = {}
    valid_counts_by_group: dict[str, int] = {}
    for group_candidates, group_skipped_count, group_invalid_counts, group_valid_counts in results:
        ranked_candidates.extend(group_candidates)
        skipped_by_bound_count += group_skipped_count
        for reason, count in group_invalid_counts.items():
            invalid_reason_counts[reason] = invalid_reason_counts.get(reason, 0) + count
        for group_key, count in group_valid_counts.items():
            valid_counts_by_group[group_key] = valid_counts_by_group.get(group_key, 0) + count
    ranked_candidates.sort(key=lambda item: item.ranking_key())
    validation_time_ms = (perf_counter() - validation_started) * 1000.0
    return EvaluationSummary(
        ranked_candidates=ranked_candidates,
        generated_count=generated_count,
        validated_count=shared_state.validation_count,
        valid_count=len(ranked_candidates),
        parallel_used=parallel_used,
        group_count=len(prepared_groups),
        pruned_count=pruned_count,
        skipped_by_bound_count=skipped_by_bound_count,
        deadline_hit=preparation_deadline_hit or shared_state.deadline_hit,
        validation_budget_hit=shared_state.validation_budget_hit,
        best_proxy_candidate=best_proxy_candidate,
        best_proxy_score=best_proxy_score,
        top_proxy_candidates=top_proxy_candidates,
        preparation_time_ms=preparation_time_ms,
        validation_time_ms=validation_time_ms,
        invalid_reason_counts=invalid_reason_counts,
        valid_counts_by_group=valid_counts_by_group,
    )
