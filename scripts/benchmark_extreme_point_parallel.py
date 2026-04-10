from __future__ import annotations

import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor

from app.agents.extreme_point import GreedyExtremePointAgent
from app.agents.extreme_point.local_runner import LocalEpisodeResult, run_local_episode
from app.engine.truck_packing_engine import TruckPackingEngine
from app.models.entities import EngineConfig


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def run_single(seed: int, *, queue_length: int, parallel: bool, agent_max_workers: int | None) -> LocalEpisodeResult:
    engine = TruckPackingEngine(config=EngineConfig(default_queue_length=queue_length))
    agent = GreedyExtremePointAgent(engine=engine, parallel=parallel, max_workers=agent_max_workers)
    return run_local_episode(agent=agent, seed=seed, engine=engine)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark serial vs parallel greedy extreme-point local episodes.")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--queue-length", type=int, default=32)
    parser.add_argument("--agent-max-workers", type=int, default=None)
    parser.add_argument("--serial-only", action="store_true")
    args = parser.parse_args()

    seeds = [args.seed + offset for offset in range(args.episodes)]
    modes = [("serial", False)] if args.serial_only else [("serial", False), ("parallel", True)]

    for label, parallel in modes:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                executor.map(
                    lambda seed: run_single(
                        seed,
                        queue_length=args.queue_length,
                        parallel=parallel,
                        agent_max_workers=args.agent_max_workers,
                    ),
                    seeds,
                )
            )

        densities = [result.density for result in results]
        all_latencies = [latency for result in results for latency in result.decision_latencies_ms]
        generated = [count for result in results for count in result.candidates_generated_per_move]
        validated = [count for result in results for count in result.candidates_validated_per_move]
        print(f"[{label}] episodes={len(results)}")
        print(f"[{label}] mean_density={statistics.fmean(densities):.6f}")
        print(f"[{label}] median_density={statistics.median(densities):.6f}")
        print(f"[{label}] min_density={min(densities):.6f}")
        print(f"[{label}] max_density={max(densities):.6f}")
        print(f"[{label}] invalid_actions={sum(result.invalid_submission_count for result in results)}")
        print(f"[{label}] fallback_count={sum(result.fallback_count for result in results)}")
        print(f"[{label}] avg_decision_ms={statistics.fmean(all_latencies) if all_latencies else 0.0:.3f}")
        print(f"[{label}] p50_decision_ms={percentile(all_latencies, 0.5):.3f}")
        print(f"[{label}] p90_decision_ms={percentile(all_latencies, 0.9):.3f}")
        print(f"[{label}] avg_candidates_generated={statistics.fmean(generated) if generated else 0.0:.2f}")
        print(f"[{label}] avg_candidates_validated={statistics.fmean(validated) if validated else 0.0:.2f}")


if __name__ == "__main__":
    main()
