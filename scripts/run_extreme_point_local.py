from __future__ import annotations

import argparse
import statistics

from app.agents.extreme_point import GreedyExtremePointAgent
from app.agents.extreme_point.local_runner import run_local_episode
from app.engine.truck_packing_engine import TruckPackingEngine
from app.models.entities import EngineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the greedy extreme-point agent against the local in-process engine.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mode", choices=("dev", "compete"), default="dev")
    parser.add_argument("--queue-length", type=int, default=32)
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    engine = TruckPackingEngine(config=EngineConfig(default_queue_length=args.queue_length))
    agent = GreedyExtremePointAgent(engine=engine, parallel=args.parallel, max_workers=args.max_workers)

    results = [
        run_local_episode(
            agent=agent,
            seed=args.seed + episode_index,
            mode=args.mode,
            engine=engine,
        )
        for episode_index in range(args.episodes)
    ]
    densities = [result.density for result in results]
    print(f"episodes={len(results)}")
    print(f"mean_density={statistics.fmean(densities):.6f}")
    print(f"min_density={min(densities):.6f}")
    print(f"max_density={max(densities):.6f}")
    print(f"fallbacks={sum(result.fallback_count for result in results)}")
    print(f"invalid_submissions={sum(result.invalid_submission_count for result in results)}")


if __name__ == "__main__":
    main()
