from __future__ import annotations

import argparse
import os
import statistics

from app.agents.extreme_point import GreedyExtremePointAgent
from app.agents.extreme_point.http_client import ChallengeLikeHttpClient
from app.agents.extreme_point.remote_runner import run_remote_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the greedy extreme-point agent against a challenge-like HTTP API.")
    parser.add_argument("--base-url", default=os.getenv("FORESIGHT_API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--api-key", default=os.getenv("FORESIGHT_API_KEY", "local-dev"))
    parser.add_argument("--path-prefix", default="/challenge/api")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--mode", choices=("dev", "compete"), default="dev")
    parser.add_argument("--parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-workers", type=int, default=None)
    args = parser.parse_args()

    agent = GreedyExtremePointAgent(parallel=args.parallel, max_workers=args.max_workers)
    client = ChallengeLikeHttpClient(
        base_url=args.base_url,
        api_key=args.api_key,
        path_prefix=args.path_prefix,
    )
    try:
        results = [run_remote_episode(client=client, agent=agent, mode=args.mode) for _ in range(args.episodes)]
    finally:
        client.close()

    densities = [result.density for result in results]
    print(f"episodes={len(results)}")
    print(f"mean_density={statistics.fmean(densities):.6f}")
    print(f"min_density={min(densities):.6f}")
    print(f"max_density={max(densities):.6f}")
    print(f"fallbacks={sum(result.fallback_count for result in results)}")


if __name__ == "__main__":
    main()
