from __future__ import annotations

from typing import Any, Callable

from app.rl.raw_env import RawEpisodeEnv


class ParallelEpisodeManager:
    def __init__(self, env_factory: Callable[[], RawEpisodeEnv] | None = None) -> None:
        self._env_factory = env_factory or (lambda: RawEpisodeEnv())
        self._envs: list[RawEpisodeEnv] = []

    def create_many(self, count: int) -> list[RawEpisodeEnv]:
        self._envs = [self._env_factory() for _ in range(count)]
        return self._envs

    def reset_many(self, seeds: list[int | None] | None = None) -> list[dict[str, Any]]:
        if not self._envs:
            return []
        seeds = seeds or [None] * len(self._envs)
        return [env.reset(seed=seed) for env, seed in zip(self._envs, seeds, strict=False)]

    def step_many(self, actions: list[dict[str, Any]]) -> list[tuple[dict[str, Any], float, bool, bool, dict[str, Any]]]:
        return [env.step(action) for env, action in zip(self._envs, actions, strict=False)]
