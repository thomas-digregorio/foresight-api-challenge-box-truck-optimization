from __future__ import annotations

from app.rl.parallel_manager import ParallelEpisodeManager


def test_parallel_manager_handles_independent_episodes() -> None:
    manager = ParallelEpisodeManager()
    envs = manager.create_many(2)
    observations = manager.reset_many([1, 2])

    assert len(envs) == 2
    assert len(observations) == 2
    assert observations[0]["game_id"] != observations[1]["game_id"]
    assert observations[0]["current_box"]["id"] != observations[1]["current_box"]["id"] or observations[0]["boxes_remaining"] == observations[1]["boxes_remaining"]
