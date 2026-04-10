from __future__ import annotations

from typing import Any

import numpy as np

from app.engine.truck_packing_engine import TruckPackingEngine
from app.models.entities import PlacementAction
from app.rl.action_repair import ActionRepairPolicy
from app.services.episode_registry import EpisodeRegistry
from app.services.episode_service import EpisodeService
from app.api.serializers import serialize_local_state


class RawEpisodeEnv:
    """Raw variable-length environment wrapper.

    A future strict Gymnasium adapter can sit on top of this and encode fixed-size observations.
    """

    def __init__(
        self,
        *,
        engine: TruckPackingEngine | None = None,
        registry: EpisodeRegistry | None = None,
        mode: str = "dev",
    ) -> None:
        self.engine = engine or TruckPackingEngine()
        self.registry = registry or EpisodeRegistry()
        self.service = EpisodeService(self.registry, self.engine)
        self.repair_policy = ActionRepairPolicy(self.engine)
        self.mode = mode
        self.game_id: str | None = None

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        state = self.service.start_episode(
            mode=self.mode,
            seed=seed,
            api_key="local-rl",
            api_variant="local",
            enable_local_timeout=True,
            auto_terminate_on_no_feasible_placement=True,
        )
        self.game_id = state.game_id
        return serialize_local_state(state, self.engine.config)

    def step(self, action: dict[str, Any] | np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.game_id is None:
            obs = self.reset()
            return obs, 0.0, False, False, {"auto_reset": True}
        state = self.service.get_state(self.game_id, expected_api_variant="local")
        if isinstance(action, np.ndarray):
            payload = action.tolist()
            if len(payload) != 7:
                raise ValueError("np.ndarray action must have 7 values: xyz + quaternion wxyz.")
            placement = PlacementAction(
                box_id=state.current_box.id if state.current_box is not None else "",
                position=(float(payload[0]), float(payload[1]), float(payload[2])),
                orientation_wxyz=(float(payload[3]), float(payload[4]), float(payload[5]), float(payload[6])),
            )
        else:
            placement = PlacementAction(
                box_id=str(action["box_id"]),
                position=tuple(action["position"]),
                orientation_wxyz=tuple(action["orientation_wxyz"]),
            )

        repaired = self.repair_policy.repair(state, placement)
        info: dict[str, Any] = {"repair_attempted": True}
        if repaired is None:
            state.game_status = "timed_out"
            state.termination_reason = "repair_failed"
            state.timer_state = None
            self.registry.update(state)
            obs = serialize_local_state(state, self.engine.config)
            return obs, 0.0, False, True, {"repair_failed": True}

        self.service.place_box(self.game_id, repaired, source="rl_repair", expected_api_variant="local")
        next_state = self.service.get_state(self.game_id, expected_api_variant="local")
        obs = serialize_local_state(next_state, self.engine.config)
        terminated = next_state.game_status == "completed"
        truncated = next_state.game_status == "timed_out"
        reward = next_state.density if terminated else 0.0
        info["repaired_action"] = {
            "box_id": repaired.box_id,
            "position": repaired.position,
            "orientation_wxyz": repaired.orientation_wxyz,
        }
        info["game_status"] = next_state.game_status
        return obs, reward, terminated, truncated, info
