from __future__ import annotations

from app.core.exceptions import NotFoundError
from app.engine.truck_packing_engine import TruckPackingEngine
from app.models.entities import GameState, PlacementAction
from app.services.episode_registry import EpisodeRegistry


class EpisodeService:
    def __init__(self, registry: EpisodeRegistry, engine: TruckPackingEngine) -> None:
        self.registry = registry
        self.engine = engine

    def start_episode(
        self,
        *,
        mode: str,
        seed: int | None,
        api_key: str,
        api_variant: str,
        enable_local_timeout: bool = False,
        auto_terminate_on_no_feasible_placement: bool = False,
    ) -> GameState:
        state = self.engine.start_episode(
            mode=mode,
            seed=seed,
            enable_local_timeout=enable_local_timeout,
            auto_terminate_on_no_feasible_placement=auto_terminate_on_no_feasible_placement,
            metadata={
                "api_key": api_key,
                "api_variant": api_variant,
                "loading_guide_x": self.engine.config.local_loading_guide_x if api_variant == "local" else None,
            },
        )
        self.registry.add(state)
        return state

    def get_state(self, game_id: str, *, expected_api_variant: str) -> GameState:
        state = self.registry.get(game_id)
        self._ensure_api_variant(state, expected_api_variant)
        self.engine.handle_timeout_if_needed(state)
        self.registry.update(state)
        return state

    def place_box(
        self,
        game_id: str,
        action: PlacementAction,
        *,
        source: str = "manual",
        expected_api_variant: str,
    ) -> GameState:
        state = self.registry.get(game_id)
        self._ensure_api_variant(state, expected_api_variant)
        self.engine.handle_timeout_if_needed(state)
        self.engine.commit_place_action(state, action, source=source)
        self.registry.update(state)
        return state

    def stop_episode(self, game_id: str, *, expected_api_variant: str) -> GameState:
        state = self.registry.get(game_id)
        self._ensure_api_variant(state, expected_api_variant)
        self.engine.stop_episode(state)
        self.registry.update(state)
        return state

    def _ensure_api_variant(self, state: GameState, expected_api_variant: str) -> None:
        if state.metadata.get("api_variant") == expected_api_variant:
            return
        raise NotFoundError(
            f"Unknown game_id: {state.game_id}",
            category="invalid_game_id",
            details={"game_id": state.game_id},
        )
