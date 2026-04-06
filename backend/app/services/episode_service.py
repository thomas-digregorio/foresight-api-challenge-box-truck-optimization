from __future__ import annotations

from app.engine.truck_packing_engine import TruckPackingEngine
from app.models.entities import GameState, PlacementAction
from app.services.episode_registry import EpisodeRegistry


class EpisodeService:
    def __init__(self, registry: EpisodeRegistry, engine: TruckPackingEngine) -> None:
        self.registry = registry
        self.engine = engine

    def start_episode(self, *, mode: str, seed: int | None) -> GameState:
        state = self.engine.start_episode(mode=mode, seed=seed)
        self.registry.add(state)
        return state

    def get_state(self, game_id: str) -> GameState:
        state = self.registry.get(game_id)
        self.engine.handle_timeout_if_needed(state)
        self.registry.update(state)
        return state

    def place_box(self, game_id: str, action: PlacementAction, *, source: str = "manual") -> GameState:
        state = self.registry.get(game_id)
        self.engine.handle_timeout_if_needed(state)
        self.engine.commit_place_action(state, action, source=source)
        self.registry.update(state)
        return state

    def stop_episode(self, game_id: str) -> GameState:
        state = self.registry.get(game_id)
        self.engine.stop_episode(state)
        self.registry.update(state)
        return state

