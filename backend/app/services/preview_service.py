from __future__ import annotations

from app.core.exceptions import NotFoundError
from app.engine.truck_packing_engine import TruckPackingEngine
from app.models.entities import PlacementAction, PreviewAction, ValidationResult
from app.services.episode_registry import EpisodeRegistry


class PreviewService:
    def __init__(self, registry: EpisodeRegistry, engine: TruckPackingEngine) -> None:
        self.registry = registry
        self.engine = engine

    def update_preview(
        self,
        game_id: str,
        action: PreviewAction,
        *,
        expected_api_variant: str,
    ) -> tuple[ValidationResult, PlacementAction | None, PlacementAction | None, PlacementAction | None]:
        state = self.registry.get(game_id)
        if state.metadata.get("api_variant") != expected_api_variant:
            raise NotFoundError(
                f"Unknown game_id: {game_id}",
                category="invalid_game_id",
                details={"game_id": game_id},
            )
        validation = self.engine.update_preview(state, action)
        current_box = state.current_box
        support_aligned = None
        nearby_valid = None
        any_valid = None
        if current_box is not None:
            normalized_action = validation.normalized_action or PlacementAction(
                box_id=action.box_id,
                position=action.position,
                orientation_wxyz=action.orientation_wxyz,
            )
            support_aligned = self.engine.find_valid_action_at_current_xy(state, normalized_action)
            if not validation.is_valid:
                nearby_valid = self.engine.find_valid_action_near(state, normalized_action)
                any_valid = self.engine.find_any_valid_action(state)
        self.registry.update(state)
        return validation, support_aligned, nearby_valid, any_valid
