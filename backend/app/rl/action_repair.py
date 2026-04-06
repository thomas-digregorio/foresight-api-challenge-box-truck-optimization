from __future__ import annotations

from app.engine.truck_packing_engine import TruckPackingEngine
from app.models.entities import GameState, PlacementAction


class ActionRepairPolicy:
    def __init__(self, engine: TruckPackingEngine) -> None:
        self.engine = engine

    def repair(self, state: GameState, action: PlacementAction) -> PlacementAction | None:
        return self.engine.find_valid_action_near(state, action)

