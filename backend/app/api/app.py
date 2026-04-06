from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.exceptions import EngineError, NotFoundError, StateConflictError, ValidationError
from app.engine.truck_packing_engine import TruckPackingEngine
from app.services.episode_registry import EpisodeRegistry
from app.services.episode_service import EpisodeService
from app.services.preview_service import PreviewService


@dataclass(slots=True)
class AppServices:
    registry: EpisodeRegistry
    engine: TruckPackingEngine
    episode_service: EpisodeService
    preview_service: PreviewService


def create_app() -> FastAPI:
    registry = EpisodeRegistry()
    engine = TruckPackingEngine()
    services = AppServices(
        registry=registry,
        engine=engine,
        episode_service=EpisodeService(registry, engine),
        preview_service=PreviewService(registry, engine),
    )
    app = FastAPI(title="Foresight Local Challenge", version="0.1.0")
    app.state.services = services
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.exception_handler(NotFoundError)
    async def handle_not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content=exc.to_payload())

    @app.exception_handler(ValidationError)
    async def handle_validation(_: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content=exc.to_payload())

    @app.exception_handler(StateConflictError)
    async def handle_conflict(_: Request, exc: StateConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content=exc.to_payload())

    @app.exception_handler(EngineError)
    async def handle_engine(_: Request, exc: EngineError) -> JSONResponse:
        return JSONResponse(status_code=400, content=exc.to_payload())

    return app

