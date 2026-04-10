from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
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


def _challenge_error_payload(*, error: str, message: str) -> dict[str, str]:
    return {"error": error, "message": message}


def _local_error_payload(message: str, category: str, details: dict[str, object]) -> dict[str, object]:
    return {
        "error": "validation_error",
        "message": message,
        "details": {
            "category": category,
            **details,
        },
    }


def build_error_response(path: str, exc: Exception) -> tuple[int, dict[str, object]]:
    is_challenge_request = path.startswith("/challenge/api/")
    if isinstance(exc, NotFoundError):
        if is_challenge_request:
            return 404, _challenge_error_payload(error="invalid_game_id", message=exc.message)
        return 404, _local_error_payload(exc.message, exc.category, exc.details)
    if isinstance(exc, ValidationError):
        if is_challenge_request:
            error = "invalid_box_id" if exc.category == "invalid_box_id" else "validation"
            status_code = 400 if exc.category == "invalid_box_id" else 422
            return status_code, _challenge_error_payload(error=error, message=exc.message)
        return 422, _local_error_payload(exc.message, exc.category, exc.details)
    if isinstance(exc, StateConflictError):
        if is_challenge_request:
            return 404, _challenge_error_payload(error="invalid_game_id", message=exc.message)
        return 409, _local_error_payload(exc.message, exc.category, exc.details)
    if isinstance(exc, EngineError):
        if is_challenge_request:
            return 400, _challenge_error_payload(error="validation", message=exc.message)
        return 400, _local_error_payload(exc.message, exc.category, exc.details)
    if isinstance(exc, RequestValidationError):
        if is_challenge_request:
            return 422, _challenge_error_payload(error="validation", message="Request validation failed.")
        return 422, {
            "error": "validation_error",
            "message": "Request validation failed.",
            "details": {"errors": exc.errors()},
        }
    raise TypeError(f"Unsupported exception type: {type(exc)!r}")


def create_app() -> FastAPI:
    registry = EpisodeRegistry()
    engine = TruckPackingEngine()
    services = AppServices(
        registry=registry,
        engine=engine,
        episode_service=EpisodeService(registry, engine),
        preview_service=PreviewService(registry, engine),
    )
    app = FastAPI(
        title="Foresight Local Challenge",
        version="0.1.0",
        docs_url="/challenge/api/docs",
        openapi_url="/challenge/api/openapi.json",
    )
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
    async def handle_not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        status_code, payload = build_error_response(request.url.path, exc)
        return JSONResponse(status_code=status_code, content=payload)

    @app.exception_handler(ValidationError)
    async def handle_validation(request: Request, exc: ValidationError) -> JSONResponse:
        status_code, payload = build_error_response(request.url.path, exc)
        return JSONResponse(status_code=status_code, content=payload)

    @app.exception_handler(StateConflictError)
    async def handle_conflict(request: Request, exc: StateConflictError) -> JSONResponse:
        status_code, payload = build_error_response(request.url.path, exc)
        return JSONResponse(status_code=status_code, content=payload)

    @app.exception_handler(EngineError)
    async def handle_engine(request: Request, exc: EngineError) -> JSONResponse:
        status_code, payload = build_error_response(request.url.path, exc)
        return JSONResponse(status_code=status_code, content=payload)

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        status_code, payload = build_error_response(request.url.path, exc)
        return JSONResponse(status_code=status_code, content=payload)

    return app
