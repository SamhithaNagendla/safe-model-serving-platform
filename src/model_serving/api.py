from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_serving.config import get_settings
from model_serving.metrics import MetricsStore
from model_serving.routing import RoutingConfig
from model_serving.service import ModelService


class PredictRequest(BaseModel):
    routing_key: str = Field(min_length=1, max_length=128)
    features: list[float] = Field(min_length=2, max_length=2)


class ConfigRequest(BaseModel):
    champion: str = "v1"
    challenger: str = "v2"
    challenger_percent: int = Field(ge=0, le=100)
    shadow_enabled: bool = True


class LabelRequest(BaseModel):
    request_id: str
    actual_label: int = Field(ge=0, le=1)


@lru_cache
def get_service() -> ModelService:
    settings = get_settings()
    return ModelService(
        settings.artifact_root,
        MetricsStore(settings.metrics_db_path),
        timeout_ms=settings.prediction_timeout_ms,
        shadow_timeout_ms=settings.shadow_timeout_ms,
    )


ServiceDependency = Annotated[ModelService, Depends(get_service)]


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        get_service().close()

    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "healthy", "version": settings.app_version}

    @app.post("/predict", tags=["prediction"])
    def predict(request: PredictRequest, service: ServiceDependency) -> dict[str, object]:
        try:
            return service.predict(request.routing_key, request.features)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.put("/routing", tags=["deployment"])
    def routing(request: ConfigRequest, service: ServiceDependency) -> dict[str, object]:
        try:
            return service.update_routing(RoutingConfig(**request.model_dump()))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/rollback", tags=["deployment"])
    def rollback(service: ServiceDependency) -> dict[str, object]:
        return service.rollback()

    @app.post("/labels", tags=["evaluation"])
    def labels(request: LabelRequest, service: ServiceDependency) -> dict[str, object]:
        if not service.metrics.add_label(request.request_id, request.actual_label):
            raise HTTPException(status_code=404, detail="prediction not found")
        return {"updated": True, **request.model_dump()}

    @app.get("/predictions/{request_id}", tags=["evaluation"])
    def prediction(request_id: str, service: ServiceDependency) -> dict[str, object]:
        row = service.metrics.get(request_id)
        if row is None:
            raise HTTPException(status_code=404, detail="prediction not found")
        return row

    @app.get("/metrics", tags=["operations"])
    def metrics(service: ServiceDependency) -> dict[str, dict[str, object]]:
        return service.metrics.summary()

    return app


app = create_app()
