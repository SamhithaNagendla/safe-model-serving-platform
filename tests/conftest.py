from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from model_serving.api import create_app, get_service
from model_serving.metrics import MetricsStore
from model_serving.models import LogisticModel
from model_serving.service import ModelService


@pytest.fixture
def registry():
    return {
        "v1": LogisticModel("v1", [0.5, -0.2], 0),
        "v2": LogisticModel("v2", [0.7, -0.1], -0.1),
    }


@pytest.fixture
def service(tmp_path: Path, registry):
    instance = ModelService(
        tmp_path,
        MetricsStore(tmp_path / "metrics.db"),
        registry=registry,
        timeout_ms=100,
    )
    yield instance
    instance.close()


@pytest.fixture
def api_client(service):
    app = create_app()
    app.dependency_overrides[get_service] = lambda: service
    with TestClient(app) as client:
        yield client
