from pathlib import Path
from time import sleep

from model_serving.metrics import MetricsStore
from model_serving.models import FailingModel, LogisticModel
from model_serving.routing import RoutingConfig, bucket, choose
from model_serving.service import ModelService


def test_assignment_is_stable() -> None:
    assert bucket("user-1") == bucket("user-1")


def test_zero_and_hundred_percent() -> None:
    assert choose("x", RoutingConfig(challenger_percent=0)) == "v1"
    assert choose("x", RoutingConfig(challenger_percent=100)) == "v2"


def test_distribution_is_close() -> None:
    config = RoutingConfig(challenger_percent=20)
    count = sum(choose(str(index), config) == "v2" for index in range(5000))
    assert 850 < count < 1150


def test_shadow_does_not_change_primary(service) -> None:
    service.update_routing(RoutingConfig(challenger_percent=0, shadow_enabled=True))
    result = service.predict("u", [1, 2])
    assert result["served_version"] == "v1"
    sleep(0.05)
    stored = service.metrics.get(result["request_id"])
    assert stored["shadow_version"] == "v2"
    assert stored["shadow_score"] is not None


def test_challenger_failure_falls_back(tmp_path: Path) -> None:
    registry = {
        "v1": LogisticModel("v1", [0.5, -0.2]),
        "v2": FailingModel(),
    }
    service = ModelService(
        tmp_path,
        MetricsStore(tmp_path / "metrics.db"),
        registry=registry,
    )
    service.update_routing(
        RoutingConfig(champion="v1", challenger="v2", challenger_percent=100, shadow_enabled=False)
    )
    try:
        result = service.predict("u", [1, 2])
        assert result["served_version"] == "v1"
        assert result["fallback_used"] is True
    finally:
        service.close()


def test_metrics_survive_restart(tmp_path: Path) -> None:
    path = tmp_path / "metrics.db"
    first = MetricsStore(path)
    first.add(
        {
            "request_id": "r1",
            "routing_key": "u",
            "served_version": "v1",
            "score": 0.8,
            "latency_ms": 4.0,
            "error": None,
            "fallback_used": False,
        }
    )
    assert first.add_label("r1", 1)
    second = MetricsStore(path)
    assert second.get("r1")["score"] == 0.8
    assert second.summary()["v1"]["accuracy"] == 1.0
