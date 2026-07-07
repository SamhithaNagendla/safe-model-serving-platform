from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from model_serving.metrics import MetricsStore
from model_serving.models import LogisticModel, Predictor
from model_serving.routing import RoutingConfig, choose


class ModelService:
    def __init__(
        self,
        artifact_root: str | Path,
        metrics: MetricsStore,
        *,
        timeout_ms: int = 100,
        shadow_timeout_ms: int = 250,
        registry: dict[str, Predictor] | None = None,
    ):
        root = Path(artifact_root)
        self.registry = registry or {
            path.stem: LogisticModel.load(path) for path in root.glob("*.json")
        }
        if not self.registry:
            raise ValueError("no model artifacts found")
        versions = sorted(self.registry)
        self.config = RoutingConfig(
            champion=versions[0],
            challenger=versions[1] if len(versions) > 1 else versions[0],
        )
        self.metrics = metrics
        self.timeout_ms = timeout_ms
        self.shadow_timeout_ms = shadow_timeout_ms
        self.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="model")

    def update_routing(self, config: RoutingConfig) -> dict[str, object]:
        missing = {config.champion, config.challenger} - set(self.registry)
        if missing:
            raise ValueError(f"unknown model versions: {sorted(missing)}")
        self.config = config
        return config.as_dict()

    def _predict_with_timeout(self, model: Predictor, features: list[float], timeout_ms: int) -> float:
        return self.pool.submit(model.predict, features).result(timeout=timeout_ms / 1000)

    def _record_shadow(self, request_id: str, version: str, future: Future) -> None:
        try:
            score = future.result(timeout=self.shadow_timeout_ms / 1000)
        except Exception:
            score = None
        self.metrics.update_shadow(request_id, version, score)

    def predict(self, key: str, features: list[float]) -> dict[str, object]:
        assigned_version = choose(key, self.config)
        served_version = assigned_version
        request_id = str(uuid4())
        started = perf_counter()
        error: str | None = None
        fallback_used = False
        score: float | None = None

        try:
            score = self._predict_with_timeout(
                self.registry[assigned_version], features, self.timeout_ms
            )
        except (Exception, TimeoutError) as exc:
            error = str(exc)
            if assigned_version != self.config.champion:
                served_version = self.config.champion
                fallback_used = True
                try:
                    score = self._predict_with_timeout(
                        self.registry[served_version], features, self.timeout_ms
                    )
                    error = None
                except (Exception, TimeoutError) as champion_exc:
                    error = str(champion_exc)

        latency = (perf_counter() - started) * 1000
        shadow_version: str | None = None
        row = {
            "request_id": request_id,
            "routing_key": key,
            "assigned_version": assigned_version,
            "served_version": served_version,
            "score": score,
            "shadow_version": None,
            "shadow_score": None,
            "latency_ms": latency,
            "error": error,
            "fallback_used": fallback_used,
        }
        self.metrics.add(row)

        if self.config.shadow_enabled and served_version == self.config.champion:
            shadow_version = self.config.challenger
            future = self.pool.submit(self.registry[shadow_version].predict, features)
            self.pool.submit(self._record_shadow, request_id, shadow_version, future)
            row["shadow_version"] = shadow_version

        if error:
            raise RuntimeError(error)
        return row

    def rollback(self) -> dict[str, object]:
        self.config.challenger_percent = 0
        self.config.shadow_enabled = False
        return self.config.as_dict()

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
