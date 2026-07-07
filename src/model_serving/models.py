from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Predictor(Protocol):
    version: str

    def predict(self, features: list[float]) -> float: ...


@dataclass(frozen=True)
class LogisticModel:
    version: str
    weights: list[float]
    bias: float = 0.0
    metadata: dict[str, object] | None = None

    @classmethod
    def load(cls, path: str | Path) -> LogisticModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            version=str(payload["version"]),
            weights=[float(value) for value in payload["weights"]],
            bias=float(payload.get("bias", 0.0)),
            metadata=dict(payload.get("training", {})),
        )

    def predict(self, features: list[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError(f"expected {len(self.weights)} features")
        score = sum(weight * value for weight, value in zip(self.weights, features, strict=True))
        score += self.bias
        return 1 / (1 + math.exp(-score))


class FailingModel:
    version = "failing"

    def predict(self, features: list[float]) -> float:
        raise RuntimeError("simulated model failure")
