from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256


@dataclass
class RoutingConfig:
    champion: str = "v1"
    challenger: str = "v2"
    challenger_percent: int = 10
    shadow_enabled: bool = True

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def bucket(key: str) -> int:
    return int(sha256(key.encode()).hexdigest()[:8], 16) % 100


def choose(key: str, config: RoutingConfig) -> str:
    return (
        config.challenger
        if bucket(key) < config.challenger_percent
        else config.champion
    )
