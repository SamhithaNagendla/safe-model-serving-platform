from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Safe Model Serving Platform"
    app_version: str = "0.2.0"
    artifact_root: Path = Path("artifacts")
    metrics_db_path: Path = Path("data/metrics.db")
    prediction_timeout_ms: int = 100
    shadow_timeout_ms: int = 250

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="MODEL_SERVING_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def prepare(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.metrics_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.prepare()
    return settings
