from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id TEXT PRIMARY KEY,
    routing_key TEXT NOT NULL,
    served_version TEXT NOT NULL,
    score REAL,
    shadow_version TEXT,
    shadow_score REAL,
    latency_ms REAL NOT NULL,
    error TEXT,
    fallback_used INTEGER NOT NULL DEFAULT 0,
    actual_label INTEGER
)
"""


class MetricsStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        with self._connect() as connection:
            connection.execute(SCHEMA)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def add(self, row: dict[str, object]) -> None:
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    request_id, routing_key, served_version, score, shadow_version,
                    shadow_score, latency_ms, error, fallback_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["request_id"],
                    row["routing_key"],
                    row["served_version"],
                    row.get("score"),
                    row.get("shadow_version"),
                    row.get("shadow_score"),
                    row["latency_ms"],
                    row.get("error"),
                    int(bool(row.get("fallback_used"))),
                ),
            )
            connection.commit()

    def update_shadow(self, request_id: str, version: str, score: float | None) -> None:
        with self.lock, self._connect() as connection:
            connection.execute(
                "UPDATE predictions SET shadow_version=?, shadow_score=? WHERE request_id=?",
                (version, score, request_id),
            )
            connection.commit()

    def add_label(self, request_id: str, actual_label: int) -> bool:
        with self.lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE predictions SET actual_label=? WHERE request_id=?",
                (actual_label, request_id),
            )
            connection.commit()
            return cursor.rowcount == 1

    def get(self, request_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM predictions WHERE request_id=?", (request_id,)
            ).fetchone()
        return dict(row) if row else None

    def summary(self) -> dict[str, dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT served_version,
                       COUNT(*) AS requests,
                       SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,
                       AVG(latency_ms) AS avg_latency_ms,
                       SUM(fallback_used) AS fallbacks,
                       SUM(CASE WHEN actual_label IS NOT NULL THEN 1 ELSE 0 END) AS labeled,
                       SUM(CASE WHEN actual_label IS NOT NULL AND
                           ((score >= 0.5 AND actual_label = 1) OR
                            (score < 0.5 AND actual_label = 0)) THEN 1 ELSE 0 END) AS correct
                FROM predictions
                GROUP BY served_version
                ORDER BY served_version
                """
            ).fetchall()
        result: dict[str, dict[str, object]] = {}
        for row in rows:
            labeled = int(row["labeled"] or 0)
            correct = int(row["correct"] or 0)
            result[str(row["served_version"])] = {
                "requests": int(row["requests"]),
                "errors": int(row["errors"] or 0),
                "avg_latency_ms": round(float(row["avg_latency_ms"] or 0.0), 3),
                "fallbacks": int(row["fallbacks"] or 0),
                "labeled": labeled,
                "accuracy": round(correct / labeled, 4) if labeled else None,
            }
        return result
