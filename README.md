# Safe Model Serving Platform

A FastAPI prediction service for champion/challenger deployment, deterministic A/B assignment,
non-blocking shadow execution, automatic challenger fallback, immediate rollback, delayed labels,
and persistent per-version operational metrics.

## Production-risk controls demonstrated

- stable assignment through SHA-256 hashing;
- traffic percentages from 0 to 100%;
- shadow predictions never determine the primary response;
- challenger failures fall back to the champion;
- champion and challenger artifacts are loaded from versioned JSON files;
- prediction history and labels persist in SQLite across service restarts;
- metrics report requests, errors, average latency, fallbacks, labeled volume, and accuracy;
- routing validation prevents unknown model versions.

The included artifact metadata records that both model versions originated from deterministic
logistic-regression training experiments rather than arbitrary hand-entered demo weights.

## Run



```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
pytest
uvicorn model_serving.api:app --reload
```

Use `/routing`, `/predict`, `/labels`, `/metrics`, and `/rollback` through the generated `/docs` UI.

## Current limitations

- SQLite is suitable for a single local service instance; production deployment would use a shared
  metrics database;
- routing configuration is in memory and should move to a durable control plane;
- this MVP demonstrates binary classification with two numerical features.


## Local Test Result

This project was tested locally on macOS using Python 3.11.

```bash
pytest

Test coverage: 95.91%



