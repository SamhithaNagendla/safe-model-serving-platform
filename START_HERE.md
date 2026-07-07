# Start Here

1. `pip install -e ".[dev]"`
2. `pytest`
3. `cp .env.example .env`
4. `uvicorn model_serving.api:app --reload`
5. Route 20% to v2, submit predictions, label them, compare `/metrics`, then call `/rollback`.
