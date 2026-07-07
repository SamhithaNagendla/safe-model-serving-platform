.PHONY: install test lint run docker-up docker-down
install:
	pip install -e ".[dev]"
test:
	pytest --cov=model_serving --cov-report=term-missing
lint:
	ruff check .
run:
	uvicorn model_serving.api:app --reload
docker-up:
	docker compose up --build
docker-down:
	docker compose down -v
