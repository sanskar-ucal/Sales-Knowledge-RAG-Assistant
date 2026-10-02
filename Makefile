.PHONY: install run test lint format eval eval-agent up down

install:
	uv sync

run:
	uv run uvicorn app.main:app --reload --port 8000

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

format:
	uv run ruff format . && uv run ruff check --fix .

eval:
	uv run python -m evaluation.run_eval --mode rag

eval-agent:
	uv run python -m evaluation.run_eval --mode agent --judge

up:
	docker compose up --build -d

down:
	docker compose down
