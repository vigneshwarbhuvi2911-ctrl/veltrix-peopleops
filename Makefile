test:
	python -m pytest tests

test-corpus:
	python -m pytest tests_corpus

install:
	pip install -r requirements-dev.txt

run:
	uvicorn app.main:get_app --factory --host 0.0.0.0 --port 8000

up:
	docker compose up --build

test-docker:
	docker compose run --rm --no-deps peopleops python -m pytest tests tests_corpus

smoke:
	python scripts/smoke_test.py

.PHONY: test test-corpus install run up test-docker smoke
