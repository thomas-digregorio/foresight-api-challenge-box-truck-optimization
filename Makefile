CONDA_ENV=foresight-challenge
PYTHON=conda run -n $(CONDA_ENV) python
PIP=conda run -n $(CONDA_ENV) python -m pip

.PHONY: install backend-install frontend-install backend-dev frontend-dev test-backend test-frontend test build-frontend

install: backend-install frontend-install

backend-install:
	$(PIP) install -e ./backend[dev]

frontend-install:
	cd frontend && npm install

backend-dev:
	$(PYTHON) -m uvicorn main:app --app-dir backend --reload --host 0.0.0.0 --port 8000

frontend-dev:
	cd frontend && npm run dev

test-backend:
	$(PYTHON) -m pytest backend/app/tests

test-frontend:
	cd frontend && npm test

test: test-backend test-frontend

build-frontend:
	cd frontend && npm run build

