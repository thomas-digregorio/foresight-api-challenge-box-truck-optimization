# Foresight Local Truck Packing Challenge

Engine-first local truck-packing environment that mirrors the Dexterity Foresight API Challenge closely enough to swap a local client for the remote API later with minimal rewrite.

The project is built around the same raw action contract as the challenge:

- `position: [x, y, z]`
- `orientation_wxyz: [w, x, y, z]`

The backend accepts arbitrary non-zero quaternions at the API boundary, normalizes them, converts them into oriented box geometry, and then applies deterministic v1 validity rules that only allow gravity-compatible resting poses with effectively horizontal support faces.

## What’s Included

- Deterministic geometry engine with OBB corner generation, truck-bounds checks, SAT-based OBB collision, and 90% support-area validation.
- In-memory episode registry and timeout logic with `latest_preview_action` / `latest_valid_preview_action` tracking.
- FastAPI adapter with official-like `/challenge/api/*` routes and local-only `/local/api/preview`.
- React + Vite + React Three Fiber manual-play UI styled after the provided mockup.
- RL-ready `RawEpisodeEnv`, `ActionRepairPolicy`, and `ParallelEpisodeManager`.
- Backend pytest coverage for geometry, support, timeout, repair, API validation, and parallel episode handling.

## Architecture

### Engine First

The pure engine lives under [backend/app/engine](/home/thomasdigregorio/code/foresight-api-challenge-box-truck-optimization/backend/app/engine) and owns:

- quaternion normalization and rotation handling
- oriented cuboid generation
- SAT collision checks
- floor/support-plane validation
- box queue progression
- timeout auto-placement
- density scoring
- feasibility probing and nearby action repair

No FastAPI, React, database, or persistence concerns live in the engine layer.

### Raw API Shape Preserved

The API adapter keeps the observation and action payloads close to the challenge contract on purpose:

- variable-length `placed_boxes`
- raw `current_box`
- `boxes_remaining`
- `density`
- `game_status`
- `termination_reason`

Strict Gymnasium encoding is intentionally deferred. The repo already includes [raw_env.py](/home/thomasdigregorio/code/foresight-api-challenge-box-truck-optimization/backend/app/rl/raw_env.py), but a future fixed-shape wrapper should sit on top of the raw engine observation instead of distorting the canonical state now.

### Quaternion Semantics

- Orientation is always `[w, x, y, z]`.
- Quaternions are normalized before use.
- The engine converts them to rotation matrices and world-space OBB corners.
- The API and stored placements keep quaternions even though deterministic v1 stability is limited to gravity-compatible resting configurations.

## Timeout Fallback

Each box gets 10 seconds. The frontend continuously sends ghost-box previews to `/local/api/preview`.

The backend stores:

- `current_box_started_at`
- `current_box_deadline`
- `latest_preview_action`
- `latest_valid_preview_action`

When a timeout is detected on a state read or write:

- if `latest_valid_preview_action` exists for the active box, it is auto-committed
- otherwise the episode ends with `game_status = "timed_out"` and `termination_reason = "timeout_no_valid_preview"`

## Local API

Challenge-like routes:

- `POST /challenge/api/start`
- `POST /challenge/api/place`
- `GET /challenge/api/status/{game_id}`
- `POST /challenge/api/stop`
- `GET /challenge/api/health`

Local-only route:

- `POST /local/api/preview`

Example payloads live in [shared/api_examples/start.json](/home/thomasdigregorio/code/foresight-api-challenge-box-truck-optimization/shared/api_examples/start.json) and [shared/api_examples/place.json](/home/thomasdigregorio/code/foresight-api-challenge-box-truck-optimization/shared/api_examples/place.json).

## Run It

The repo expects the conda environment named `foresight-challenge`.

### Backend

```bash
make backend-install
make backend-dev
```

Backend URL: `http://127.0.0.1:8000`

### Frontend

```bash
make frontend-install
make frontend-dev
```

Frontend URL: `http://127.0.0.1:5173`

### Tests

```bash
make test-backend
make test-frontend
```

### Production Frontend Build

```bash
make build-frontend
```

## Current Limits

- Stability is deterministic and geometry-only. There is no settling, friction, or rigid-body simulation yet.
- The feasibility search and RL repair routines are coarse deterministic searches, not optimized packers.
- Episodes are in-memory only. There is no replay storage, analytics pipeline, or database.

## Roadmap

- persistence, replay storage, and optional analytics
- strict Gymnasium tensor wrapper on top of the raw observation
- remote Dexterity API client swap-in
- physics-backed settling/stability backend
- heuristic, search-based, and RL packing agents
