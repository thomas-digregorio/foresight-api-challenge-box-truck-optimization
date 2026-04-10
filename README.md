# Foresight Local Truck Packing Challenge

This repository is a local approximation of the Dexterity Foresight API Challenge, not the official simulator.

Confirmed from Dexterity's public challenge docs on April 9, 2026:

- the challenge is a REST API at `https://dexterity.ai/challenge/api`
- boxes arrive one at a time and each placement response returns full state
- documented truck dimensions are `depth=2.0 m`, `width=2.6 m`, `height=2.75 m`
- public mode names are `dev` and `compete`
- Dexterity's hosted backend uses a physics simulation in `compete`, and no public simulator is provided

This project preserves the same raw action contract:

- `position: [x, y, z]`
- `orientation_wxyz: [w, x, y, z]`

The backend accepts non-zero quaternions at the API boundary, normalizes them, converts them into oriented box geometry, and applies deterministic geometry-only validity checks. That deterministic stability logic is a local approximation, not confirmed official physics behavior.

## Fidelity Boundaries

### Confirmed from Public Docs

- `POST /challenge/api/start`
- `POST /challenge/api/place`
- `GET /challenge/api/status/{game_id}`
- `POST /challenge/api/stop`
- `GET /challenge/api/health`
- truck coordinates are right-handed with origin at the front-bottom-left corner
- `dev` returns exact placed poses with no physics settling
- `compete` uses a backend physics simulation and is rate-limited on Dexterity's hosted service

### Local Approximation

- support and stability are deterministic and geometry-only
- no rigid-body settling, friction, collapse propagation, or post-placement displacement tracking
- local `compete` uses the confirmed public mode name, but it is still backed by the same deterministic local validator rather than Dexterity's physics backend
- box ID generation, queue generation, and seeded local starts are local conveniences

### Local-Only Extensions

- `/local/api/start`
- `/local/api/place`
- `/local/api/status/{game_id}`
- `/local/api/stop`
- `/local/api/preview`
- manual-play timeout fallback with `current_box_deadline`
- preview repair suggestions and latest-valid-preview tracking
- the `loading_guide_x` UI guide line at `0.78 m`

`/challenge/api/*` is intentionally thinner and closer to the public contract. Local-only preview, timeout, and debug behavior now stays under `/local/api/*`.

## Architecture

### Engine First

The pure engine lives under [backend/app/engine](/home/thomasdigregorio/code/foresight-api-challenge-box-truck-optimization/backend/app/engine) and owns:

- quaternion normalization and rotation handling
- oriented cuboid generation
- SAT collision checks
- floor/support-plane validation
- box queue progression
- density scoring
- feasibility probing and nearby action repair

No FastAPI, React, database, or persistence concerns live in the engine layer.

### Canonical vs Local Routes

Canonical challenge-like routes are exposed under `/challenge/api/*` and use public `dev` / `compete` mode names. They do not enforce the old `x = 0.78` loading-line rule, and they avoid returning local timeout/debug fields.

The manual-play UI uses `/local/api/*` so it can keep local-only features such as preview repair suggestions, timeout fallback, and the local loading guide without polluting the challenge-like contract.

### Quaternion Semantics

- orientation is always `[w, x, y, z]`
- quaternions are normalized before use
- the engine converts them to rotation matrices and world-space OBB corners
- the API and stored placements keep quaternions even though deterministic v1 stability is limited to gravity-compatible resting configurations

## Run It

The repo expects the conda environment named `foresight-challenge`.

### Backend

```bash
make backend-install
make backend-dev
```

Backend URL: `http://127.0.0.1:8000`

Swagger UI: `http://127.0.0.1:8000/challenge/api/docs`

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

### Example Payloads

Canonical challenge-like request examples live in [shared/api_examples/start.json](/home/thomasdigregorio/code/foresight-api-challenge-box-truck-optimization/shared/api_examples/start.json) and [shared/api_examples/place.json](/home/thomasdigregorio/code/foresight-api-challenge-box-truck-optimization/shared/api_examples/place.json).

Local-only manual-play start examples live in [shared/api_examples/local_start.json](/home/thomasdigregorio/code/foresight-api-challenge-box-truck-optimization/shared/api_examples/local_start.json).

## Current Limits

- Stability is deterministic and geometry-only. There is no local physics settling backend yet.
- The canonical challenge-like contract is conservative where public docs are incomplete. This repo does not claim undocumented official response fields or semantics.
- `/challenge/api/my-games` is not implemented locally because the public docs do not expose enough response detail to reproduce it faithfully without guessing.
- Local timeout fallback is intentionally not treated as official challenge behavior.

## Remaining Known Fidelity Gaps

- Dexterity's hosted `compete` mode performs backend physics settling; this repo does not.
- Public docs describe `unstable` termination for 3+ displaced boxes, but that requires physics/displacement tracking the local engine does not have.
- Some official response details remain unverified because no live API credentials or captured live responses were used in this audit.

See [docs/fidelity_audit.md](/home/thomasdigregorio/code/foresight-api-challenge-box-truck-optimization/docs/fidelity_audit.md) for the audit summary, fixes made, and remaining approximation gaps.
