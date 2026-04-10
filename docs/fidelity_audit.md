# Fidelity Audit

Audit date: April 9, 2026

Live API verification performed: No

No Dexterity API credentials or checked-in captured live responses were available in this repository or environment during this audit, so this pass used:

- Dexterity's public challenge page at `https://dexterity.ai/challenge`
- checked-in local examples under `shared/api_examples`
- local repository code and tests

## Confirmed Official Facts

Confirmed from Dexterity's public challenge page on April 9, 2026:

- Base URL is `https://dexterity.ai/challenge/api`
- documented routes include `/start`, `/place`, `/status/{game_id}`, `/stop`, `/my-games`, and `/health`
- the challenge is one-box-at-a-time and each placement returns full state
- public mode names are `dev` and `compete`
- documented truck dimensions are `depth=2.0 m`, `width=2.6 m`, `height=2.75 m`
- public docs state `dev` has no physics simulation and `compete` uses full backend physics simulation
- public docs describe canonical termination states as `game_status: in_progress|completed`
- public docs describe `termination_reason` values including `unstable` and `player_stop`
- public docs show canonical error codes such as `invalid_game_id`, `invalid_box_id`, and `validation`

## Mismatches Found

### High Priority

- The engine enforced `loading_line_x = 0.78` during canonical placement validation and feasibility search.
- That limit contradicted the documented `2.0 m` truck depth and would materially distort local training and evaluation.

### Canonical Contract Leaks

- `/challenge/api/*` exposed local-only fields such as `created_at`, `current_box_started_at`, `current_box_deadline`, and `timeout_seconds`.
- canonical placed-box serialization included `weight`, which is not shown in Dexterity's public placed-box examples
- canonical responses used local-only game states such as `timed_out` and `no_feasible_placement`
- the repo exposed the local mode name `compete_stub` instead of the public `compete`

### Local Convenience Behavior Presented Too Canonically

- manual UI timeout fallback could terminate challenge-like games as if it were official behavior
- local no-feasible-placement auto-termination could end challenge-like games even though that behavior is not publicly documented
- the `0.78 m` loading line appeared as if it were a canonical truck constraint rather than a local guide

### Documentation Drift

- `README.md` described the repo as mirroring the official API too strongly without clearly separating local approximation from confirmed official behavior
- `system_design.txt` still encoded older assumptions such as `compete_stub` and challenge-route timeout behavior
- `shared/api_examples/start.json` used a local seeded start payload even though seed is a local extension rather than a confirmed challenge field

## Fixes Made

- Removed the `0.78 m` loading-line constraint from canonical engine validation and feasibility search.
- Renamed external mode handling from `compete_stub` to the public `compete`.
- Split route families:
  - `/challenge/api/*` now uses challenge-like serializers with a thinner public-contract surface.
  - `/local/api/*` now carries manual-play helpers, timeout metadata, preview repair suggestions, and the local loading guide.
- Disabled local timeout and no-feasible-placement auto-termination for challenge-route episodes.
- Added route-family isolation so local game IDs are not silently treated as challenge-route sessions and vice versa.
- Updated canonical error handling toward public error codes:
  - `invalid_game_id`
  - `invalid_box_id`
  - `validation`
- Added local-only `loading_guide_x` so the existing UI can keep the visual guide without affecting canonical validation.
- Updated the frontend to use `/local/api/*` for manual play.
- Updated shared examples so canonical examples no longer rely on local seeded start behavior.

## Remaining Approximation Gaps

- Local `compete` still does not reproduce Dexterity's hosted physics simulation.
- The local engine does not model post-placement settling, displacement, or the public `unstable` termination rule.
- `/challenge/api/my-games` is not implemented because the public docs do not provide enough response detail to reproduce it faithfully without guessing.
- The exact official `/stop` response schema was not verified from a live response, so the local challenge-like response remains best-effort.
- Deterministic support validation is still a local approximation of the hosted backend's physics-consistent placement behavior.

## Notes For Future Verification

If Dexterity API credentials become available later, use a minimal number of live requests to confirm:

- the exact `/stop` response payload
- whether `/status/{game_id}` includes `termination_reason`
- whether any additional documented error codes or response fields are present in the live OpenAPI schema
