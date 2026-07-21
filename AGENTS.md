# GRACE Agent Guide

## Read First

- `README.md` for product and reproduction entry points.
- `CONTRIBUTING.md` for development, testing, release, and pull-request rules.
- `SECURITY.md` for vulnerability reporting.
- `docs/` for product integration, API, schema, and provider behavior.

## Repository Boundaries

- `src/grace/` is the reusable distributed product.
- `reproduction/tau2_telecom/` is checkout-only paper reproduction code; do not add it or Tau2 dependencies to package metadata.
- Preserve benchmark task IDs, pinned revisions, manifest order, and fail-closed provenance behavior unless the task explicitly changes the research protocol.

## Safe Development

- Default to deterministic offline work: set `GRACE_RUN_LIVE_TESTS=0` and do not pass provider credentials to tests.
- Never commit `.env`, service-account files, provider keys, raw provider responses, trajectories, or generated artifact directories.
- Use synthetic or sanitized test data and do not expose private instructions or diagnoses in documentation or logs.

## Validation and Handoff

- Add deterministic tests for behavior changes and run focused tests first.
- Before a pull request, run the relevant offline tests plus the checks in `CONTRIBUTING.md` that match the change.
- State changed files, commands run, and any unresolved release blocker in the pull-request description.
