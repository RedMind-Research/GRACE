# Contributing to GRACE

Thank you for helping improve GRACE. The project keeps the reusable Evolution product separate from the Tau² Telecom research reproduction. Changes should preserve that boundary and remain verifiable without a model call.

## License and contributions

GRACE is licensed under the [Apache License 2.0](LICENSE). Unless you explicitly state otherwise, an intentional contribution submitted for inclusion is licensed under the same terms, as described in Section 5 of the license. Submit only material you have the right to contribute, preserve applicable notices, and identify any third-party provenance in the pull request. Citation metadata must remain aligned with the public arXiv record and `CITATION.cff`.

## Development environment

GRACE supports Python 3.10 through 3.13. Create an isolated environment and install the declared development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,gemini]'
```

For work that imports the pinned telecom benchmark rather than its offline adapters, use:

```bash
python -m pip install -e '.[dev,gemini]' -r requirements/reproduction.txt
```

The exact Tau² VCS requirement is deliberately repository-level in `requirements/reproduction.txt`; its editable form preserves upstream source-resident benchmark data and must never be copied into `redmind-grace` wheel metadata. The reproduction runner is executed from a repository checkout and is not installed as a second distribution.

Never commit `.env`, service-account files, provider keys, raw responses, raw trajectories, or generated run directories. Use synthetic or explicitly sanitized test data.

## Offline-first verification

Normal development and pull-request checks are zero-provider tests. Keep `GRACE_RUN_LIVE_TESTS=0`, do not export provider credentials into the test process, and run:

```bash
python -m compileall -q src/grace reproduction/tau2_telecom
python -m pytest -m 'not live'
ruff check .
ruff format --check .
```

Tests install a network-deny fixture. Do not bypass it or replace a scripted provider with LiteLLM merely to make a test pass. Live qualification requires a separate user authorization, reviewed manifest, route, and budget; a pull request does not grant that authorization.

## Type checking and coverage review

Type checking covers both the product and reproduction source trees:

```bash
python -m mypy --no-incremental \
  src/grace reproduction/tau2_telecom
```

Coverage review focuses on this deterministic critical set:

- `grace.schemas.loader` and `grace.schemas.models`;
- `grace.graph.models`, `validation`, `operations`, `repair`, and `diff`;
- `grace.evolution.reconstruction`.

Generate and inspect the report:

```bash
mkdir -p .release
python -m pytest -m 'not live' \
  --cov=grace.schemas.loader \
  --cov=grace.schemas.models \
  --cov=grace.graph.models \
  --cov=grace.graph.validation \
  --cov=grace.graph.operations \
  --cov=grace.graph.repair \
  --cov=grace.graph.diff \
  --cov=grace.evolution.reconstruction \
  --cov-branch --cov-report=term \
  --cov-report=json:.release/critical-coverage.json
```

Treat coverage as evidence rather than a substitute for behavioral assertions. A regression must be addressed with meaningful tests; do not add blanket exclusions, `type: ignore`, `noqa`, or lower a check to manufacture a green result.

## Distribution qualification

Build from a clean tree with the source commit timestamp and qualify from environments outside the source checkout:

```bash
export SOURCE_DATE_EPOCH="$(git log -1 --pretty=%ct)"
python -m pip install 'twine>=6,<7'
python -m build --outdir dist/core .
find dist/core -type f -name '*.tar.gz' -print0 \
  | xargs -0 python scripts/normalize_sdist.py --epoch "$SOURCE_DATE_EPOCH"
python -m twine check dist/core/*
python scripts/verify_distributions.py \
  --core-wheel "$(find dist/core -maxdepth 1 -name '*.whl' -print -quit)" \
  --core-sdist "$(find dist/core -maxdepth 1 -name '*.tar.gz' -print -quit)"
```

The normalizer removes build-time metadata variance from the source archive. The verifier applies explicit content allowlists to the wheel and source distribution and rejects reproduction code, Tau² dependencies, or a benchmark CLI in either artifact. CI installs both core artifact formats outside the source checkout and tests the repository-only reproduction separately. During `0.x`, compatible fixes increment the patch version, while new functionality or a breaking public or artifact contract increments the minor version. The release tag must match `src/grace/_version.py`; publish only final `X.Y.Z` versions unless an explicit external release-candidate phase is approved, and never publish `.dev` versions as releases.

When dependency inputs change, update and test the affected surface:

```bash
python -m pip install -r requirements/minimum.txt
python -m pip install -r requirements/gemini.txt
python -m pip install -r requirements/reproduction.txt
```

`requirements/minimum.txt` and `requirements/gemini.txt` exercise the lowest supported core/provider versions. `requirements/reproduction.txt` records the reviewed Tau² VCS revision for checkout-only research execution; it must never be copied into package metadata.

## Pull-request expectations

- Keep product APIs independent of tau2, internal workspace modules, and parent repository paths.
- Preserve full benchmark task IDs, collision-resistant UIDs, pinned revisions, exact manifest order, and fail-closed completeness checks in reproduction changes.
- Begin every new first-party Python file with the repository's complete Dan C. Hsu and Luke Lu / Apache 2.0 source header; do not apply it to third-party code, Tau²-derived data, or separately licensed paper assets.
- Add deterministic tests for success, malformed input, integrity failure, retry/resume, and privacy boundaries as applicable.
- Update user documentation when a public contract changes, without copying internal experiment notes or private artifacts.
- State which checks were run and list every unresolved release blocker.

Report suspected vulnerabilities according to `SECURITY.md`, never in a public pull request.
