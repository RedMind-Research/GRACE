# Security policy

## Supported versions

The `redmind-grace` version is sourced from `grace._version`. After the first public release, security fixes target the latest `0.1.x` patch and the default branch. The checkout-only reproduction is identified by its exact Git revision and embedded protocol identities rather than a second package version. This policy is not a response-time or remediation SLA.

| Version                            | Supported                  |
| ---------------------------------- | -------------------------- |
| Latest published `0.1.x` patch     | Yes                        |
| Unreleased default branch          | Best effort                |
| Development and untagged snapshots | No compatibility guarantee |

The table must be updated when a newer release line becomes supported.

## Reporting a vulnerability

Do not disclose a suspected vulnerability, credential, private trajectory, or provider response in a public issue, discussion, pull request, or test fixture.

Use this repository's GitHub **Security → Report a vulnerability** form. The repository owner must enable GitHub private vulnerability reporting before the repository is made public. If that form is unavailable, open a public issue that requests a private reporting channel without including any technical or sensitive detail. Do not send secrets as proof of impact.

A useful private report includes:

- the affected GRACE version or commit;
- whether the issue affects the product API, artifact store, reproduction harness, provider boundary, or build/release process;
- minimal reproduction steps using synthetic data;
- expected and observed security boundaries;
- impact and any known mitigations;
- whether the issue has been disclosed elsewhere.

The maintainers do not currently promise a fixed acknowledgement or remediation time. They will coordinate validation, remediation, release, and disclosure privately, and will credit reporters when requested and safe.

## Security boundaries

- GRACE does not auto-load `.env` files. Provider credentials must come from an explicit environment or workload identity selected by the operator.
- Offline tests and qualification must not receive live provider credentials or make provider/network calls.
- Checkpoint and audit artifacts must never contain API keys, authentication material, raw provider SDK objects, hidden chain-of-thought fields, or unreviewed private trajectories.
- The Tau² reproduction is optional, pinned to its reviewed upstream commit, excluded from the installed product distribution, and available only from a repository checkout.
- Synthetic tests and task-selection manifests are publication artifacts. Raw trajectories remain outside the repository unless a separate privacy review explicitly approves them.

If a local run may have persisted sensitive content, stop sharing the artifact, revoke exposed credentials, preserve only access-controlled forensic evidence, and report through the private channel.

## Supply-chain checks

The security workflow performs CodeQL analysis and Python dependency auditing. Equivalent dependency-audit commands, starting from the repository root, are:

```bash
python -m venv .security-venv
.security-venv/bin/python -m pip install --upgrade pip
.security-venv/bin/python -m pip install \
  -r requirements/reproduction.txt \
  -e '.[gemini]' \
  'pip-audit>=2.9,<3'
.security-venv/bin/python -m pip_audit --progress-spinner=off
```

The pinned Tau² revision remains a repository-level VCS requirement. The security workflow includes it in dependency auditing, while the CI adapter job separately verifies the checkout-only benchmark integration.

## Legal and citation metadata

GRACE is licensed under the Apache License 2.0. Citation metadata is published in `CITATION.cff` and must remain aligned with the public arXiv record. Security reports are welcome regardless of whether the reporter uses the software.
