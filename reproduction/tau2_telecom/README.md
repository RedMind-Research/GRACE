# Reproducing the Tau² Telecom procedure

The repository freezes the procedure, not model outputs: task selection, benchmark revision, model roles, the seed, prompts, method settings, ordering, and metric definitions. Reproducers generate their own trajectories through the provider APIs.

The reproduction runner is source-only: it is versioned in this repository but deliberately excluded from the `redmind-grace` wheel and source distribution. Run it from the root of the matching tagged checkout.

## Checkout and environment

The released recipe pairs the `v0.1.0` repository state with the exact `redmind-grace==0.1.0` core and the Tau² revision locked in `requirements/reproduction.txt`:

```bash
git clone --branch v0.1.0 --depth 1 https://github.com/RedMind-Research/GRACE.git
cd GRACE
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'redmind-grace[gemini]==0.1.0'
python -m pip install -r requirements/reproduction.txt
```

The Tau² requirement is an editable VCS install because the pinned upstream revision keeps benchmark data in its source tree rather than its wheel. The requirement file also pins the model-facing LiteLLM, OpenAI, and Google SDK versions used by the qualified path. The VCS install record and clean source commit are checked before Tau² code is imported; an unverifiable or different benchmark revision fails closed. Do not replace this requirement with a non-editable Tau² wheel. Contributors testing modified core code may replace the pinned core command with `python -m pip install -e '.[dev,gemini]'`, but that is a development checkout rather than the frozen release pairing.

For Vertex AI, configure application-default credentials and the user-simulator key without writing either into the repository:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
export GRACE_VERTEX_PROJECT=YOUR_PROJECT_ID
export GRACE_VERTEX_LOCATION=us-central1
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

See [Providers](../../docs/providers.md) for Google AI Studio and route-specific authentication.

## Controlled shift

| Split               |           MMS |   Mobile Data |   Service | Total |
| ------------------- | ------------: | ------------: | --------: | ----: |
| Held-out evaluation | 30 (10/10/10) | 30 (10/10/10) | 6 (2/2/2) |    66 |
| Experience phase A  | 30 (10/10/10) |     9 (3/3/3) | 3 (1/1/1) |    42 |
| Experience phase B  |     9 (3/3/3) | 30 (10/10/10) | 3 (1/1/1) |    42 |

Parentheses give Easy/Hard/None persona counts. Ten disjoint experience batches follow `AABBAABBAA`; evaluation tasks are held out from every update.

## One-command workflow

Validate the frozen selection and inspect the exact episode count without provider credentials:

```bash
python -m reproduction.tau2_telecom validate-split
python -m reproduction.tau2_telecom plan --evaluation in-loop:all
```

This plan contains 420 experience episodes and 2,178 evaluation episodes, for 2,598 total. An `offline:0,6,8,10` plan contains 420 experience and 792 evaluation episodes, for 1,212 total. Then configure providers and run:

```bash
python -m reproduction.tau2_telecom run-all \
  --method grace \
  --evaluation in-loop:all
```

The terminal reports `initialize`, `experience`, `diagnosis`, `evolution`, and `evaluation` stages. For `grace` and `grace-no-sa`, checkpoint `0` is the Prompt-to-Graph state `ℓ₀`; for `hce`, checkpoint `0` retains the original flat-text instruction and does not call Prompt-to-Graph. Update `t=1..10` uses the experience batch `B(t-1)` and publishes the method-specific checkpoint `ℓ_t`.

`--evaluation` controls two independent dimensions in one value:

- timing: `in-loop` or `offline`;
- checkpoint selection: `all` or increasing indices such as `0,6,8,10`.

Examples are `in-loop:all`, `in-loop:0,6,8,10`, `offline:all`, and `offline:0,6,8,10`. Offline evaluation runs only after all ten updates; it is still part of the same `run-all` command.

Use `--method grace-no-sa` for the paper's GRACE without structural analysis condition. This changes only the effective `structural_analysis` setting; P2G, task selection, diagnosis, Evolution, models, seed, and evaluation remain aligned with GRACE. Use `--method hce` for the flat-text HCE comparison, which keeps the original instruction at checkpoint `0` and uses its own prompt-evolution function after a complete diagnosis. Every method participates in the run fingerprint, preventing cross-method artifact reuse.

Detailed stage commands are also available:

```bash
python -m reproduction.tau2_telecom initialize --method grace --evaluation in-loop:all
python -m reproduction.tau2_telecom run-batch --update 1 --method grace --evaluation in-loop:all
python -m reproduction.tau2_telecom diagnose --update 1 --method grace --evaluation in-loop:all
python -m reproduction.tau2_telecom evolve --update 1 --method grace --evaluation in-loop:all
python -m reproduction.tau2_telecom evaluate --checkpoint 1 --method grace --evaluation in-loop:all
python -m reproduction.tau2_telecom status --method grace --evaluation in-loop:all
```

## Exact execution contract

1. Install Tau² at the revision locked in `requirements/reproduction.txt` and declared in `reproduction/tau2_telecom/configs/reproduction.yaml`.
2. Load task definitions from that verified dependency and select the IDs in `reproduction/tau2_telecom/data/splits.json` in declared order.
3. Execute each 42-task experience batch sequentially with the current instruction checkpoint.
4. Classify all 42 expected trajectory records deterministically as a successful control, an eligible behavioral failure, an infrastructure error, or invalid input. Reflect only on eligible behavioral failures in canonical task order; stop without synthesis when a record is missing, duplicated, malformed, infrastructure-failed, or lacks its required reflection.
5. Apply the method-specific update exactly once. For `grace` and `grace-no-sa`, a complete diagnosis calls `GraceEngine.evolve`; a `no_update` diagnosis calls `GraceEngine.advance_no_update`, publishing an unchanged integrity-bound child with zero provider calls. For `hce`, a complete diagnosis calls `evolve_hce`; a `no_update` diagnosis skips that provider call and publishes a child checkpoint containing the unchanged flat-text instruction.
6. At reported checkpoints, run every one of the 66 held-out tasks for trials 0, 1, and 2.
7. Require the exact 198-cell matrix, then compute metrics.

No parallel path is part of the public protocol. A failed or interrupted run may resume only when task, trial, task definition, prompt, model, configuration, checkpoint, protocol, installed core version, source identities, Python runtime, and selected execution-stack dependency versions still match. Each `run.json` records the `redmind-grace` version, SHA-256 digests of the actually imported core package and behavior-bearing repository reproduction files (including the pinned reproduction requirement), a runtime fingerprint, plus the Git commit and behavior-relevant dirty state when the checkout has Git metadata. The `status` command reports the provenance persisted at run creation; a stage command performs the current-environment compatibility check before resuming.

## Metrics

- `pass@1`: mean success across all task/trial episodes.
- `pass@k`: fraction of tasks with at least one successful trial.
- `pass^k`: fraction of tasks successful in every trial.

`compute_evaluation_metrics` rejects missing, unknown or extra, duplicate, wrong-task, and wrong-trial inputs. Observation order does not affect the metric. Formal evaluation always uses all 66 tasks and trials 0, 1, and 2.

## Model routes

The paper uses Gemini 2.5 Flash for the task agent, diagnosis, and Evolution, through Vertex AI in this recipe. GPT-4.1 is the user-simulator family; the public config pins `gpt-4.1-2025-04-14` so future executions do not follow a moving alias. The deterministic tau2 evaluator does not use a judge LLM. Google lists October 16, 2026 as the scheduled retirement date for `gemini-2.5-flash`; verify the official [model lifecycle documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) before a future run and configure any replacement explicitly.

See the executable contracts in `reproduction/tau2_telecom/harness` and their fully synthetic tests in `tests/reproduction`.
