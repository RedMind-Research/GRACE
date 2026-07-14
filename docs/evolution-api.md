# Evolution API

GRACE exposes a small product boundary: initialize a typed graph from an instruction, then evolve the graph and instruction from a diagnosis supplied by the caller.

```text
instruction + network schema
        │
        ▼
GraceEngine.initialize()
        │
        ▼
GraceState(G0, instruction0)
        │ + caller diagnosis report
        ▼
GraceEngine.evolve()
        │
        ▼
GraceState(G1, instruction1)
```

Trajectory collection, failure diagnosis, task execution, user simulation, and evaluation are intentionally outside the product API. An integrating team can keep its existing agent and trajectory-analysis system and use only the Evolution component.

## Inputs

Before the lifecycle begins, construct `GraceEngine` with a provider implementing `LLMProvider` and either the default schema or a caller-defined `NetworkSchema`. `GraceEngine.initialize()` then requires only the exact system-level instruction to maintain; the engine reuses its configured provider and schema.

`GraceConfig.max_output_tokens` controls the output-token ceiling forwarded to every P2G and Evolution request. Its released default is `65536`; integrations can select another positive integer without changing the provider protocol. See [Providers](providers.md) for Gemini and custom-provider contracts.

`GraceEngine.evolve` requires:

- one integrity-checked `GraceState`;
- a non-empty diagnosis report describing observed gaps and desired corrective behavior.

The diagnosis can be Markdown, structured prose, or a report rendered by the caller's own analyzer. GRACE does not require the caller to adopt the telecom diagnosis procedure.

## A diagnosis prompt is not automatically a rubric

A system prompt that defines how to read a graph, apply operations, and obey a network schema is an operational specification. It can contain validation criteria without being a scoring rubric.

A rubric normally adds an explicit judgment contract: named criteria, observable evidence, score levels or pass/fail thresholds, and rules for combining those judgments. GRACE's internal prompts constrain graph use and typed outputs; they are not presented as numerical evaluation rubrics. A team may separately use a rubric to produce its diagnosis report, but that is an upstream integration choice rather than an Evolution API requirement.

## Results and failure semantics

Initialization returns `InitializationResult`; Evolution returns `EvolutionResult`. Both include the accepted state, validation telemetry, provider usage and attempt records, warnings, and an artifact path when persistence is enabled. Evolution also includes the graph change log, structural-validation scope, reconstruction report, and input provenance.

Fatal provider, schema, state-integrity, and artifact conditions raise public GRACE exception subclasses and do not return an accepted state. Invalid call arguments such as an empty instruction or diagnosis raise `ValueError`. Neither category returns an accepted state. Examples include:

- a state whose graph, instruction, schema, or lineage hashes do not match;
- a graph that remains invalid under the active schema;
- exhausted provider transport or response-validation attempts;
- artifact integrity or persistence failure.

Non-fatal fallbacks remain visible as `completed_with_warnings`; rejected graph operations, deterministic forced drops, structural-analysis caps, and PATCH fallbacks are never silently converted into an ordinary success.

## Persistence and resume

Checkpoint writing is enabled by default because a long-running Evolution process must be able to prove which graph, instruction, schema, configuration, prompts, model route, and parent produced each accepted state.

| Mode | Intended use | Persisted evidence |
| --- | --- | --- |
| `checkpoint` | Normal product integration | Reloadable state, schema snapshot, compact reports, prompt hashes, model/usage/attempt telemetry, manifest, and file hashes. |
| `audit` | Research and release qualification | Everything in checkpoint plus exact prompt snapshots, full validation history, and a validated audit-payload envelope. |
| `none` | Caller-managed ephemeral execution | No files. Returned states remain typed, but GRACE cannot verify lineage across processes. |

Accepted state directories are immutable and published atomically. Loading a state verifies the allowlisted file set, file hashes, schema snapshot, graph validity, state identity, and its embedded parent ID and step rules; the existence of an output file alone is never sufficient for resume. Supply the actual parent state, or call `load_parent()`, when a reader must also verify the one-step link to the persisted parent checkpoint.

Resume with the same artifact root and `run_id`. Load the last accepted state before calling Evolution again:

```python
from grace import DefaultGraceSchema, GraceEngine
from grace.providers import LiteLLMProvider, VertexAIADCConfig

route = VertexAIADCConfig(
    project="your-google-cloud-project",
    location="us-central1",
)
provider = LiteLLMProvider(route)
engine = GraceEngine(
    provider=provider,
    schema=DefaultGraceSchema(),
    artifact_dir="./grace_runs",
    artifact_mode="checkpoint",
    run_id=previous_run_id,
)

parent = engine.artifacts.load_parent(previous_state_id, expected_schema=engine.schema)
state = engine.artifacts.load_state(
    previous_state_id,
    expected_schema=engine.schema,
    parent_state=parent,
)
updated = engine.evolve(
    state=state,
    diagnosis_report=diagnosis_report,
)
print(updated.artifact_location)
```

With checkpoint or audit persistence enabled, the engine verifies before spending a provider call that the supplied input state exists and matches its persisted record in the active run. Each accepted artifact records the configuration hash and model provenance used for that step. A caller may deliberately use a different engine configuration or provider route for the next step; that child records new provenance rather than treating the change itself as an invalid resume. For a custom ontology, the verified record also exposes its canonical schema snapshot so a resumed process can reconstruct the intended schema rather than guess from an identifier.

## Artifact safety boundary

Checkpoint mode stores the hashes of the exact system and user prompts sent by each logical call; audit mode additionally stores the prompt snapshots. Engine-generated artifacts do not intentionally include credentials, hidden reasoning, authorization headers, raw SDK objects, provider exception text, or opaque provider payloads. `ArtifactStore` enforces JSON-safe values, rejects forbidden credential or reasoning field names, scans credential-shaped text and caller-supplied canaries, and rejects raw-output fields by default. Raw model outputs require an explicit `ArtifactStore(allow_raw_model_outputs=True)` opt-in for an approved audit use case; the other checks still apply.

No generic filter can infer the meaning of every caller-owned string. Callers that write an `audit_payload` must therefore use an application allowlist and must not submit provider exception text or opaque upstream payloads merely under neutral field names.

Provider configuration resolves credentials at runtime. Use environment variables or a secret manager for real credentials and review application-owned diagnosis text before enabling audit persistence. Callers can also provide runtime canaries through `ArtifactStore.sensitive_values`; publication fails closed if one appears in a candidate artifact.

## Deliberate no-update steps

The product method rejects an empty diagnosis instead of guessing whether a batch means “no update.” Integrations that do not require a fixed checkpoint sequence can simply omit an Evolution call. A fixed-sequence workflow can call `engine.advance_no_update(state=state, diagnosis_report=report)` to persist an integrity-bound child with unchanged graph and instruction, incremented lineage, and zero provider calls; the telecom reproduction uses this route.
