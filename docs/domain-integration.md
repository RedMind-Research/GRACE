# Apply GRACE to another domain

GRACE evolves one system-level instruction. It does not require a particular agent framework, task runner, trajectory format, or diagnosis model. Configure an `LLMProvider` and a network schema when constructing `GraceEngine`; then supply an instruction during initialization and a caller-produced diagnosis report for each Evolution step.

The provider and schema are engine-level configuration, not fields that callers repeat on every `initialize()` or `evolve()` call. The same `GraceEngine` implementation is used in every domain. A mandatory domain adapter would couple the product to one team's runtime, so GRACE deliberately does not define one.

## Choose an initialization path

Use Prompt-to-Graph when the only existing artifact is an instruction:

```python
initial = engine.initialize(instruction=system_instruction)
```

Use a caller-owned graph when an internal system already maintains one:

```python
initial = engine.initialize_from_graph(
    graph=current_graph,
    instruction=system_instruction,
)
```

`initialize_from_graph` makes no provider call. It validates the graph under the active schema, creates an integrity-bound state zero, and persists it through the same artifact contract as Prompt-to-Graph initialization.

## Supply a diagnosis, not an internal trajectory

The caller keeps trajectories and analysis inside its own trust boundary. GRACE receives only an actionable report:

```python
updated = engine.evolve(
    state=initial.state,
    diagnosis_report=diagnosis_report,
)
```

A useful report normally states:

- the observed behavioral gap;
- evidence or context that makes the gap credible;
- the affected instruction behavior;
- the desired corrective behavior and important constraints.

Numeric scores are optional. A report can be produced by an LLM judge, deterministic telemetry, human review, or a combination. GRACE does not require an upstream rubric.

## Trust and data boundaries

Trajectory collection and diagnosis generation remain outside GRACE. The caller decides what evidence to include in the diagnosis report, but the configured provider receives the system instruction, current graph context, and diagnosis content needed by Prompt-to-Graph or Evolution. Use a provider inside the required trust boundary when those inputs cannot be sent to an external service.

Checkpoint persistence is enabled by default. Checkpoint mode stores the graph, reconstructed instruction, schema snapshot, typed reports, hashes, and provider telemetry needed for integrity-checked resume; audit mode additionally stores exact prompt snapshots; `none` leaves persistence to the caller. GRACE does not intentionally persist credentials, but checkpoint artifacts contain instruction and graph data, while audit prompt snapshots can also contain diagnosis content. The integrating team owns access control, retention, and review of both provider traffic and the configured artifact directory.

## Decide whether to define an ontology

The paper's schema (`identity`, `norm`, `knowledge`) is a valid general default. A custom schema is useful when a professional domain has semantic categories or relations whose distinction materially improves validation.

Keep the ontology small. Define instruction-level meaning, not every entity in a database. For each schema, specify:

- object types and their natural-language meanings;
- relation types and their natural-language meanings;
- allowed source/target type pairs;
- relations that must remain acyclic.

Pass the selected schema once as `GraceEngine(..., schema=schema)`. GRACE then renders that active schema into every graph-facing prompt and enforces it again through deterministic validation. See [Network schemas](network-schema.md) for the typed contract and custom-schema example.

## Framework-neutral lifecycle

```text
existing agent runtime
    -> collect its own evidence
    -> produce a diagnosis report
    -> GRACE evolve(current state, report)
    -> review the accepted diff and instruction
    -> deploy through the existing runtime
```

The integrating system owns deployment approval and evaluation. GRACE owns graph operations, structural validation, instruction reconstruction, state integrity, and artifact lineage.

Run the complete zero-network custom-domain example:

```bash
python examples/custom_domain.py --artifact-dir ./grace_runs
```

The example uses incident response to demonstrate a custom ontology and a caller-owned initial graph. Replace its scripted provider with the Vertex or Google AI Studio adapter, or implement the documented two-member [custom provider contract](providers.md#custom-providers) for another model or internal inference gateway.
