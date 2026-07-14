# Network schemas and domain integration

GRACE represents the evolving instruction as a directed typed graph `G = (V, E)`. A high-level network schema is

```text
T_G = (A, R, {sigma_r | r in R})
```

where `A` is the set of object types, `R` is the set of relation types, and `sigma_r` declares the admissible source/target object-type pairs for relation `r`. Each type also has a natural-language definition. Together, these ontology-level definitions tell the model what graph objects mean while the deterministic layer enforces valid identifiers, endpoints, signatures, uniqueness, and acyclicity.

## Default paper schema

The built-in `DefaultGraceSchema` uses three object types:

| Object type | Role |
| --- | --- |
| `identity` | The agent's standing and mandate as the subject of action. |
| `norm` | A standard that governs conduct and can be satisfied or breached. |
| `knowledge` | A self-contained description of the task world used for reasoning. |

It uses three directed relation types:

| Relation | Admissible signatures | Constraint |
| --- | --- | --- |
| `supports` | `identity→norm`, `knowledge→norm`, `knowledge→knowledge`, `norm→norm` | Directed grounding. |
| `refines` | same-type `identity`, `norm`, or `knowledge` pairs | Acyclic specialization. |
| `sequence` | `norm→norm` | Acyclic procedural order. |

This schema is a useful default and is the schema used by the telecom research workflow.

## When to define a custom ontology

An external team can use the default schema without any domain adapter. For a specialized domain, we recommend defining a small ontology-derived schema when the domain has important semantic categories or relations that the default schema cannot express precisely.

The goal is not to copy every database entity into the graph. Define the high-level object and relation types needed to maintain and verify the system-level instruction. A practical design sequence is:

1. Identify the semantic kinds of instruction units that should be compared under the same rules.
2. Define only relations that materially affect interpretation, dependency, or order.
3. Declare each relation's allowed source/target signatures and whether it must be acyclic.
4. Give every type an unambiguous natural-language definition.
5. Freeze the schema ID, version, declaration order, and content hash for a run.

GRACE renders the active schema into every graph-facing prompt. Generic graph operations do not hard-code `identity`, `norm`, or `knowledge`.

## Minimal custom schema

```python
from grace import NetworkSchema, ObjectType, RelationType

schema = NetworkSchema(
    id="clinical-workflow",
    version="1",
    description="High-level instruction ontology for a clinical workflow agent.",
    object_types=(
        ObjectType(
            id="role",
            description="The authorized role and scope under which the agent acts.",
        ),
        ObjectType(
            id="safety_rule",
            description="A safety constraint that governs an agent action.",
        ),
        ObjectType(
            id="procedure",
            description="An ordered operational step in the workflow.",
        ),
    ),
    relation_types=(
        RelationType(
            id="governs",
            description="The source role governs the target safety rule.",
            allowed_pairs=(("role", "safety_rule"),),
        ),
        RelationType(
            id="precedes",
            description="The source procedure must occur before the target procedure.",
            allowed_pairs=(("procedure", "procedure"),),
            acyclic=True,
        ),
    ),
)
```

This is an ontology configuration, not a domain-specific GRACE fork. Inject it when constructing the engine; schema selection is fixed for that engine and is not a per-call argument:

```python
from grace import GraceEngine

engine = GraceEngine(
    provider=provider,
    schema=schema,
    artifact_dir="./grace_runs",
)
initial = engine.initialize(instruction=system_instruction)
updated = engine.evolve(
    state=initial.state,
    diagnosis_report=diagnosis_report,
)
```

The engine automatically reuses the active schema for Prompt-to-Graph, Evolution prompts, deterministic validation, state identity, and persisted artifacts. `initialize()` and `evolve()` therefore do not accept a separate `schema` parameter.

## Schema identity and artifacts

Declaration order is operational because it affects schema-rendered prompts; the schema hash therefore preserves that order. Checkpoint and audit runs write one canonical `schema.json` snapshot at the run root. A resumed process must load or provide a schema with the same ID and hash. This prevents a graph from being silently interpreted under a changed ontology.

An optional `GroundingPolicy` can define a domain-specific root/backbone rule. Omit it when the ontology has no equivalent requirement; generic validation and Evolution still work.
