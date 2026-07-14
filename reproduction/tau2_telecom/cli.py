# Copyright 2026 Dan C. Hsu and Luke Lu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Command-line interface for formal Tau2 telecom reproduction."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys
from typing import Any

from pydantic import TypeAdapter, ValidationError

from grace.providers import GoogleAIStudioConfig, LiteLLMProvider, VertexAIADCConfig

from .artifacts import Tau2TelecomArtifactLayout
from .experiment import (
    Tau2TelecomEpisodePlan,
    Tau2TelecomEvaluationPlan,
    Tau2TelecomExperimentConfig,
)
from .harness.attempt_ledger import FileProviderAttemptLedger
from .harness.identity import canonical_json
from .harness.metrics import EvaluationObservation, compute_evaluation_metrics
from .harness.models import ExpectedMatrix
from .harness.splits import load_tau2_telecom_task_split
from .harness.tau2_interface import get_tau2_initial_policy
from .live_backend import Tau2TelecomLiveBackend
from .runner import Tau2TelecomExperimentRunner


EXIT_OK = 0
EXIT_INVALID = 2
DEFAULT_CONFIG = Path(__file__).parent / "configs" / "reproduction.yaml"


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=("grace", "grace-no-sa", "hce"), default="grace")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override the reproduction-config seed (default: 1024)",
    )
    parser.add_argument("--evaluation", default="in-loop:all")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifacts", type=Path, default=Path("./tau2_telecom_runs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--instruction", type=Path, default=None)
    parser.add_argument("--provider", choices=("vertex", "ai-studio"), default="vertex")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reproduction.tau2_telecom",
        description="Run or inspect the formal GRACE Tau2 telecom workflow.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run-all", "initialize", "status"):
        _add_run_arguments(commands.add_parser(name))
    batch = commands.add_parser("run-batch")
    _add_run_arguments(batch)
    batch.add_argument("--update", type=int, required=True)
    diagnose = commands.add_parser("diagnose")
    _add_run_arguments(diagnose)
    diagnose.add_argument("--update", type=int, required=True)
    evolve = commands.add_parser("evolve")
    _add_run_arguments(evolve)
    evolve.add_argument("--update", type=int, required=True)
    evaluate = commands.add_parser("evaluate")
    _add_run_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=int, required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--evaluation", default="in-loop:all")
    plan.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    validate = commands.add_parser("validate-split")
    validate.add_argument("manifest", nargs="?", type=Path, default=None)
    metrics = commands.add_parser("metrics")
    metrics.add_argument("--expected", required=True, type=Path)
    metrics.add_argument("--observations", required=True, type=Path)
    return parser


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _emit(value: object, *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    stream.write(canonical_json(value) + "\n")


def _progress(message: str) -> None:
    sys.stderr.write(f"[GRACE] {message}\n")


def _configured(arguments: argparse.Namespace) -> Tau2TelecomExperimentConfig:
    config = Tau2TelecomExperimentConfig.load(arguments.config)
    seed = getattr(arguments, "seed", None)
    if seed is not None:
        config = config.model_copy(
            update={"execution": config.execution.model_copy(update={"seed": seed})}
        )
    return config


def _provider_and_runtime(
    arguments: argparse.Namespace,
    *,
    ledger: FileProviderAttemptLedger,
) -> tuple[LiteLLMProvider, dict[str, Any]]:
    if arguments.provider == "vertex":
        project = os.getenv("GRACE_VERTEX_PROJECT")
        if not project:
            raise ValueError("GRACE_VERTEX_PROJECT is required for the Vertex provider")
        location = os.getenv("GRACE_VERTEX_LOCATION", "us-central1")
        provider = LiteLLMProvider(
            VertexAIADCConfig(project=project, location=location), attempt_hook=ledger
        )
        return provider, {"vertex_project": project, "vertex_location": location}
    provider = LiteLLMProvider(GoogleAIStudioConfig(), attempt_hook=ledger)
    return provider, {}


def _runner(arguments: argparse.Namespace) -> Tau2TelecomExperimentRunner:
    config = _configured(arguments)
    evaluation = Tau2TelecomEvaluationPlan.parse(
        arguments.evaluation, final_checkpoint=config.execution.evolution_updates
    )
    run_id = arguments.run_id or (f"tau2-telecom-{arguments.method}-seed-{config.execution.seed}")
    run_dir = arguments.artifacts.expanduser().resolve() / run_id
    ledger = FileProviderAttemptLedger(run_dir / "provider_attempts")
    provider, agent_runtime = _provider_and_runtime(arguments, ledger=ledger)
    # The route may change, but the semantic Gemini model remains fixed.
    if provider.model != config.models.task_agent:
        config = config.model_copy(
            update={
                "models": config.models.model_copy(
                    update={
                        "task_agent": provider.model,
                        "diagnosis": provider.model,
                        "evolution": provider.model,
                        "p2g": provider.model,
                    }
                )
            }
        )
    split = load_tau2_telecom_task_split(config.benchmark.task_split)
    instruction = (
        arguments.instruction.read_text(encoding="utf-8")
        if arguments.instruction is not None
        else get_tau2_initial_policy()
    )
    backend = Tau2TelecomLiveBackend(
        config=config,
        split=split,
        provider=provider,
        artifact_root=arguments.artifacts,
        run_id=run_id,
        attempt_ledger=ledger,
        agent_runtime_args=agent_runtime,
    )
    return Tau2TelecomExperimentRunner(
        config=config,
        method=arguments.method,
        evaluation=evaluation,
        initial_instruction=instruction,
        artifact_root=arguments.artifacts,
        run_id=run_id,
        backend=backend,
        progress=_progress,
    )


def _offline_status(arguments: argparse.Namespace) -> dict[str, object]:
    config = _configured(arguments)
    run_id = arguments.run_id or (f"tau2-telecom-{arguments.method}-seed-{config.execution.seed}")
    layout = Tau2TelecomArtifactLayout(arguments.artifacts, run_id)
    contract = layout.load_contract()
    if contract is None:
        raise ValueError(f"run {run_id!r} does not exist")
    checkpoints = tuple(
        index
        for index in range(config.execution.evolution_updates + 1)
        if layout.load_checkpoint(index) is not None
    )
    evaluations = tuple(
        index
        for index in contract.evaluation.checkpoint_indices
        if layout.load_stage("evaluation", index) is not None
    )
    return {
        "run_id": contract.run_id,
        "method": contract.method,
        "run_fingerprint": contract.run_fingerprint,
        "provenance": contract.provenance.model_dump(mode="json"),
        "completed_checkpoints": checkpoints,
        "completed_evaluations": evaluations,
        "complete": checkpoints == tuple(range(config.execution.evolution_updates + 1))
        and evaluations == contract.evaluation.checkpoint_indices,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "validate-split":
            manifest = load_tau2_telecom_task_split(arguments.manifest)
            _emit(
                {
                    "benchmark_revision": manifest.benchmark_revision,
                    "evaluation_tasks": len(manifest.eval),
                    "experience_updates": len(manifest.experience),
                    "seed": Tau2TelecomExperimentConfig.load(DEFAULT_CONFIG).execution.seed,
                    "status": "valid",
                }
            )
            return EXIT_OK
        if arguments.command == "metrics":
            expected = ExpectedMatrix.model_validate(_read_json(arguments.expected))
            observations = TypeAdapter(tuple[EvaluationObservation, ...]).validate_python(
                _read_json(arguments.observations)
            )
            _emit(compute_evaluation_metrics(expected, observations).model_dump(mode="json"))
            return EXIT_OK
        if arguments.command == "plan":
            config = _configured(arguments)
            evaluation = Tau2TelecomEvaluationPlan.parse(
                arguments.evaluation,
                final_checkpoint=config.execution.evolution_updates,
            )
            plan = Tau2TelecomEpisodePlan.create(config=config, evaluation=evaluation)
            _emit(plan.model_dump(mode="json"))
            return EXIT_OK
        if arguments.command == "status":
            _emit(_offline_status(arguments))
            return EXIT_OK
        runner = _runner(arguments)
        if arguments.command == "run-all":
            runner.run_all()
        elif arguments.command == "initialize":
            runner.initialize()
        elif arguments.command == "run-batch":
            runner.run_batch(arguments.update)
        elif arguments.command == "diagnose":
            runner.diagnose(arguments.update)
        elif arguments.command == "evolve":
            runner.evolve(arguments.update)
        elif arguments.command == "evaluate":
            runner.evaluate(arguments.checkpoint)
        _emit(runner.status())
        return EXIT_OK
    except (OSError, RuntimeError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        _emit({"error": str(exc), "status": "invalid"}, error=True)
        return EXIT_INVALID


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["EXIT_INVALID", "EXIT_OK", "main"]
