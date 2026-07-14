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

"""Run the complete product contract with a scripted, zero-network provider."""

from __future__ import annotations

import argparse
import json
import uuid
from collections import deque
from pathlib import Path

from grace import GraceEngine
from grace.providers import PromptRequest, ProviderCallResult, UsageRecord


class ScriptedProvider:
    """Minimal provider used only to demonstrate dependency injection."""

    def __init__(self) -> None:
        self._responses = deque(
            [
                {
                    "nodes": [
                        {
                            "id": "N001",
                            "type": "identity",
                            "content": "You are an account-support agent.",
                        },
                        {
                            "id": "N002",
                            "type": "norm",
                            "content": "Verify the account before changing it.",
                        },
                    ],
                    "edges": [
                        {
                            "source": "N001",
                            "target": "N002",
                            "relation": "supports",
                        }
                    ],
                },
                {
                    "missing_node": [],
                    "unfaithful": [],
                    "mistyped": [],
                    "wrong_relation": [],
                    "missing_relation": [],
                },
                {
                    "operations": [
                        {
                            "op": "AddNode",
                            "type": "knowledge",
                            "content": "The account status must be inspected before a change.",
                        }
                    ]
                },
                {"operations": []},
                {"contradictions": [], "redundancies": []},
                {
                    "operations": [
                        {
                            "op": "InsertAfter",
                            "anchor": "Verify the account before changing it.",
                            "new_text": "Inspect and record the current account status first.",
                        }
                    ]
                },
            ]
        )
        self.calls = 0

    @property
    def model(self) -> str:
        return "scripted/offline-quickstart"

    def complete(self, request: PromptRequest) -> ProviderCallResult:
        del request
        self.calls += 1
        return ProviderCallResult(
            parsed_content=self._responses.popleft(),
            usage=UsageRecord(
                model=self.model,
                route="scripted",
                input_tokens=10,
                output_tokens=5,
                latency_seconds=0.0,
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("grace_runs"))
    args = parser.parse_args()

    provider = ScriptedProvider()
    engine = GraceEngine(
        provider=provider,
        artifact_dir=args.artifact_dir,
        artifact_mode="audit",
        run_id=f"offline-example-{uuid.uuid4().hex[:12]}",
    )
    initial = engine.initialize(
        instruction=("You are an account-support agent.\nVerify the account before changing it.")
    )
    evolved = engine.evolve(
        state=initial.state,
        diagnosis_report=(
            "Observed gap: the agent verified identity but did not inspect the "
            "current account status before making a change. Add an explicit "
            "pre-change inspection step."
        ),
    )
    print(
        json.dumps(
            {
                "artifact_root": str(args.artifact_dir.absolute()),
                "evolved_artifact": str(evolved.artifact_location),
                "evolved_state_id": evolved.state.state_id,
                "initial_state_id": initial.state.state_id,
                "provider_calls": provider.calls,
                "run_id": engine.run_id,
                "status": evolved.status.value,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
