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

"""Small real Vertex AI initialization and Evolution example."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from grace import GraceEngine
from grace.providers import LiteLLMProvider, VertexAIADCConfig


INSTRUCTION = """You are an incident triage agent.
Verify the reported symptoms before recommending a remediation.
Escalate when evidence is incomplete or the action is irreversible.
"""

DIAGNOSIS = """Observed failure: the agent recommended a restart before checking
service health. Root cause: the instruction does not make evidence collection
an explicit prerequisite. Add a rule requiring health and dependency checks
before remediation, while preserving the existing escalation boundary.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("./grace_runs"))
    arguments = parser.parse_args()
    provider = LiteLLMProvider(
        VertexAIADCConfig(
            project=os.environ["GRACE_VERTEX_PROJECT"],
            location=os.getenv("GRACE_VERTEX_LOCATION", "us-central1"),
        )
    )
    engine = GraceEngine(provider=provider, artifact_dir=arguments.artifact_dir)
    initial = engine.initialize(instruction=INSTRUCTION)
    evolved = engine.evolve(state=initial.state, diagnosis_report=DIAGNOSIS)
    print(evolved.state.instruction)


if __name__ == "__main__":
    main()
