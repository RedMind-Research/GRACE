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

"""Configuration for the public GRACE evolution engine."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MAX_OUTPUT_TOKENS = 65536


class GraceConfig(BaseModel):
    """Method and persistence defaults for GRACE.

    The values reproduce the paper implementation unless a caller
    explicitly overrides them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, gt=0, strict=True)
    p2g_max_rounds: int = Field(default=10, ge=1)
    structural_analysis: bool = True
    sa_max_rounds: int = Field(default=3, ge=1)
    sa_radius_step: int = Field(default=3, ge=1)
    schema_repair_max_rounds: int = Field(default=3, ge=0)
    artifact_dir: Path = Path("./grace_runs")
    artifact_mode: Literal["checkpoint", "audit", "none"] = "checkpoint"
