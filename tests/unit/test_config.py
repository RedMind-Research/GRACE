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

from pathlib import Path

import pytest
from pydantic import ValidationError

from grace.config import GraceConfig


def test_config_uses_public_product_defaults() -> None:
    config = GraceConfig()

    assert config.max_output_tokens == 65536
    assert config.p2g_max_rounds == 10
    assert config.structural_analysis is True
    assert config.sa_max_rounds == 3
    assert config.sa_radius_step == 3
    assert config.schema_repair_max_rounds == 3
    assert config.artifact_dir == Path("grace_runs")
    assert config.artifact_mode == "checkpoint"


def test_config_is_frozen_and_rejects_unknown_fields() -> None:
    config = GraceConfig()

    with pytest.raises(ValidationError):
        config.artifact_mode = "audit"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        GraceConfig(unknown_option=True)  # type: ignore[call-arg]


@pytest.mark.parametrize("value", [0, -1, True, "4096"])
def test_max_output_tokens_requires_a_strict_positive_integer(value: object) -> None:
    with pytest.raises(ValidationError):
        GraceConfig(max_output_tokens=value)  # type: ignore[arg-type]
