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

"""Safe YAML and mapping loaders for custom GRACE schemas."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from grace.errors import ConfigurationError
from grace.schemas.models import NetworkSchema


def load_schema_data(data: Mapping[str, Any]) -> NetworkSchema:
    """Validate a schema mapping and return its immutable typed form."""

    try:
        return NetworkSchema.model_validate(dict(data))
    except ValidationError as exc:
        raise ConfigurationError(f"invalid network schema: {exc}") from exc


def load_schema(path: str | Path) -> NetworkSchema:
    """Load a custom network schema with ``yaml.safe_load``."""

    schema_path = Path(path)
    try:
        with schema_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot load network schema {schema_path}: {exc}") from exc

    if not isinstance(data, Mapping):
        raise ConfigurationError(f"network schema {schema_path} must contain one YAML mapping")
    return load_schema_data(data)


__all__ = ["load_schema", "load_schema_data"]
