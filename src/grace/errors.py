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

"""Public exception hierarchy for GRACE."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grace.providers.base import ProviderAttemptRecord


class GraceError(Exception):
    """Base class for public GRACE failures."""


class ConfigurationError(GraceError):
    """Raised when engine, schema, or artifact configuration is invalid."""


class StateIntegrityError(GraceError):
    """Raised when graph, instruction, schema, or lineage hashes do not match."""


class SchemaValidationError(GraceError):
    """Raised when a graph cannot satisfy deterministic schema constraints."""


class ProviderError(GraceError):
    """Raised when an LLM provider request or response cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        attempts: tuple[ProviderAttemptRecord, ...] = (),
    ) -> None:
        super().__init__(message)
        # Attempt records are typed by the provider layer.  Keeping this base
        # module dependency-free avoids an errors <-> providers import cycle.
        self.attempts: tuple[ProviderAttemptRecord, ...] = attempts


class ProviderConfigurationError(ProviderError):
    """Raised when a provider route or runtime policy is invalid."""


class ProviderAuthenticationError(ProviderError):
    """Raised when an explicitly configured credential cannot be discovered."""


class ProviderTransportError(ProviderError):
    """Raised after bounded transient provider attempts are exhausted."""


class ProviderResponseError(ProviderError):
    """Raised when a provider response is empty, malformed, or unsupported."""


class ArtifactError(GraceError):
    """Raised when an artifact cannot be written, finalized, or loaded safely."""
