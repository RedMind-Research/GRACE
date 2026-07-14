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

from grace.providers.base import ProviderAttemptContext, ProviderAttemptRecord, UsageRecord
from grace.providers.config import ProviderDescriptor
from reproduction.tau2_telecom.harness.attempt_ledger import (
    FileProviderAttemptLedger,
    ProviderAttemptReservation,
)


def context(*, elapsed: float = 0) -> ProviderAttemptContext:
    return ProviderAttemptContext(
        descriptor=ProviderDescriptor(
            route="vertex_ai_adc",
            model_id="gemini-2.5-flash",
            model="vertex_ai/gemini-2.5-flash",
        ),
        stage="test",
        logical_call_id="a" * 64,
        attempt=1,
        input_token_bound=100,
        temperature=0.0,
        timeout_seconds=10,
        overall_timeout_seconds=10,
        elapsed_since_call_start_seconds=elapsed,
        max_tokens=100,
    )


def test_ledger_reserves_before_dispatch_and_settles_once(tmp_path: Path) -> None:
    ledger = FileProviderAttemptLedger(tmp_path)
    reservation = ledger.before_attempt(context())
    assert isinstance(reservation, ProviderAttemptReservation)
    attempt_id = reservation.provider_attempt_id

    assert (tmp_path / "pending" / f"{attempt_id}.json").exists()
    ledger.after_attempt(
        reservation,
        attempt=ProviderAttemptRecord(
            attempt=1,
            temperature=0.0,
            timeout_seconds=10,
            overall_timeout_seconds=10,
            elapsed_since_call_start_seconds=0,
            latency_seconds=1,
            outcome="succeeded",
            category="test",
        ),
        usage=UsageRecord(
            model="vertex_ai/gemini-2.5-flash",
            input_tokens=1,
            output_tokens=1,
            latency_seconds=1,
        ),
    )

    assert not (tmp_path / "pending" / f"{attempt_id}.json").exists()
    assert (tmp_path / "settled" / f"{attempt_id}.json").exists()
    with pytest.raises(ValueError, match="settled"):
        ledger.before_attempt(context())


def test_pending_attempt_blocks_automatic_redispatch(tmp_path: Path) -> None:
    ledger = FileProviderAttemptLedger(tmp_path)
    ledger.before_attempt(context())
    with pytest.raises(ValueError, match="already exists"):
        ledger.before_attempt(context(elapsed=3.5))
