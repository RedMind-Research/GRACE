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

from grace.artifacts.models import ReconstructionStatus
from grace.evolution.reconstruction import (
    DeleteSpan,
    InsertAfter,
    ReplaceSpan,
    apply_patch_operations,
)


def test_unambiguous_patch_operations_apply_sequentially() -> None:
    result = apply_patch_operations(
        "Header\nRule A\nRule B",
        [
            InsertAfter(anchor="Header", new_text="Introduction"),
            ReplaceSpan(anchor="Rule A", new_text="Rule A updated"),
            DeleteSpan(anchor="Rule B"),
        ],
    )

    assert result.instruction == "Header\nIntroduction\nRule A updated\n"
    assert result.status == ReconstructionStatus.APPLIED
    assert len(result.proposed_operations) == 3
    assert len(result.applied_operations) == 3
    assert result.skipped_operations == ()
    assert result.appended_operations == ()
    assert result.to_reconstruction_report().output_instruction_hash == (
        result.output_instruction_hash
    )


def test_missing_insert_anchor_appends_but_other_bad_anchors_skip() -> None:
    result = apply_patch_operations(
        "Rule Rule",
        [
            {"op": "ReplaceSpan", "anchor": "Rule", "new_text": "Updated"},
            {"op": "InsertAfter", "anchor": "Missing", "new_text": "Fallback"},
            {"op": "Unknown", "anchor": "Rule"},
            "malformed",
        ],
    )

    assert result.instruction == "Rule Rule\nFallback"
    assert result.status == ReconstructionStatus.COMPLETED_WITH_FALLBACKS
    assert len(result.proposed_operations) == 4
    assert result.applied_operations == ()
    assert len(result.appended_operations) == 1
    assert len(result.skipped_operations) == 3
    assert "3 patch operation(s) skipped" in result.warnings


def test_empty_patch_list_is_a_byte_preserving_no_op() -> None:
    instruction = "Keep trailing whitespace.  \n"
    result = apply_patch_operations(instruction, [])

    assert result.instruction == instruction
    assert result.input_instruction_hash == result.output_instruction_hash
    assert result.status == ReconstructionStatus.NO_OP
