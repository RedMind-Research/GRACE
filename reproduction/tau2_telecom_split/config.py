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

"""Pinned input and per-persona quotas for rebuilding the telecom task split."""

SPLIT_SEED = 42
PINNED_TASKS_SHA256 = "37e562e1ae3242577407e1303b1548bc64e7ea68e37d36173e6747990ceaf8a4"
PERSONAS = ("None", "Easy", "Hard")

# Dict insertion order sets the intent traversal order for every draw.
EVALUATION_QUOTAS = {"mms_issue": 10, "mobile_data_issue": 10, "service_issue": 2}
EXPERIENCE_QUOTAS = {
    "A": {"mms_issue": 10, "mobile_data_issue": 3, "service_issue": 1},
    "B": {"mms_issue": 3, "mobile_data_issue": 10, "service_issue": 1},
}
