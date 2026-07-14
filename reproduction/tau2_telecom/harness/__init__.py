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

"""Minimal, user-owned tau2 telecom reproduction primitives.

Importing this package performs no network, provider, or tau2 setup.
"""

from .completeness import (
    CompletenessError,
    build_expected_matrix,
    compare_exact_matrix,
    require_exact_matrix,
)
from .diagnosis import (
    DiagnosisArtifactError,
    DiagnosisError,
    DiagnosisFingerprintMismatch,
    FileDiagnosisStore,
    classify_trajectory,
    render_diagnosis_report,
    run_diagnosis,
)
from .evaluation import run_evaluation_smoke
from .experience import run_experience_collection
from .metrics import (
    EvaluationMetrics,
    EvaluationObservation,
    MetricsContractError,
    compute_evaluation_metrics,
)
from .models import ExpectedMatrix, TaskSelectionManifest, TrajectoryRecord
from .tasks import (
    TaskResolutionError,
    load_task_selection_manifest,
    load_tasks_in_manifest_order,
    resolve_task_selection,
)
from .tau2_interface import (
    EpisodeHarnessConfig,
    execute_matrix_sequential,
    load_tau2_task_selection,
    require_tau2_revision,
)

__all__ = [
    "CompletenessError",
    "DiagnosisArtifactError",
    "DiagnosisError",
    "DiagnosisFingerprintMismatch",
    "EpisodeHarnessConfig",
    "EvaluationMetrics",
    "EvaluationObservation",
    "ExpectedMatrix",
    "FileDiagnosisStore",
    "MetricsContractError",
    "TaskResolutionError",
    "TaskSelectionManifest",
    "TrajectoryRecord",
    "build_expected_matrix",
    "classify_trajectory",
    "compare_exact_matrix",
    "compute_evaluation_metrics",
    "execute_matrix_sequential",
    "load_task_selection_manifest",
    "load_tasks_in_manifest_order",
    "load_tau2_task_selection",
    "render_diagnosis_report",
    "require_exact_matrix",
    "require_tau2_revision",
    "resolve_task_selection",
    "run_diagnosis",
    "run_evaluation_smoke",
    "run_experience_collection",
]
