from .contract import CustomEstimator, CustomTransformer
from .contracts import (
    CandidatePluginContract,
    ExecutionBackendContract,
    PilotResourceReport,
    ProblemValidationContract,
    ResourceRequestContract,
    ResultBundleContract,
    TrainingStateContract,
    file_sha256,
    CustomJobSpecContract,
    TrainingCheckpointContract,
    TrainingResultContract,
    WorkerHandshakeContract,
)
from .admission import (
    AdmissionController,
    AdmissionDecision,
    OOMRecoveryPolicy,
    OOMRecoveryPlan,
    feature_detect_amp,
)
from .kaggle_backend import KaggleExecutionBackend
from .tuning import (
    BudgetedOptunaTuner,
    BudgetedTrial,
    TrialPruningHook,
    TrialState,
)
from .delivery import (
    CustomBundlePackager,
    BundleVerificationError,
)

try:
    from .torch_candidate import TinyTorchNeuralCandidate
except ImportError:
    TinyTorchNeuralCandidate = None  # type: ignore

try:
    from .sequential_candidate import SequentialTorchCandidate
except ImportError:
    SequentialTorchCandidate = None  # type: ignore

from .tabular_candidate import TabularCandidatePlugin

def run_worker(*args, **kwargs):
    from .worker import run_worker as _run_worker
    return _run_worker(*args, **kwargs)

__all__ = [
    "CustomEstimator",
    "CustomTransformer",
    "CandidatePluginContract",
    "ExecutionBackendContract",
    "PilotResourceReport",
    "ProblemValidationContract",
    "ResourceRequestContract",
    "ResultBundleContract",
    "TrainingStateContract",
    "file_sha256",
    "CustomJobSpecContract",
    "TrainingCheckpointContract",
    "TrainingResultContract",
    "WorkerHandshakeContract",
    "TinyTorchNeuralCandidate",
    "SequentialTorchCandidate",
    "TabularCandidatePlugin",
    "AdmissionController",
    "AdmissionDecision",
    "OOMRecoveryPolicy",
    "OOMRecoveryPlan",
    "feature_detect_amp",
    "KaggleExecutionBackend",
    "BudgetedOptunaTuner",
    "BudgetedTrial",
    "TrialPruningHook",
    "TrialState",
    "CustomBundlePackager",
    "BundleVerificationError",
    "run_worker",
]
