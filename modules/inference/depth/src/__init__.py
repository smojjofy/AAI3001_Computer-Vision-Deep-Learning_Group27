from .contracts import (
    DEPTH_KINDS,
    DepthEstimation,
    DepthFrameResult,
    DepthKind,
)
# The DepthEstimator class is the main entry point for performing depth estimation on images. It loads an official Depth Anything V2 checkpoint, runs it on single frames, and returns depth in the shared depth convention (see DEPTH_CONVENTION.md).
from .estimator import (
    DEFAULT_MODEL,
    DepthEstimator,
    find_default_model,
    find_repo,
    resolve_device,
)
from .normalize import (
    edge_confidence,
    normalize_prediction,
)
# The __all__ variable is a list of public objects of that module, as interpreted by import *. It defines the symbols that will be exported when the module is imported using the from module import * syntax. This helps to control the namespace and avoid polluting it with unnecessary or internal symbols.
__all__ = [
    "DepthEstimator",
    "DepthEstimation",
    "DepthFrameResult",
    "DepthKind",
    "DEPTH_KINDS",
    "DEFAULT_MODEL",
    "find_default_model",
    "find_repo",
    "resolve_device",
    "normalize_prediction",
    "edge_confidence",
]
