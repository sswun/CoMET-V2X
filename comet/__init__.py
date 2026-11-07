"""CoMET: causal common-mode error tracking for asynchronous V2X cooperative detection."""
from .acquisition import AcquisitionParams, CoMETAEstimator, acquire_coarse_offset, estimate_offsets_with_acquisition
from .estimator import (
    CoMETEstimator,
    CommonModeKF,
    apply_offset,
    common_mode_observation,
    estimate_offsets,
    mutual_hungarian,
)
from .fusion import bev_iou, combine_two_tiers, fit_unmatched_coefficient
from .stress import injection_shifts, sequence_direction

__all__ = [
    "AcquisitionParams",
    "CoMETAEstimator",
    "CoMETEstimator",
    "CommonModeKF",
    "apply_offset",
    "common_mode_observation",
    "estimate_offsets",
    "mutual_hungarian",
    "acquire_coarse_offset",
    "estimate_offsets_with_acquisition",
    "bev_iou",
    "combine_two_tiers",
    "fit_unmatched_coefficient",
    "injection_shifts",
    "sequence_direction",
]
__version__ = "0.1.0"
