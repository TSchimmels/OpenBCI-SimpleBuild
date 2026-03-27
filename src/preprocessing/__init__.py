"""EEG preprocessing module."""

from .filters import (
    bandpass_filter,
    notch_filter,
    common_average_reference,
    laplacian_reference,
    CausalFilterState,
)
from .artifacts import reject_epochs, detect_bad_channels

__all__ = [
    "bandpass_filter",
    "notch_filter",
    "common_average_reference",
    "laplacian_reference",
    "CausalFilterState",
    "reject_epochs",
    "detect_bad_channels",
]
