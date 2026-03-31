"""Feature extraction module."""

from .csp import CSPExtractor
from .chaos import ChaosFeatureExtractor
from .bandpower import BandPowerExtractor
from .jacobian_features import JacobianFeatureExtractor


def __getattr__(name):
    if name == "VariableSelector":
        from .variable_selector import VariableSelector
        return VariableSelector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CSPExtractor", "ChaosFeatureExtractor", "BandPowerExtractor",
    "JacobianFeatureExtractor", "VariableSelector",
]
