"""Classification module."""

from .base import BaseClassifier
from .pipeline import ClassifierFactory


def __getattr__(name):
    if name == "AdaptiveClassifierRouter":
        from .adaptive_router import AdaptiveClassifierRouter
        return AdaptiveClassifierRouter
    if name == "NeuralSDEClassifier":
        from .neural_sde import NeuralSDEClassifier
        return NeuralSDEClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseClassifier", "ClassifierFactory",
    "AdaptiveClassifierRouter", "NeuralSDEClassifier",
]
