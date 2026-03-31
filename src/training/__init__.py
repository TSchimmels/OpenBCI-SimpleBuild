"""Training module."""

from .paradigm import GrazParadigm
from .recorder import DataRecorder
from .trainer import ModelTrainer


def __getattr__(name):
    if name == "JEPAPretrainer":
        from .pretrain import JEPAPretrainer
        return JEPAPretrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GrazParadigm", "DataRecorder", "ModelTrainer", "JEPAPretrainer"]
