"""Training module."""

from .recorder import DataRecorder
from .trainer import ModelTrainer


def __getattr__(name):
    if name == "GrazParadigm":
        from .paradigm import GrazParadigm
        return GrazParadigm
    if name == "JEPAPretrainer":
        from .pretrain import JEPAPretrainer
        return JEPAPretrainer
    if name == "AdvancedTrainingPipeline":
        from .advanced_pipeline import AdvancedTrainingPipeline
        return AdvancedTrainingPipeline
    if name == "SubjectProfile":
        from .advanced_pipeline import SubjectProfile
        return SubjectProfile
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GrazParadigm",
    "DataRecorder",
    "ModelTrainer",
    "JEPAPretrainer",
    "AdvancedTrainingPipeline",
    "SubjectProfile",
]
