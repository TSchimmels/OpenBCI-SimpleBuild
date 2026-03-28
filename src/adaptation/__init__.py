"""Self-adapting BCI module.

Uses involuntary ErrP (error-related potential) and P300 (confirmation)
brain signals as reward signals for continuous classifier adaptation
via the SEAL (Self-Evolving Adaptive Learning) engine.

References:
    Schmidt & Blankertz (2010). Online detection of error-related
    potentials boosts mental typewriters.

    Chavarriaga et al. (2014). Errare machinale est: the use of
    error-related potentials in BCI.
"""

from .errp_detector import ErrPP300Detector
from .seal_engine import SEALAdaptationEngine

__all__ = ["ErrPP300Detector", "SEALAdaptationEngine"]
