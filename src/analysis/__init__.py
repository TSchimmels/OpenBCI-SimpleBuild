"""ERP analysis and neurofeedback module.

Provides real-time ERP accumulation, time-frequency decomposition
(ERDS%), and scalp topographic mapping for motor imagery BCI training.
"""

from .erp import ERPAccumulator
from .time_frequency import ERDSComputer
from .topography import TopoMapper

__all__ = ["ERPAccumulator", "ERDSComputer", "TopoMapper"]
