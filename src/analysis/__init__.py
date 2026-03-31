"""ERP analysis, neurofeedback, and advanced signal analysis module."""

from .erp import ERPAccumulator
from .time_frequency import ERDSComputer
from .topography import TopoMapper
from .state_monitor import BCIStateMonitor


def __getattr__(name):
    if name == "CausalChannelDiscovery":
        from .causal_channels import CausalChannelDiscovery
        return CausalChannelDiscovery
    if name == "KoopmanEEGDecomposition":
        from .koopman_decomposition import KoopmanEEGDecomposition
        return KoopmanEEGDecomposition
    if name == "FTLEAnalyzer":
        from .ftle_analysis import FTLEAnalyzer
        return FTLEAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ERPAccumulator", "ERDSComputer", "TopoMapper",
    "BCIStateMonitor", "CausalChannelDiscovery",
    "KoopmanEEGDecomposition", "FTLEAnalyzer",
]
