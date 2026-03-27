"""Cursor control module for pure EEG-based BCI."""

from .mouse import MouseController
from .mapping import ControlMapper
from .cursor_control import EEGCursorController

__all__ = ["MouseController", "ControlMapper", "EEGCursorController"]
