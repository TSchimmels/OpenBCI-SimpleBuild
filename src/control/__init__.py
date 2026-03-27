"""Cursor control module for pure EEG-based BCI.

MouseController requires a display server (X11/Wayland) for pyautogui.
Imports are lazy to allow headless usage of ControlMapper and other
non-display components (e.g., in tests or server environments).
"""


def __getattr__(name):
    if name == "MouseController":
        from .mouse import MouseController
        return MouseController
    if name == "ControlMapper":
        from .mapping import ControlMapper
        return ControlMapper
    if name == "EEGCursorController":
        from .cursor_control import EEGCursorController
        return EEGCursorController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MouseController", "ControlMapper", "EEGCursorController"]
