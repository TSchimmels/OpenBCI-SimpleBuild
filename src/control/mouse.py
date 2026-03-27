"""Mouse cursor control via pyautogui.

Provides a thin wrapper around pyautogui that enforces screen-bound
clamping and exposes both absolute and relative movement for absolute positioning
and MI velocity-based control.
"""

from typing import Tuple

import pyautogui


class MouseController:
    """Low-level cursor driver.

    Wraps pyautogui with zero artificial delay and fail-safe enabled so
    that moving the physical mouse to a screen corner aborts execution.

    Example::

        mc = MouseController()
        mc.move_relative(dx=10, dy=0)   # nudge cursor right
        mc.click()

    """

    def __init__(self) -> None:
        pyautogui.PAUSE = 0
        pyautogui.FAILSAFE = True
        self._screen_w, self._screen_h = pyautogui.size()

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move_to(self, x: float, y: float) -> None:
        """Move cursor to an absolute screen position.

        Coordinates are clamped to the screen bounds so the call never
        raises an ``FailSafeException`` due to out-of-range values.
        This is the primary interface for absolute positioning-guided control.

        Args:
            x: Target x-coordinate in pixels.
            y: Target y-coordinate in pixels.
        """
        x = max(0, min(int(x), self._screen_w - 1))
        y = max(0, min(int(y), self._screen_h - 1))
        pyautogui.moveTo(x, y, _pause=False)

    def move_relative(self, dx: float, dy: float) -> None:
        """Move cursor relative to its current position.

        After applying the offset the resulting position is clamped to
        screen bounds.  This is the primary interface for pure MI
        (motor-imagery) velocity-based control.

        Args:
            dx: Horizontal offset in pixels (positive = right).
            dy: Vertical offset in pixels (positive = down).
        """
        cur_x, cur_y = pyautogui.position()
        new_x = max(0, min(int(cur_x + dx), self._screen_w - 1))
        new_y = max(0, min(int(cur_y + dy), self._screen_h - 1))
        pyautogui.moveTo(new_x, new_y, _pause=False)

    # ------------------------------------------------------------------
    # Clicks
    # ------------------------------------------------------------------

    def click(self, button: str = "left") -> None:
        """Single click.

        Args:
            button: ``'left'``, ``'middle'``, or ``'right'``.
        """
        pyautogui.click(button=button, _pause=False)

    def double_click(self) -> None:
        """Double left-click at the current position."""
        pyautogui.doubleClick(_pause=False)

    def right_click(self) -> None:
        """Single right-click at the current position."""
        pyautogui.click(button="right", _pause=False)

    # ------------------------------------------------------------------
    # Drag helpers
    # ------------------------------------------------------------------

    def mouse_down(self, button: str = "left") -> None:
        """Press and hold a mouse button (drag start).

        Args:
            button: ``'left'``, ``'middle'``, or ``'right'``.
        """
        pyautogui.mouseDown(button=button, _pause=False)

    def mouse_up(self, button: str = "left") -> None:
        """Release a held mouse button (drag end).

        Args:
            button: ``'left'``, ``'middle'``, or ``'right'``.
        """
        pyautogui.mouseUp(button=button, _pause=False)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self) -> Tuple[int, int]:
        """Return the current cursor position.

        Returns:
            ``(x, y)`` pixel coordinates.
        """
        pos = pyautogui.position()
        return (pos[0], pos[1])

    def get_screen_size(self) -> Tuple[int, int]:
        """Return the screen resolution.

        Returns:
            ``(width, height)`` in pixels.
        """
        return (self._screen_w, self._screen_h)
