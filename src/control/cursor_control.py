"""EEG-driven cursor controller.

Translates 5-class motor imagery classification into 4-directional
cursor movement with confidence-based velocity and sustained-imagery
click detection. This replaces eye tracking + jaw clench from the
original hybrid system.

Classes:
    - rest: no movement
    - left_hand: cursor LEFT
    - right_hand: cursor RIGHT
    - feet: cursor DOWN
    - tongue: cursor UP

Click: sustained high-confidence classification of any directional
class for a configurable duration triggers a click.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mouse import MouseController
from .mapping import ControlMapper

logger = logging.getLogger(__name__)


class EEGCursorController:
    """State machine for EEG-driven cursor control.

    Combines classification output with velocity mapping, smoothing,
    and click detection into a single update() call per control loop
    iteration.

    Args:
        config: Full application configuration dictionary.
    """

    def __init__(self, config: Dict) -> None:
        control_cfg = config.get("control", {})
        click_cfg = control_cfg.get("click", {})

        self._mouse = MouseController()
        self._mapper = ControlMapper(
            dead_zone=control_cfg.get("dead_zone", 0.15),
            max_velocity=control_cfg.get("max_velocity", 25.0),
            smoothing_alpha=control_cfg.get("smoothing_alpha", 0.3),
        )

        # Direction mapping
        self._direction_map: Dict[str, str] = control_cfg.get("direction_map", {
            "left_hand": "left",
            "right_hand": "right",
            "feet": "down",
            "tongue": "up",
        })

        # Thresholds
        self._move_threshold: float = control_cfg.get("confidence_threshold", 0.5)

        # Click detection state
        self._click_hold_duration: float = click_cfg.get("hold_duration_s", 0.8)
        self._click_threshold: float = click_cfg.get("confidence_threshold", 0.7)
        self._double_click_window: float = click_cfg.get("double_click_window_s", 1.5)
        self._click_cooldown: float = click_cfg.get("cooldown_s", 0.5)

        self._sustained_class: Optional[str] = None
        self._sustained_start: float = 0.0
        self._last_click_time: float = 0.0
        self._click_count: int = 0
        self._first_click_time: float = 0.0

        # Per-axis smoothed velocity
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._alpha: float = control_cfg.get("smoothing_alpha", 0.3)
        self._max_vel: float = control_cfg.get("max_velocity", 25.0)

        # Statistics
        self.total_movements: int = 0
        self.total_clicks: int = 0

        logger.info(
            "EEGCursorController: directions=%s, move_thresh=%.2f, "
            "click_hold=%.1fs, click_thresh=%.2f",
            self._direction_map,
            self._move_threshold,
            self._click_hold_duration,
            self._click_threshold,
        )

    def update(
        self,
        class_probabilities: np.ndarray,
        class_names: List[str],
    ) -> Dict:
        """Process one classification result and update cursor.

        This is the main entry point called once per control loop
        iteration (~16 Hz).

        Args:
            class_probabilities: 1-D probability array from predict_proba,
                shape (n_classes,).
            class_names: Class name strings in label order.

        Returns:
            Dictionary with status information:
                - 'direction': active direction or None
                - 'confidence': winning class probability
                - 'velocity': (dx, dy) pixel velocity applied
                - 'click_event': 'click', 'double_click', or None
                - 'predicted_class': name of the winning class
        """
        # 1. Get direction and confidence
        direction, confidence = ControlMapper.mi_to_direction(
            class_probabilities,
            class_names,
            self._direction_map,
            threshold=self._move_threshold,
        )

        # 2. Compute target velocity
        if direction is not None:
            # Scale velocity by confidence above threshold
            effective = (confidence - self._move_threshold) / (1.0 - self._move_threshold)
            speed = effective * self._max_vel

            dx, dy = 0.0, 0.0
            if direction == "left":
                dx = -speed
            elif direction == "right":
                dx = speed
            elif direction == "up":
                dy = -speed  # Screen Y is inverted
            elif direction == "down":
                dy = speed
        else:
            dx, dy = 0.0, 0.0

        # 3. Apply EMA smoothing
        self._vx = (1.0 - self._alpha) * self._vx + self._alpha * dx
        self._vy = (1.0 - self._alpha) * self._vy + self._alpha * dy

        # 4. Move cursor (only if velocity is meaningful)
        if abs(self._vx) > 0.5 or abs(self._vy) > 0.5:
            self._mouse.move_relative(dx=self._vx, dy=self._vy)
            self.total_movements += 1

        # 5. Check click via sustained classification
        click_event = self._check_click(class_probabilities, class_names)

        # 6. Determine predicted class name for display
        max_idx = int(np.argmax(class_probabilities))
        predicted_class = class_names[max_idx] if max_idx < len(class_names) else "unknown"

        return {
            "direction": direction,
            "confidence": confidence,
            "velocity": (self._vx, self._vy),
            "click_event": click_event,
            "predicted_class": predicted_class,
        }

    def _check_click(
        self,
        class_probabilities: np.ndarray,
        class_names: List[str],
    ) -> Optional[str]:
        """Check for sustained-imagery click.

        A click is triggered when any directional class maintains
        high confidence (>= click_threshold) for >= hold_duration_s.

        Args:
            class_probabilities: Probability array.
            class_names: Class name list.

        Returns:
            'click', 'double_click', or None.
        """
        now = time.monotonic()
        max_idx = int(np.argmax(class_probabilities))
        max_prob = float(class_probabilities[max_idx])
        max_class = class_names[max_idx]

        # Only consider directional classes with high confidence
        is_directional = max_class in self._direction_map
        is_confident = max_prob >= self._click_threshold

        if is_directional and is_confident:
            if self._sustained_class == max_class:
                # Same class sustained — check duration
                elapsed = now - self._sustained_start
                if elapsed >= self._click_hold_duration:
                    # Trigger click if cooldown has passed
                    if (now - self._last_click_time) >= self._click_cooldown:
                        self._last_click_time = now
                        self._sustained_class = None  # Reset to prevent repeat
                        self._sustained_start = 0.0
                        self.total_clicks += 1

                        # Check for double-click
                        if (
                            self._click_count >= 1
                            and (now - self._first_click_time) <= self._double_click_window
                        ):
                            self._click_count = 0
                            self._first_click_time = 0.0
                            self._mouse.double_click()
                            logger.info("DOUBLE CLICK (sustained %s)", max_class)
                            return "double_click"
                        else:
                            self._first_click_time = now
                            self._click_count = 1
                            self._mouse.click()
                            logger.info("CLICK (sustained %s for %.1fs)", max_class, elapsed)
                            return "click"
            else:
                # New class — start tracking
                self._sustained_class = max_class
                self._sustained_start = now
        else:
            # Not a confident directional class — reset sustained tracking
            self._sustained_class = None
            self._sustained_start = 0.0

        # Reset double-click tracking if too much time passed
        if (
            self._click_count >= 1
            and (now - self._first_click_time) > self._double_click_window
        ):
            self._click_count = 0

        return None

    def reset(self) -> None:
        """Reset all state (new session)."""
        self._mapper.reset()
        self._vx = 0.0
        self._vy = 0.0
        self._sustained_class = None
        self._sustained_start = 0.0
        self._last_click_time = 0.0
        self._click_count = 0
        self._first_click_time = 0.0
        self.total_movements = 0
        self.total_clicks = 0

    @property
    def position(self) -> Tuple[int, int]:
        """Current cursor position."""
        return self._mouse.get_position()

    @property
    def screen_size(self) -> Tuple[int, int]:
        """Screen dimensions."""
        return self._mouse.get_screen_size()

    def __repr__(self) -> str:
        return (
            f"EEGCursorController(directions={self._direction_map}, "
            f"move_thresh={self._move_threshold}, "
            f"click_hold={self._click_hold_duration}s)"
        )
