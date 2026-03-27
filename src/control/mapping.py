"""Signal-to-cursor mapping: normalization, smoothing, velocity scaling.

Converts raw classifier decision values into pixel-per-frame velocities
and maps multi-class probabilities to discrete commands.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ControlMapper:
    """Maps classifier output to cursor velocity.

    The full pipeline (``process``) runs three stages:

    1. **Normalize** — Welford's online algorithm rescales the raw
       decision value to zero-mean, unit-variance, clipped to [-1, 1].
    2. **Smooth** — Exponential moving average suppresses jitter.
    3. **Velocity map** — Dead-zone + linear scaling converts the
       normalized value to pixels-per-frame.

    For discrete (multi-class) commands use ``mi_to_command`` instead.

    Args:
        dead_zone: Minimum absolute normalized signal to register as
            intentional movement.  Values below this become zero
            velocity.  Default 0.15.
        max_velocity: Maximum cursor speed in pixels per frame.
            Default 30.
        smoothing_alpha: EMA weight on new value (0=max smooth,
            1=no smooth).  Default 0.3.

    Example::

        mapper = ControlMapper()
        vel = mapper.process(raw_decision_value=0.42)
        mouse.move_relative(dx=vel, dy=0)
    """

    def __init__(
        self,
        dead_zone: float = 0.15,
        max_velocity: float = 30.0,
        smoothing_alpha: float = 0.3,
    ) -> None:
        self.dead_zone = dead_zone
        self.max_velocity = max_velocity
        self.smoothing_alpha = smoothing_alpha

        # Exponential moving average state
        self._prev_smoothed: float = 0.0

        # Running statistics (Welford)
        self._running_mean: float = 0.0
        self._M2: float = 0.0
        self._n_samples: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def velocity_map(self, signal: float) -> float:
        """Convert a normalized signal to a pixel velocity.

        Applies a dead-zone to ignore low-amplitude noise, then
        linearly scales the remaining range to ``[0, max_velocity]``.

        Args:
            signal: Normalized signal in the [-1, 1] range.

        Returns:
            Velocity in pixels (positive or negative).
        """
        if abs(signal) < self.dead_zone:
            return 0.0

        effective = (abs(signal) - self.dead_zone) / (1.0 - self.dead_zone)
        vel = effective * self.max_velocity
        return vel if signal > 0 else -vel

    def normalize(self, raw_signal: float) -> float:
        """Normalize using running statistics (Welford's algorithm).

        The running mean and variance are updated incrementally with
        each call.  The output is clipped to [-1, 1].

        Args:
            raw_signal: Raw classifier decision value.

        Returns:
            Normalized value in [-1, 1].
        """
        # Guard: NaN / Inf passthrough to Welford — skip update
        if not np.isfinite(raw_signal):
            logger.warning(
                "NaN/Inf in normalize(raw_signal=%s); returning 0.0 "
                "without updating running statistics.",
                raw_signal,
            )
            return 0.0

        self._n_samples += 1
        n = self._n_samples

        old_mean = self._running_mean
        self._running_mean = old_mean + (raw_signal - old_mean) / n

        if n < 2:
            # Intentional: with only 1 sample, variance is undefined.
            # Return 0.0 (no movement) until we have enough data for
            # a meaningful z-score.  This is safe for downstream code.
            return 0.0

        # Canonical Welford: track sum of squared differences
        self._M2 += (raw_signal - old_mean) * (raw_signal - self._running_mean)

        variance = self._M2 / (n - 1)
        std = max(np.sqrt(abs(variance)), 1e-8)
        normalized = (raw_signal - self._running_mean) / (3.0 * std)
        return float(np.clip(normalized, -1.0, 1.0))

    def smooth(self, value: float) -> float:
        """Apply exponential moving average smoothing.

        result = (1 - alpha) * prev + alpha * value

        Higher alpha = less smoothing (more responsive).
        Lower alpha = more smoothing (more stable).
        """
        result = (1.0 - self.smoothing_alpha) * self._prev_smoothed + self.smoothing_alpha * value
        self._prev_smoothed = result
        return result

    def process(self, raw_decision_value: float) -> float:
        """Full pipeline: normalize, smooth, velocity-map.

        This is the main entry point during the real-time control loop.

        Args:
            raw_decision_value: Raw continuous output from the
                classifier's ``decision_function``.

        Returns:
            Cursor velocity in pixels (signed).
        """
        # Guard: NaN / Inf from classifier — treat as no-movement
        if not np.isfinite(raw_decision_value):
            logger.warning(
                "NaN/Inf received from classifier (value=%s); "
                "treating as 0.0 (no movement).",
                raw_decision_value,
            )
            return 0.0

        # Guard: extremely large values that would corrupt Welford stats
        _MAX_DECISION = 1e6
        if abs(raw_decision_value) > _MAX_DECISION:
            logger.warning(
                "Extremely large decision value (%.4g); clipping to +/-%.0g.",
                raw_decision_value,
                _MAX_DECISION,
            )
            raw_decision_value = float(
                np.clip(raw_decision_value, -_MAX_DECISION, _MAX_DECISION)
            )

        normalized = self.normalize(raw_decision_value)
        smoothed = self.smooth(normalized)
        return self.velocity_map(smoothed)

    @staticmethod
    def mi_to_command(
        class_probabilities: np.ndarray,
        class_names: list,
        threshold: float = 0.6,
    ) -> str:
        """Map multi-class probabilities to a discrete command string.

        Returns the class name whose probability is highest, provided
        that probability exceeds *threshold*.  Otherwise returns
        ``"rest"``.

        Args:
            class_probabilities: 1-D array of shape ``(n_classes,)``
                from ``predict_proba``.
            class_names: List of human-readable class names in the same
                order as the probability vector, e.g.
                ``["rest", "right_hand", "left_hand"]``.
            threshold: Minimum probability required to accept a
                command.  Default 0.6.

        Returns:
            The winning class name, or ``"rest"`` if no class exceeds
            *threshold*.

        Example::

            cmd = mapper.mi_to_command(
                np.array([0.1, 0.8, 0.1]),
                ["rest", "right_hand", "left_hand"],
            )
            # cmd == "right_hand"
        """
        proba = np.asarray(class_probabilities, dtype=np.float64)

        # Guard: NaN / Inf in probabilities
        if np.any(~np.isfinite(proba)):
            logger.warning(
                "NaN/Inf in class probabilities %s; defaulting to 'rest'.",
                proba,
            )
            return "rest"

        # Guard: empty probability array
        if proba.size == 0:
            logger.warning("Empty probability array; defaulting to 'rest'.")
            return "rest"

        max_idx = int(np.argmax(proba))

        if proba[max_idx] >= threshold:
            return class_names[max_idx]
        return "rest"

    @staticmethod
    def mi_to_direction(
        class_probabilities: np.ndarray,
        class_names: list,
        direction_map: dict,
        threshold: float = 0.5,
    ) -> tuple:
        """Map multi-class probabilities to a cursor direction and velocity.

        Returns the direction and confidence-based velocity for the most
        likely non-rest class, provided its probability exceeds *threshold*.

        Args:
            class_probabilities: 1-D array of shape ``(n_classes,)``
                from ``predict_proba``.
            class_names: List of human-readable class names in the same
                order as the probability vector, e.g.
                ``["rest", "left_hand", "right_hand", "feet", "tongue"]``.
            direction_map: Maps class names to directions, e.g.
                ``{"left_hand": "left", "right_hand": "right",
                  "feet": "down", "tongue": "up"}``.
            threshold: Minimum probability required to accept a direction.

        Returns:
            A tuple ``(direction, confidence)`` where:
            - **direction** is one of ``"left"``, ``"right"``, ``"up"``,
              ``"down"``, or ``None`` (no movement / rest).
            - **confidence** is the probability of the winning class
              (0.0 if no movement).
        """
        proba = np.asarray(class_probabilities, dtype=np.float64)

        # Guard: NaN / Inf
        if np.any(~np.isfinite(proba)):
            logger.warning("NaN/Inf in class probabilities; defaulting to no movement.")
            return None, 0.0

        if proba.size == 0:
            logger.warning("Empty probability array; defaulting to no movement.")
            return None, 0.0

        max_idx = int(np.argmax(proba))
        max_prob = float(proba[max_idx])
        max_class = class_names[max_idx]

        # Check if it exceeds threshold and is a directional class
        if max_prob >= threshold and max_class in direction_map:
            return direction_map[max_class], max_prob
        return None, 0.0

    @staticmethod
    def decision_to_scalar(
        decision: np.ndarray,
        class_names: list,
        positive_class: str = "right_hand",
        negative_class: str = "left_hand",
    ) -> float:
        """Convert multi-class decision scores to a single scalar for velocity control.

        For 2-class: returns the raw scalar decision value.
        For multi-class: returns score[positive] - score[negative].

        Args:
            decision: Decision function output, shape (n_classes,) or scalar.
            class_names: Class names in label order.
            positive_class: Class that drives positive velocity (rightward).
            negative_class: Class that drives negative velocity (leftward).

        Returns:
            Scalar control signal (positive = right, negative = left, ~0 = rest).
        """
        decision = np.atleast_1d(decision)

        # Guard: NaN / Inf in decision values
        if np.any(~np.isfinite(decision)):
            logger.warning(
                "NaN/Inf in decision values %s; returning 0.0.",
                decision,
            )
            return 0.0

        if decision.shape[0] == 1:
            return float(decision[0])

        pos_idx = class_names.index(positive_class) if positive_class in class_names else -1
        neg_idx = class_names.index(negative_class) if negative_class in class_names else -1

        if pos_idx >= 0 and neg_idx >= 0:
            return float(decision[pos_idx] - decision[neg_idx])
        elif pos_idx >= 0:
            return float(decision[pos_idx])
        else:
            return 0.0

    def reset(self) -> None:
        """Reset all running statistics and smoothing state.

        Call this when a new session starts or after a long pause.
        """
        self._prev_smoothed = 0.0
        self._running_mean = 0.0
        self._M2 = 0.0
        self._n_samples = 0
