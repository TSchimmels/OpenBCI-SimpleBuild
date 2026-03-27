"""Event-Related Potential accumulator for real-time MI training.

Maintains running averages of EEG epochs time-locked to cue events,
computes baseline-corrected ERPs, trial-level statistics, and
signed-r² discriminability maps between motor imagery classes.

References:
    Luck, S. J. (2014). An Introduction to the Event-Related
    Potential Technique. MIT Press.

    Blankertz, B., Lemm, S., Treder, M., Haufe, S., & Müller, K.-R.
    (2011). Single-trial analysis and classification of ERP components.
    NeuroImage, 56(2), 814-825.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ERPAccumulator:
    """Accumulates EEG epochs and computes running ERP averages.

    Maintains per-class epoch buffers and provides baseline-corrected
    grand averages, single-trial overlays, and signed-r² maps for
    real-time neurofeedback during MI training.

    Args:
        n_channels: Number of EEG channels.
        n_samples: Samples per epoch (e.g., 750 for 6s at 125 Hz).
        sf: Sampling frequency in Hz.
        baseline_samples: Number of samples in the pre-stimulus baseline
            (from the start of the epoch). Default 125 (1.0s at 125 Hz).
        class_names: List of class name strings.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        sf: int = 125,
        baseline_samples: int = 125,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sf = sf
        self.baseline_samples = baseline_samples
        self.class_names = class_names or []

        # Per-class epoch storage: {class_name: list of (n_channels, n_samples)}
        self._epochs: Dict[str, List[np.ndarray]] = {
            name: [] for name in self.class_names
        }

        # Time vector (seconds relative to cue onset)
        # Baseline is at the start: t < 0 is pre-stimulus
        self.times = np.arange(n_samples) / sf - (baseline_samples / sf)

        logger.info(
            "ERPAccumulator: %d ch, %d samples, sf=%d Hz, "
            "baseline=%d samples (%.2f s), classes=%s",
            n_channels, n_samples, sf, baseline_samples,
            baseline_samples / sf, self.class_names,
        )

    # ------------------------------------------------------------------
    # Epoch management
    # ------------------------------------------------------------------

    def add_epoch(self, epoch: np.ndarray, class_name: str) -> None:
        """Add a single epoch to the accumulator.

        Args:
            epoch: EEG epoch, shape (n_channels, n_samples).
            class_name: Class label for this epoch.
        """
        if epoch.shape != (self.n_channels, self.n_samples):
            logger.warning(
                "Epoch shape %s does not match expected (%d, %d). Skipping.",
                epoch.shape, self.n_channels, self.n_samples,
            )
            return

        if class_name not in self._epochs:
            self._epochs[class_name] = []
            if class_name not in self.class_names:
                self.class_names.append(class_name)

        self._epochs[class_name].append(epoch.copy())
        logger.debug(
            "Added epoch for '%s' (total: %d)",
            class_name, len(self._epochs[class_name]),
        )

    def get_trial_count(self, class_name: Optional[str] = None) -> int:
        """Return number of accumulated trials."""
        if class_name is not None:
            return len(self._epochs.get(class_name, []))
        return sum(len(v) for v in self._epochs.values())

    # ------------------------------------------------------------------
    # ERP computation
    # ------------------------------------------------------------------

    def get_erp(
        self, class_name: str, baseline_correct: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the average ERP for a class.

        Args:
            class_name: Which class to average.
            baseline_correct: If True, subtract the mean of the baseline
                period from each channel.

        Returns:
            (mean_erp, std_erp) each of shape (n_channels, n_samples).
            Returns zeros if no epochs are available.
        """
        epochs = self._epochs.get(class_name, [])
        if len(epochs) == 0:
            zeros = np.zeros((self.n_channels, self.n_samples))
            return zeros, zeros

        stacked = np.stack(epochs, axis=0)  # (n_trials, n_ch, n_samples)
        mean_erp = stacked.mean(axis=0)
        std_erp = stacked.std(axis=0)

        if baseline_correct:
            bl = mean_erp[:, :self.baseline_samples].mean(axis=1, keepdims=True)
            mean_erp = mean_erp - bl
            # std doesn't change with baseline subtraction

        return mean_erp, std_erp

    def get_grand_average(
        self, baseline_correct: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute grand average across all classes.

        Returns:
            (mean_erp, std_erp) each of shape (n_channels, n_samples).
        """
        all_epochs = []
        for class_epochs in self._epochs.values():
            all_epochs.extend(class_epochs)

        if len(all_epochs) == 0:
            zeros = np.zeros((self.n_channels, self.n_samples))
            return zeros, zeros

        stacked = np.stack(all_epochs, axis=0)
        mean_erp = stacked.mean(axis=0)
        std_erp = stacked.std(axis=0)

        if baseline_correct:
            bl = mean_erp[:, :self.baseline_samples].mean(axis=1, keepdims=True)
            mean_erp = mean_erp - bl

        return mean_erp, std_erp

    def get_last_trial(self, class_name: str) -> Optional[np.ndarray]:
        """Return the most recent single trial for a class."""
        epochs = self._epochs.get(class_name, [])
        if len(epochs) == 0:
            return None
        return epochs[-1].copy()

    def get_all_trials(self, class_name: str) -> Optional[np.ndarray]:
        """Return all trials stacked: (n_trials, n_channels, n_samples)."""
        epochs = self._epochs.get(class_name, [])
        if len(epochs) == 0:
            return None
        return np.stack(epochs, axis=0)

    # ------------------------------------------------------------------
    # Discriminability analysis
    # ------------------------------------------------------------------

    def compute_signed_r2(
        self,
        class_a: str,
        class_b: str,
    ) -> np.ndarray:
        """Compute signed r² between two classes at each (channel, time).

        Signed r² = sign(mean_A - mean_B) * r²

        This shows WHERE and WHEN the brain signals differ between two
        motor imagery classes. Positive values mean class A has higher
        amplitude; negative means class B is higher.

        Args:
            class_a: First class name.
            class_b: Second class name.

        Returns:
            Signed r² map, shape (n_channels, n_samples).
            Values range from -1 to +1.
        """
        epochs_a = self._epochs.get(class_a, [])
        epochs_b = self._epochs.get(class_b, [])

        if len(epochs_a) < 2 or len(epochs_b) < 2:
            return np.zeros((self.n_channels, self.n_samples))

        X_a = np.stack(epochs_a, axis=0)  # (n_a, n_ch, n_samples)
        X_b = np.stack(epochs_b, axis=0)  # (n_b, n_ch, n_samples)

        n_a, n_b = X_a.shape[0], X_b.shape[0]
        n_total = n_a + n_b

        mean_a = X_a.mean(axis=0)
        mean_b = X_b.mean(axis=0)
        mean_all = (X_a.sum(axis=0) + X_b.sum(axis=0)) / n_total

        # Between-group sum of squares
        ss_between = (
            n_a * (mean_a - mean_all) ** 2
            + n_b * (mean_b - mean_all) ** 2
        )

        # Total sum of squares
        X_all = np.concatenate([X_a, X_b], axis=0)
        ss_total = ((X_all - mean_all[np.newaxis]) ** 2).sum(axis=0)

        # r²
        r2 = np.where(ss_total > 0, ss_between / ss_total, 0.0)

        # Sign: positive where class_a > class_b
        sign = np.sign(mean_a - mean_b)
        signed_r2 = sign * r2

        return signed_r2

    def compute_erp_snr(self, class_name: str) -> np.ndarray:
        """Compute signal-to-noise ratio of the ERP per channel.

        SNR = mean(post-stimulus amplitude) / std(baseline amplitude)

        Higher SNR means the subject's ERP is more consistent and
        detectable. Useful for feedback: "Your signal on C3 is strong."

        Args:
            class_name: Class to analyze.

        Returns:
            SNR per channel, shape (n_channels,).
        """
        mean_erp, std_erp = self.get_erp(class_name, baseline_correct=True)

        # Post-stimulus: from baseline end to epoch end
        post_stim = mean_erp[:, self.baseline_samples:]
        baseline_noise = std_erp[:, :self.baseline_samples]

        signal_power = np.abs(post_stim).mean(axis=1)
        noise_power = baseline_noise.mean(axis=1)
        noise_power = np.where(noise_power > 0, noise_power, 1e-10)

        return signal_power / noise_power

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self, class_name: Optional[str] = None) -> None:
        """Clear accumulated epochs."""
        if class_name is not None:
            self._epochs[class_name] = []
        else:
            for key in self._epochs:
                self._epochs[key] = []

    def get_epoch_times(self) -> np.ndarray:
        """Return time vector in seconds relative to cue onset."""
        return self.times.copy()

    def __repr__(self) -> str:
        counts = {k: len(v) for k, v in self._epochs.items()}
        return f"ERPAccumulator(channels={self.n_channels}, trials={counts})"
