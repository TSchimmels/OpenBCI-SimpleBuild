"""Time-frequency decomposition and ERDS% computation.

Computes Event-Related Desynchronization/Synchronization (ERDS%)
maps using Morlet wavelets or short-time FFT. These maps show how
oscillatory power changes over time and frequency during motor imagery,
revealing the mu (8-12 Hz) ERD that is the primary signal for MI-BCIs.

References:
    Pfurtscheller, G. & Lopes da Silva, F. H. (1999). Event-related
    EEG/MEG synchronization and desynchronization: basic principles.
    Clinical Neurophysiology, 110(11), 1842-1857.

    Graimann, B. & Pfurtscheller, G. (2006). Quantification and
    visualization of event-related changes in oscillatory brain
    activity in the time-frequency domain. Progress in Brain Research.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import morlet2, fftconvolve

logger = logging.getLogger(__name__)


class ERDSComputer:
    """Compute ERDS% time-frequency maps for motor imagery EEG.

    Uses complex Morlet wavelets to decompose EEG epochs into a
    time-frequency representation, then normalizes against a
    pre-stimulus baseline to produce ERDS% values:

        ERDS% = ((power(t,f) - baseline_power(f)) / baseline_power(f)) * 100

    Negative ERDS% = desynchronization (ERD) — power decrease
    Positive ERDS% = synchronization (ERS) — power increase/rebound

    Args:
        sf: Sampling frequency in Hz.
        freqs: Array of frequencies to analyze (Hz).
        n_cycles: Number of wavelet cycles per frequency. Higher values
            give better frequency resolution but worse time resolution.
            Can be a scalar or array matching freqs.
        baseline_tmin: Start of baseline period (seconds, relative to epoch start).
        baseline_tmax: End of baseline period (seconds, relative to epoch start).
    """

    def __init__(
        self,
        sf: int = 125,
        freqs: Optional[np.ndarray] = None,
        n_cycles: float = 5.0,
        baseline_tmin: float = 0.0,
        baseline_tmax: float = 1.0,
    ) -> None:
        self.sf = sf

        if freqs is None:
            # Default: 1 to 40 Hz in 1 Hz steps
            self.freqs = np.arange(1, 41, dtype=np.float64)
        else:
            self.freqs = np.asarray(freqs, dtype=np.float64)

        self.n_cycles = n_cycles
        self.baseline_tmin = baseline_tmin
        self.baseline_tmax = baseline_tmax

        # Pre-compute wavelet kernels
        self._wavelets = self._build_wavelets()

        logger.info(
            "ERDSComputer: sf=%d Hz, freqs=[%.0f-%.0f] Hz (%d), "
            "n_cycles=%.1f, baseline=[%.1f, %.1f]s",
            sf, self.freqs[0], self.freqs[-1], len(self.freqs),
            n_cycles, baseline_tmin, baseline_tmax,
        )

    def _build_wavelets(self) -> List[np.ndarray]:
        """Pre-compute complex Morlet wavelets for each frequency."""
        wavelets = []
        for freq in self.freqs:
            # Number of cycles determines the bandwidth
            if np.isscalar(self.n_cycles):
                n_cyc = self.n_cycles
            else:
                idx = np.argmin(np.abs(self.freqs - freq))
                n_cyc = self.n_cycles[idx]

            # Wavelet width in samples
            sigma_t = n_cyc / (2.0 * np.pi * freq)
            n_samples = int(6 * sigma_t * self.sf)
            if n_samples % 2 == 0:
                n_samples += 1

            # Complex Morlet wavelet
            t = np.arange(n_samples) / self.sf - (n_samples / (2 * self.sf))
            gaussian = np.exp(-t ** 2 / (2 * sigma_t ** 2))
            sinusoid = np.exp(2j * np.pi * freq * t)
            wavelet = gaussian * sinusoid

            # Normalize to unit energy
            wavelet /= np.sqrt(np.sum(np.abs(wavelet) ** 2))

            wavelets.append(wavelet)

        return wavelets

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_tfr(
        self, epoch: np.ndarray, channel: int = 0
    ) -> np.ndarray:
        """Compute time-frequency representation for a single channel.

        Args:
            epoch: EEG epoch, shape (n_channels, n_samples) or (n_samples,).
            channel: Channel index to analyze (ignored if 1D input).

        Returns:
            Power matrix, shape (n_freqs, n_samples). Values are in
            µV²/Hz (or arbitrary units depending on input scaling).
        """
        if epoch.ndim == 2:
            signal = epoch[channel]
        else:
            signal = epoch

        n_samples = len(signal)
        n_freqs = len(self.freqs)
        power = np.zeros((n_freqs, n_samples))

        for i, wavelet in enumerate(self._wavelets):
            # Convolve signal with complex wavelet
            convolved = fftconvolve(signal, wavelet, mode="same")
            # Power = |analytic signal|²
            power[i] = np.abs(convolved) ** 2

        return power

    def compute_erds(
        self,
        epoch: np.ndarray,
        channel: int = 0,
        epoch_tmin: float = -1.0,
    ) -> np.ndarray:
        """Compute ERDS% map for a single trial.

        Args:
            epoch: EEG epoch, shape (n_channels, n_samples) or (n_samples,).
            channel: Channel index.
            epoch_tmin: Time of the first sample relative to cue onset (seconds).
                Used to locate the baseline window within the epoch.

        Returns:
            ERDS% matrix, shape (n_freqs, n_samples).
            Negative = ERD (desynchronization), Positive = ERS (synchronization).
        """
        power = self.compute_tfr(epoch, channel)
        return self._baseline_normalize(power, epoch_tmin)

    def compute_erds_average(
        self,
        epochs: np.ndarray,
        channel: int = 0,
        epoch_tmin: float = -1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute average ERDS% across multiple trials.

        Args:
            epochs: Stacked epochs, shape (n_trials, n_channels, n_samples).
            channel: Channel index.
            epoch_tmin: Time of first sample relative to cue onset.

        Returns:
            (mean_erds, std_erds) each of shape (n_freqs, n_samples).
        """
        n_trials = epochs.shape[0]
        if n_trials == 0:
            n_samples = epochs.shape[-1] if epochs.ndim > 1 else 0
            zeros = np.zeros((len(self.freqs), max(n_samples, 1)))
            return zeros, zeros

        # First compute average power, then normalize
        # (more stable than averaging individual ERDS%)
        all_power = []
        for i in range(n_trials):
            p = self.compute_tfr(epochs[i], channel)
            all_power.append(p)

        stacked_power = np.stack(all_power, axis=0)  # (n_trials, n_freqs, n_samples)
        mean_power = stacked_power.mean(axis=0)
        std_power = stacked_power.std(axis=0)

        mean_erds = self._baseline_normalize(mean_power, epoch_tmin)

        # Propagate std through the normalization
        bl_start, bl_end = self._baseline_indices(mean_power.shape[1], epoch_tmin)
        baseline = mean_power[:, bl_start:bl_end].mean(axis=1, keepdims=True)
        baseline = np.where(baseline > 0, baseline, 1e-10)
        std_erds = (std_power / baseline) * 100

        return mean_erds, std_erds

    # ------------------------------------------------------------------
    # Band power timecourse
    # ------------------------------------------------------------------

    def compute_band_power(
        self,
        epoch: np.ndarray,
        channel: int = 0,
        band: Tuple[float, float] = (8.0, 12.0),
        epoch_tmin: float = -1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute band power timecourse and ERDS% for a frequency band.

        Args:
            epoch: EEG epoch.
            channel: Channel index.
            band: (low_hz, high_hz) frequency band.
            epoch_tmin: Time of first sample relative to cue onset.

        Returns:
            (power_timecourse, erds_timecourse) each of shape (n_samples,).
            power_timecourse is in absolute units.
            erds_timecourse is in % change from baseline.
        """
        power = self.compute_tfr(epoch, channel)

        # Select frequency band
        freq_mask = (self.freqs >= band[0]) & (self.freqs <= band[1])
        if not np.any(freq_mask):
            logger.warning("No frequencies in band [%.1f, %.1f] Hz.", band[0], band[1])
            return np.zeros(power.shape[1]), np.zeros(power.shape[1])

        band_power = power[freq_mask].mean(axis=0)  # (n_samples,)

        # ERDS% for this band
        bl_start, bl_end = self._baseline_indices(len(band_power), epoch_tmin)
        baseline = band_power[bl_start:bl_end].mean()
        baseline = max(baseline, 1e-10)
        erds = ((band_power - baseline) / baseline) * 100

        return band_power, erds

    def compute_band_power_average(
        self,
        epochs: np.ndarray,
        channel: int = 0,
        band: Tuple[float, float] = (8.0, 12.0),
        epoch_tmin: float = -1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute average band power timecourse across trials.

        Returns:
            (mean_power, mean_erds, std_erds) each of shape (n_samples,).
        """
        n_trials = epochs.shape[0]
        if n_trials == 0:
            zeros = np.zeros(1)
            return zeros, zeros, zeros

        all_power = []
        for i in range(n_trials):
            bp, _ = self.compute_band_power(epochs[i], channel, band, epoch_tmin)
            all_power.append(bp)

        stacked = np.stack(all_power, axis=0)  # (n_trials, n_samples)
        mean_power = stacked.mean(axis=0)

        bl_start, bl_end = self._baseline_indices(len(mean_power), epoch_tmin)
        baseline = mean_power[bl_start:bl_end].mean()
        baseline = max(baseline, 1e-10)
        mean_erds = ((mean_power - baseline) / baseline) * 100

        # Per-trial ERDS std
        all_erds = ((stacked - baseline) / baseline) * 100
        std_erds = all_erds.std(axis=0)

        return mean_power, mean_erds, std_erds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _baseline_indices(
        self, n_samples: int, epoch_tmin: float
    ) -> Tuple[int, int]:
        """Convert baseline time window to sample indices."""
        bl_start = max(0, int((self.baseline_tmin - epoch_tmin) * self.sf))
        bl_end = min(n_samples, int((self.baseline_tmax - epoch_tmin) * self.sf))
        if bl_end <= bl_start:
            bl_end = bl_start + 1
        return bl_start, bl_end

    def _baseline_normalize(
        self, power: np.ndarray, epoch_tmin: float
    ) -> np.ndarray:
        """Normalize power to ERDS% using baseline period."""
        bl_start, bl_end = self._baseline_indices(power.shape[1], epoch_tmin)
        baseline = power[:, bl_start:bl_end].mean(axis=1, keepdims=True)
        baseline = np.where(baseline > 0, baseline, 1e-10)
        erds = ((power - baseline) / baseline) * 100
        return erds

    def __repr__(self) -> str:
        return (
            f"ERDSComputer(sf={self.sf}, freqs=[{self.freqs[0]:.0f}-"
            f"{self.freqs[-1]:.0f}] Hz, n_cycles={self.n_cycles})"
        )
