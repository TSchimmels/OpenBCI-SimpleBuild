"""Band power feature extraction via Welch's PSD method.

Computes spectral power within specified frequency bands for each EEG
channel. The mu (8-12 Hz) and beta (13-30 Hz) band powers — and their
ratio — are the primary features for detecting event-related
desynchronization (ERD) and synchronization (ERS) during motor imagery.

References:
    Pfurtscheller & Lopes da Silva (1999) "Event-related EEG/MEG
    synchronization and desynchronization: basic principles."
    Clinical Neurophysiology.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson

logger = logging.getLogger(__name__)


class BandPowerExtractor:
    """Spectral band power feature extractor.

    Computes absolute band power using Welch's method and derives
    the mu/beta ratio (ERD/ERS indicator) for each channel. The
    ratio is a key discriminative feature: during motor imagery of
    one hand, contralateral mu power decreases (ERD) while ipsilateral
    mu may increase (ERS).

    Attributes:
        bands: Dictionary mapping band name to [low, high] frequency limits.
        sf: Sampling frequency in Hz.
    """

    def __init__(
        self,
        bands: Dict[str, List[float]],
        sf: int = 125,
    ) -> None:
        """Initialize band power extractor.

        Args:
            bands: Frequency bands to compute power for. Each entry maps
                a band name (str) to a [low_hz, high_hz] list. Example:
                {"mu": [8, 12], "beta": [13, 30]}. At least two bands
                are expected for ratio computation.
            sf: Sampling frequency in Hz. Must match the actual data
                rate (125 Hz for Cyton+Daisy).

        Raises:
            ValueError: If bands is empty, or any band has invalid limits.
        """
        if not bands:
            raise ValueError("bands dictionary must not be empty.")

        for name, (low, high) in bands.items():
            if low >= high:
                raise ValueError(
                    f"Band '{name}' has low ({low}) >= high ({high})"
                )
            nyquist = sf / 2.0
            if high > nyquist:
                logger.warning(
                    "Band '%s' upper limit (%.1f Hz) exceeds Nyquist "
                    "(%.1f Hz). Results may be unreliable.",
                    name,
                    high,
                    nyquist,
                )

        self.bands = {name: list(limits) for name, limits in bands.items()}
        self.sf = sf

        self._band_names = sorted(self.bands.keys())

        logger.info(
            "BandPowerExtractor initialized: bands=%s, sf=%d Hz",
            self.bands,
            sf,
        )

    def _compute_psd(
        self,
        signal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute power spectral density using Welch's method.

        Uses a Hann window with 50% overlap. The segment length is
        chosen to provide ~0.5 Hz frequency resolution (2 * sf samples)
        while still being applicable to short BCI windows.

        Args:
            signal: 1D signal, shape (n_samples,).

        Returns:
            freqs: Frequency vector, shape (n_freq_bins,).
            psd: Power spectral density, shape (n_freq_bins,).
                Units are V^2/Hz (or uV^2/Hz depending on input scaling).
        """
        n_samples = len(signal)

        # Guard: empty or single-sample signal
        if n_samples < 2:
            logger.warning(
                "Signal too short for Welch PSD (%d samples). "
                "Returning zero PSD.",
                n_samples,
            )
            return np.array([0.0]), np.array([0.0])

        # nperseg: use 2*sf for ~0.5 Hz resolution, but cap at signal length.
        nperseg = min(2 * self.sf, n_samples)
        # Welch requires nperseg >= 1; also ensure noverlap < nperseg
        nperseg = max(nperseg, 2)

        freqs, psd = welch(
            signal,
            fs=self.sf,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling="density",
        )

        # Guard: NaN in PSD output (can occur with degenerate input)
        if np.any(np.isnan(psd)):
            logger.warning(
                "NaN detected in Welch PSD output; replacing with 0.0."
            )
            psd = np.nan_to_num(psd, nan=0.0)

        return freqs, psd

    def _band_power(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        low: float,
        high: float,
    ) -> float:
        """Integrate PSD within a frequency band.

        Uses Simpson's rule for numerical integration over the PSD
        values that fall within [low, high] Hz.

        Args:
            freqs: Frequency vector from Welch.
            psd: PSD values from Welch.
            low: Lower frequency bound (Hz).
            high: Upper frequency bound (Hz).

        Returns:
            Absolute band power (integrated PSD, units V^2).
        """
        # Select frequency bins within the band.
        idx = np.logical_and(freqs >= low, freqs <= high)
        if not np.any(idx):
            logger.warning(
                "No frequency bins in range [%.1f, %.1f] Hz. "
                "Check sf and signal length.",
                low,
                high,
            )
            return 0.0

        band_freqs = freqs[idx]
        band_psd = psd[idx]

        # Simpson's rule integration.
        if len(band_freqs) < 2:
            # Not enough points for integration; return the single PSD value
            # multiplied by the nominal frequency resolution.
            freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            return float(band_psd[0] * freq_res)

        power = float(simpson(y=band_psd, x=band_freqs))
        return power

    def extract(
        self,
        data: np.ndarray,
        channel_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Extract band power features from multi-channel EEG data.

        For each channel, computes the absolute power in each configured
        band plus the ratio of the first band to the second band. The
        ratio is an ERD/ERS indicator: for {"mu": [8,12], "beta": [13,30]}
        the ratio is mu_power / beta_power.

        Args:
            data: EEG data, shape (n_channels, n_samples).
            channel_indices: Which channels to process. If None, all
                channels are used. Typically motor cortex channels
                (C3, C4, Cz).

        Returns:
            Flat feature vector. For each channel, the features are:
                [band0_power, band1_power, ..., bandN_power, ratio]
            where ratio = first_band / second_band power.
            Full shape: (n_selected_channels * (n_bands + 1),).

            If only one band is configured, the ratio is omitted:
                (n_selected_channels * n_bands,).

        Raises:
            ValueError: If data is not 2D.
            IndexError: If any channel index is out of bounds.
        """
        if data.ndim != 2:
            raise ValueError(
                f"data must be 2D (n_channels, n_samples), got shape {data.shape}"
            )

        # Guard: empty data
        if data.shape[0] == 0 or data.shape[1] == 0:
            logger.warning(
                "BandPowerExtractor.extract received empty data (shape %s); "
                "returning empty feature vector.",
                data.shape,
            )
            return np.array([], dtype=np.float64)

        # Guard: NaN / Inf in input
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            logger.warning(
                "NaN/Inf detected in BandPowerExtractor input; "
                "replacing with zeros."
            )
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if channel_indices is None:
            channel_indices = list(range(data.shape[0]))

        all_features: List[float] = []

        for ch_idx in channel_indices:
            if ch_idx < 0 or ch_idx >= data.shape[0]:
                raise IndexError(
                    f"Channel index {ch_idx} out of bounds for data "
                    f"with {data.shape[0]} channels"
                )

            signal = data[ch_idx]
            freqs, psd = self._compute_psd(signal)

            band_powers: List[float] = []
            for band_name in self._band_names:
                low, high = self.bands[band_name]
                power = self._band_power(freqs, psd, low, high)
                band_powers.append(power)

            all_features.extend(band_powers)

            # Compute ratio if at least two bands are configured.
            # Uses the first two bands in sorted order (beta, mu for
            # {"mu": ..., "beta": ...} -> ratio = beta / mu ... but we
            # want mu/beta for ERD/ERS. So use the original dict order.
            if len(self._band_names) >= 2:
                ratio = self._compute_ratio(band_powers)
                all_features.append(ratio)

        result = np.array(all_features, dtype=np.float64)

        logger.debug(
            "Band power extraction: %d channels -> %d features",
            len(channel_indices),
            len(result),
        )
        return result

    def _compute_ratio(self, band_powers: List[float]) -> float:
        """Compute ERD/ERS ratio from band powers.

        The ratio is defined as first_band / second_band in the sorted
        band name order. For the standard {"mu": [8,12], "beta": [13,30]}
        config, sorted order is ["beta", "mu"], so the ratio is
        beta_power / mu_power. However, the canonical ERD/ERS ratio for
        MI is mu/beta, so we use: band_powers[mu_index] / band_powers[beta_index].

        For simplicity and generality, this returns:
            power[first_sorted_band] / power[second_sorted_band]

        with epsilon to avoid division by zero.

        Args:
            band_powers: Power values in sorted band-name order.

        Returns:
            Power ratio (first / second band).
        """
        eps = 1e-12
        numerator = band_powers[0]
        denominator = band_powers[1]
        ratio = numerator / (denominator + eps)
        # Guard: NaN/Inf from degenerate power values
        if not np.isfinite(ratio):
            logger.warning(
                "Band power ratio is non-finite (num=%.4g, den=%.4g); "
                "returning 0.0.",
                numerator,
                denominator,
            )
            return 0.0
        return ratio

    def get_feature_names(
        self,
        channel_indices: Optional[List[int]] = None,
        n_channels: int = 1,
    ) -> List[str]:
        """Get human-readable names for each output feature.

        Args:
            channel_indices: Channel indices used in extraction. If None,
                uses range(n_channels).
            n_channels: Number of channels (used when channel_indices is None).

        Returns:
            List of feature name strings.
        """
        if channel_indices is None:
            channel_indices = list(range(n_channels))

        names: List[str] = []
        for ch_idx in channel_indices:
            for band_name in self._band_names:
                names.append(f"ch{ch_idx}_{band_name}_power")
            if len(self._band_names) >= 2:
                names.append(
                    f"ch{ch_idx}_{self._band_names[0]}_{self._band_names[1]}_ratio"
                )
        return names

    @property
    def n_features_per_channel(self) -> int:
        """Number of feature values produced per channel."""
        n = len(self._band_names)
        if n >= 2:
            n += 1  # ratio
        return n

    def __repr__(self) -> str:
        return (
            f"BandPowerExtractor(bands={self.bands}, sf={self.sf})"
        )
