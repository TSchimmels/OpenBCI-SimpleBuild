"""Spatial and temporal filters for EEG preprocessing.

Provides bandpass, notch, and spatial reference filters with both
offline (zero-phase) and real-time (causal) modes. The CausalFilterState
class maintains filter state across streaming chunks.
"""

import logging
from typing import List

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, sosfiltfilt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temporal filters
# ---------------------------------------------------------------------------

def bandpass_filter(
    data: np.ndarray,
    sf: int,
    low: float,
    high: float,
    order: int = 4,
    causal: bool = False,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter.

    Args:
        data: EEG data. Shape ``(n_samples,)`` for a single channel or
            ``(n_channels, n_samples)`` for multi-channel.
        sf: Sampling frequency in Hz.
        low: Lower cutoff frequency in Hz.
        high: Upper cutoff frequency in Hz.
        order: Filter order (default 4).
        causal: If False, use zero-phase filtering (``sosfiltfilt``,
            suitable for offline analysis). If True, use causal filtering
            (``sosfilt``, suitable for real-time streaming).

    Returns:
        Filtered data with the same shape as *data*.

    Raises:
        ValueError: If ``low >= high`` or frequencies are outside the
            valid Nyquist range.
    """
    nyq = sf / 2.0
    if low <= 0 or high <= 0:
        raise ValueError("Cutoff frequencies must be positive.")
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high}).")
    if high >= nyq:
        raise ValueError(
            f"high ({high}) must be below Nyquist frequency ({nyq})."
        )

    # --- Edge-case guards ---

    # NaN / Inf guard: replace with zeros to prevent garbage output
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        logger.warning(
            "NaN/Inf detected in bandpass_filter input, replacing with zeros."
        )
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Single-sample or empty input: filter cannot operate
    n_samples = data.shape[-1]
    if n_samples == 0:
        logger.warning("bandpass_filter received empty data; returning as-is.")
        return data
    if n_samples == 1:
        logger.warning(
            "bandpass_filter received single-sample input; returning zeros."
        )
        return np.zeros_like(data)

    # Minimum length for sosfiltfilt: need > 3 * padlen, where padlen
    # defaults to 3 * max(len(b), len(a)) per section.  For a Butterworth
    # of given order, the SOS has (order) sections, and sosfiltfilt
    # requires n_samples > padlen (default = 3 * n_sections * 2).
    # A safe minimum is 3 * (order * 2) + 1 = 6 * order + 1.
    min_samples = 6 * order + 1
    if not causal and n_samples < min_samples:
        logger.warning(
            "Data too short for zero-phase bandpass filter "
            "(%d samples, need >= %d for order %d). "
            "Falling back to causal (single-pass) filtering.",
            n_samples,
            min_samples,
            order,
        )
        causal = True  # fall back to causal to avoid scipy crash

    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")

    if causal:
        return sosfilt(sos, data, axis=-1)
    return sosfiltfilt(sos, data, axis=-1)


def notch_filter(
    data: np.ndarray,
    sf: int,
    freq: float,
    quality: float = 30.0,
    causal: bool = False,
) -> np.ndarray:
    """Apply a notch (band-stop) filter to remove line noise.

    Args:
        data: EEG data. Shape ``(n_samples,)`` for a single channel or
            ``(n_channels, n_samples)`` for multi-channel.
        sf: Sampling frequency in Hz.
        freq: Centre frequency to remove (e.g. 50 or 60 Hz).
        quality: Quality factor that determines the -3 dB bandwidth.
            Higher values give a narrower notch. Default 30.
        causal: If False, use zero-phase filtering (offline).
            If True, use causal filtering (real-time).

    Returns:
        Filtered data with the same shape as *data*.
    """
    # NaN / Inf guard
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        logger.warning(
            "NaN/Inf detected in notch_filter input, replacing with zeros."
        )
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    n_samples = data.shape[-1]
    if n_samples == 0:
        logger.warning("notch_filter received empty data; returning as-is.")
        return data
    if n_samples == 1:
        logger.warning(
            "notch_filter received single-sample input; returning zeros."
        )
        return np.zeros_like(data)

    b, a = iirnotch(freq, quality, fs=sf)

    # Convert b, a to SOS for numerical stability and consistent API
    # iirnotch returns a 2nd-order section directly, so we can pack it
    # into a single SOS row: [b0, b1, b2, 1, a1, a2]
    sos = np.zeros((1, 6))
    sos[0, :3] = b
    sos[0, 3:] = a

    # sosfiltfilt requires at least 7 samples for a 2nd-order notch
    if not causal and n_samples < 7:
        logger.warning(
            "Data too short for zero-phase notch filter (%d samples, "
            "need >= 7). Falling back to causal filtering.",
            n_samples,
        )
        causal = True

    if causal:
        return sosfilt(sos, data, axis=-1)
    return sosfiltfilt(sos, data, axis=-1)


# ---------------------------------------------------------------------------
# Spatial references
# ---------------------------------------------------------------------------

def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Re-reference to the common average.

    Subtracts the mean across all channels at each time point, removing
    global noise that is shared across electrodes.

    Args:
        data: EEG data, shape ``(n_channels, n_samples)``.

    Returns:
        Re-referenced data with the same shape as *data*.
    """
    return data - data.mean(axis=0, keepdims=True)


def laplacian_reference(
    data: np.ndarray,
    channel_idx: int,
    neighbor_indices: List[int],
) -> np.ndarray:
    """Compute a surface Laplacian reference for a single channel.

    The Laplacian enhances focal activity by subtracting the mean of
    surrounding electrodes, acting as a spatial high-pass filter.

    Args:
        data: Multi-channel EEG data, shape ``(n_channels, n_samples)``.
        channel_idx: Index of the target channel.
        neighbor_indices: Indices of the surrounding (neighbor) channels.

    Returns:
        Laplacian-referenced signal for the target channel, shape
        ``(n_samples,)``.

    Raises:
        ValueError: If *neighbor_indices* is empty.
    """
    if len(neighbor_indices) == 0:
        raise ValueError("neighbor_indices must contain at least one index.")

    neighbors_mean = data[neighbor_indices].mean(axis=0)
    return data[channel_idx] - neighbors_mean


# ---------------------------------------------------------------------------
# Stateful causal filter for real-time streaming
# ---------------------------------------------------------------------------

class CausalFilterState:
    """Maintains SOS filter state for chunk-by-chunk causal filtering.

    Use this class in a real-time loop to bandpass-filter successive data
    chunks without transient artefacts at chunk boundaries.

    Example::

        filt = CausalFilterState(sf=250, low=8.0, high=30.0)
        while streaming:
            chunk = board.get_data()      # (n_channels, n_new_samples)
            filtered = filt.apply(chunk)

    Args:
        sf: Sampling frequency in Hz.
        low: Lower cutoff frequency in Hz.
        high: Upper cutoff frequency in Hz.
        order: Butterworth filter order (default 4).
    """

    def __init__(
        self,
        sf: int,
        low: float,
        high: float,
        order: int = 4,
    ) -> None:
        nyq = sf / 2.0
        self.sos: np.ndarray = butter(
            order, [low / nyq, high / nyq], btype="band", output="sos"
        )
        # sosfilt_zi returns shape (n_sections, 2).  We will broadcast
        # it to multi-channel data on the first call to apply().
        self._zi_template: np.ndarray = sosfilt_zi(self.sos)
        self._zi: np.ndarray | None = None

    # -----------------------------------------------------------------
    def apply(self, data_chunk: np.ndarray) -> np.ndarray:
        """Filter a new chunk of data, preserving continuity.

        Args:
            data_chunk: New samples. Shape ``(n_samples,)`` for a single
                channel or ``(n_channels, n_samples)`` for multi-channel.

        Returns:
            Filtered chunk with the same shape as *data_chunk*.
        """
        # Empty chunk guard: return immediately without corrupting state
        if data_chunk.size == 0 or data_chunk.shape[-1] == 0:
            logger.warning(
                "CausalFilterState.apply() received empty chunk (shape %s); "
                "returning as-is.",
                data_chunk.shape,
            )
            return data_chunk

        # NaN / Inf guard
        if np.any(np.isnan(data_chunk)) or np.any(np.isinf(data_chunk)):
            logger.warning(
                "NaN/Inf detected in CausalFilterState input, "
                "replacing with zeros."
            )
            data_chunk = np.nan_to_num(
                data_chunk, nan=0.0, posinf=0.0, neginf=0.0
            )

        if data_chunk.ndim == 1:
            # Single channel: zi shape (n_sections, 2)
            if self._zi is None:
                self._zi = self._zi_template * data_chunk[0]
            filtered, self._zi = sosfilt(
                self.sos, data_chunk, zi=self._zi
            )
            return filtered

        # Multi-channel: zi shape (n_channels, n_sections, 2)
        n_channels = data_chunk.shape[0]
        if self._zi is None:
            # Broadcast template to each channel, scaled by first sample
            self._zi = np.repeat(
                self._zi_template[np.newaxis, :, :], n_channels, axis=0
            )
            for ch in range(n_channels):
                self._zi[ch] = self._zi_template * data_chunk[ch, 0]

        filtered = np.empty_like(data_chunk)
        for ch in range(n_channels):
            filtered[ch], self._zi[ch] = sosfilt(
                self.sos, data_chunk[ch], zi=self._zi[ch]
            )
        return filtered

    # -----------------------------------------------------------------
    def reset(self) -> None:
        """Reset filter state.

        Call this when a new recording session starts or when
        continuity has been broken (e.g. after a pause/resume).
        """
        self._zi = None
