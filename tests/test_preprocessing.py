"""Tests for the preprocessing module (filters and artifact handling).

Covers bandpass filtering, notch filtering, common average reference,
epoch rejection, and bad channel detection using synthetic signals.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from numpy.fft import rfft, rfftfreq

from src.preprocessing.filters import bandpass_filter, notch_filter, common_average_reference
from src.preprocessing.artifacts import reject_epochs, detect_bad_channels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_rate():
    """Standard sampling rate used across tests (Hz)."""
    return 250


@pytest.fixture
def duration():
    """Signal duration in seconds."""
    return 4.0


@pytest.fixture
def composite_signal(sample_rate, duration):
    """Synthetic signal: 5 Hz + 20 Hz + 80 Hz sine waves.

    Returns (signal, time_vector).
    """
    t = np.arange(0, duration, 1.0 / sample_rate)
    signal = (
        1.0 * np.sin(2 * np.pi * 5 * t)
        + 1.0 * np.sin(2 * np.pi * 20 * t)
        + 1.0 * np.sin(2 * np.pi * 80 * t)
    )
    return signal, t


# ---------------------------------------------------------------------------
# Bandpass filter tests
# ---------------------------------------------------------------------------

class TestBandpassFilter:

    def test_bandpass_filter(self, composite_signal, sample_rate):
        """Bandpass 8-30 Hz should preserve 20 Hz and attenuate 5 Hz and 80 Hz."""
        signal, t = composite_signal
        filtered = bandpass_filter(signal, sf=sample_rate, low=8.0, high=30.0)

        # Verify shape preserved
        assert filtered.shape == signal.shape

        # Compute FFT magnitudes
        n = len(filtered)
        freqs = rfftfreq(n, d=1.0 / sample_rate)
        magnitude = np.abs(rfft(filtered)) / n

        # Helper: get magnitude at a target frequency (nearest bin)
        def mag_at(freq_hz):
            idx = np.argmin(np.abs(freqs - freq_hz))
            return magnitude[idx]

        mag_5 = mag_at(5.0)
        mag_20 = mag_at(20.0)
        mag_80 = mag_at(80.0)

        # 20 Hz (in passband) should be much stronger than 5 Hz and 80 Hz
        assert mag_20 > 10 * mag_5, (
            f"20 Hz ({mag_20:.4f}) should dominate over 5 Hz ({mag_5:.4f})"
        )
        assert mag_20 > 10 * mag_80, (
            f"20 Hz ({mag_20:.4f}) should dominate over 80 Hz ({mag_80:.4f})"
        )

    def test_bandpass_causal(self, composite_signal, sample_rate):
        """Causal bandpass should produce output of same shape."""
        signal, _ = composite_signal
        filtered = bandpass_filter(
            signal, sf=sample_rate, low=8.0, high=30.0, causal=True
        )
        assert filtered.shape == signal.shape
        # Causal filter introduces phase shift, but output must still exist
        # and not be all zeros
        assert np.any(filtered != 0), "Causal filter output should not be all zeros"


# ---------------------------------------------------------------------------
# Notch filter tests
# ---------------------------------------------------------------------------

class TestNotchFilter:

    def test_notch_filter(self, sample_rate, duration):
        """Notch at 60 Hz should attenuate a 60 Hz component."""
        t = np.arange(0, duration, 1.0 / sample_rate)
        # Signal: 10 Hz (keep) + 60 Hz (remove)
        signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t)

        filtered = notch_filter(signal, sf=sample_rate, freq=60.0)

        # FFT analysis
        n = len(filtered)
        freqs = rfftfreq(n, d=1.0 / sample_rate)
        magnitude = np.abs(rfft(filtered)) / n

        idx_10 = np.argmin(np.abs(freqs - 10.0))
        idx_60 = np.argmin(np.abs(freqs - 60.0))

        mag_10 = magnitude[idx_10]
        mag_60 = magnitude[idx_60]

        # 60 Hz should be strongly attenuated relative to 10 Hz
        assert mag_10 > 10 * mag_60, (
            f"10 Hz ({mag_10:.4f}) should dominate over notched 60 Hz ({mag_60:.4f})"
        )


# ---------------------------------------------------------------------------
# Common average reference tests
# ---------------------------------------------------------------------------

class TestCommonAverageReference:

    def test_common_average_reference(self, rng):
        """After CAR, mean across channels at each time point should be ~0."""
        n_channels, n_samples = 4, 500
        data = rng.standard_normal((n_channels, n_samples)) * 50  # microvolts

        referenced = common_average_reference(data)

        # Verify shape preserved
        assert referenced.shape == data.shape

        # Mean across channels at each time point should be near zero
        channel_mean = referenced.mean(axis=0)
        assert np.allclose(channel_mean, 0, atol=1e-10), (
            f"Channel mean should be ~0 everywhere, max deviation: "
            f"{np.max(np.abs(channel_mean)):.2e}"
        )


# ---------------------------------------------------------------------------
# Epoch rejection tests
# ---------------------------------------------------------------------------

class TestRejectEpochs:

    def test_reject_epochs(self, rng):
        """Epoch with high amplitude should be rejected."""
        n_epochs, n_channels, n_samples = 10, 4, 250
        # Create normal epochs with amplitude well within threshold
        epochs = rng.standard_normal((n_epochs, n_channels, n_samples)) * 10
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        # Make epoch index 3 have a spike that exceeds the 100 uV threshold
        epochs[3, 0, 100] = 200.0  # peak-to-peak will be ~200 - (-~30) > 100

        clean_epochs, clean_labels, rejected = reject_epochs(
            epochs, labels, threshold_uv=100.0
        )

        # Epoch 3 should be rejected
        assert 3 in rejected, f"Epoch 3 should be rejected, got rejected={rejected}"
        assert clean_epochs.shape[0] == n_epochs - len(rejected)
        assert clean_labels.shape[0] == clean_epochs.shape[0]
        # Remaining epochs should still have correct dimensionality
        assert clean_epochs.ndim == 3
        assert clean_epochs.shape[1] == n_channels
        assert clean_epochs.shape[2] == n_samples


# ---------------------------------------------------------------------------
# Bad channel detection tests
# ---------------------------------------------------------------------------

class TestDetectBadChannels:

    def test_detect_bad_channels(self, rng):
        """Flatline and excessively noisy channels should be detected."""
        n_channels, n_samples = 8, 1000

        # Create normal channels with moderate variance
        data = rng.standard_normal((n_channels, n_samples)) * 10.0

        # Channel 2: flatline (near-zero variance)
        data[2, :] = 0.0

        # Channel 6: very noisy (high variance)
        data[6, :] = rng.standard_normal(n_samples) * 500.0

        bad = detect_bad_channels(data, threshold_std=3.0)

        assert 2 in bad, f"Flatline channel 2 should be detected as bad, got {bad}"
        assert 6 in bad, f"Noisy channel 6 should be detected as bad, got {bad}"
        # Normal channels should not be flagged
        for ch in [0, 1, 3, 4, 5, 7]:
            assert ch not in bad, f"Normal channel {ch} should not be flagged, got {bad}"
