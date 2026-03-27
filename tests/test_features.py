"""Tests for feature extraction modules (CSP, chaos, band power).

Covers CSPExtractor, ChaosFeatureExtractor, and BandPowerExtractor
using synthetic EEG-like data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.features.csp import CSPExtractor
from src.features.chaos import ChaosFeatureExtractor
from src.features.bandpower import BandPowerExtractor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_rate():
    """Sampling rate in Hz."""
    return 250


# ---------------------------------------------------------------------------
# CSP tests
# ---------------------------------------------------------------------------

class TestCSPExtractor:

    def test_csp_extractor(self, rng):
        """CSP should fit on 2-class data and return correct output shape.

        Class 0: more power in the first half of channels.
        Class 1: more power in the second half of channels.
        """
        n_trials = 60
        n_channels = 8
        n_samples = 250
        n_components = 4

        X = rng.standard_normal((n_trials, n_channels, n_samples)).astype(np.float64)
        y = np.array([0] * 30 + [1] * 30)

        # Boost power in first half of channels for class 0
        X[:30, :4, :] *= 3.0
        # Boost power in second half of channels for class 1
        X[30:, 4:, :] *= 3.0

        csp = CSPExtractor(n_components=n_components)
        csp.fit(X, y)
        features = csp.transform(X)

        # Output shape: (n_trials, n_components)
        assert features.shape == (n_trials, n_components), (
            f"Expected shape ({n_trials}, {n_components}), got {features.shape}"
        )

        # Features should be finite (no NaN/Inf)
        assert np.all(np.isfinite(features)), "CSP features contain NaN or Inf"


# ---------------------------------------------------------------------------
# Chaos feature tests
# ---------------------------------------------------------------------------

# Check if antropy is available for conditional skipping
try:
    import antropy
    _ANTROPY_AVAILABLE = True
except ImportError:
    _ANTROPY_AVAILABLE = False


class TestChaosFeatureExtractor:

    @pytest.mark.skipif(not _ANTROPY_AVAILABLE, reason="antropy not installed")
    def test_chaos_features_single_channel(self, rng):
        """Extract all chaos features from a single channel; verify numeric output."""
        all_features = [
            "hjorth", "perm_entropy", "spectral_entropy",
            "higuchi_fd", "petrosian_fd", "katz_fd",
            "svd_entropy", "dfa",
        ]
        sf = 125
        n_samples = 500
        signal = rng.standard_normal(n_samples)

        extractor = ChaosFeatureExtractor(features=all_features, sf=sf)
        result = extractor.extract_single_channel(signal)

        # Expected length: hjorth=2, rest=1 each -> 2 + 7 = 9
        expected_len = extractor.n_features_per_channel
        assert len(result) == expected_len, (
            f"Expected {expected_len} features, got {len(result)}"
        )
        # All values should be numeric and not NaN
        assert np.all(np.isfinite(result)), (
            f"Chaos features contain non-finite values: {result}"
        )

    @pytest.mark.skipif(not _ANTROPY_AVAILABLE, reason="antropy not installed")
    def test_chaos_features_multi_channel(self, rng):
        """Extract chaos features from multiple channels; verify output length."""
        features = ["hjorth", "perm_entropy", "higuchi_fd"]
        sf = 125
        n_channels = 8
        n_samples = 500
        channel_indices = [0, 3, 7]

        data = rng.standard_normal((n_channels, n_samples))
        extractor = ChaosFeatureExtractor(features=features, sf=sf)

        result = extractor.extract_multi_channel(data, channel_indices)

        # Per channel: hjorth=2, perm_entropy=1, higuchi_fd=1 -> 4
        per_channel = extractor.n_features_per_channel
        expected_len = len(channel_indices) * per_channel
        assert len(result) == expected_len, (
            f"Expected {expected_len} features for {len(channel_indices)} channels, "
            f"got {len(result)}"
        )
        assert np.all(np.isfinite(result)), "Multi-channel features contain non-finite values"

    @pytest.mark.skipif(_ANTROPY_AVAILABLE, reason="antropy IS installed; test needs it absent")
    def test_chaos_graceful_degradation(self):
        """When antropy is not available, extractor should return empty arrays."""
        features = ["hjorth", "perm_entropy"]
        extractor = ChaosFeatureExtractor(features=features, sf=125)

        signal = np.random.default_rng(0).standard_normal(500)
        result = extractor.extract_single_channel(signal)

        assert len(result) == 0, (
            f"Without antropy, expected empty feature vector, got length {len(result)}"
        )
        assert extractor.n_features_per_channel == 0


# ---------------------------------------------------------------------------
# Band power tests
# ---------------------------------------------------------------------------

class TestBandPowerExtractor:

    def test_bandpower_extractor(self, sample_rate):
        """Signal with strong 10 Hz component should show more mu power than beta."""
        n_samples = sample_rate * 4  # 4 seconds of data
        t = np.arange(n_samples) / sample_rate

        # Strong 10 Hz (mu band: 8-12 Hz) + weak noise
        signal = 10.0 * np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.default_rng(42).standard_normal(n_samples)

        # Shape: (1 channel, n_samples)
        data = signal.reshape(1, -1)

        bands = {"mu": [8, 12], "beta": [13, 30]}
        extractor = BandPowerExtractor(bands=bands, sf=sample_rate)
        features = extractor.extract(data, channel_indices=[0])

        # Sorted band names: ["beta", "mu"]. Features order per channel:
        # [beta_power, mu_power, ratio]
        # ratio = beta / mu
        beta_power = features[0]
        mu_power = features[1]

        assert mu_power > beta_power, (
            f"Mu power ({mu_power:.4f}) should exceed beta power ({beta_power:.4f}) "
            f"for a 10 Hz dominant signal"
        )

        # Features should all be finite
        assert np.all(np.isfinite(features)), f"Band power features not finite: {features}"
