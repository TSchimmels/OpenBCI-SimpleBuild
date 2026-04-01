"""Tests for synthetic data generation."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_synthetic_data import generate_mi_epoch


class TestGenerateMIEpoch:
    """Test the per-epoch MI signal generator."""

    def setup_method(self):
        self.n_channels = 16
        self.n_samples = 625
        self.sf = 250
        self.rng = np.random.RandomState(42)

    def test_output_shape(self):
        epoch = generate_mi_epoch("left_hand", self.n_channels, self.n_samples, self.sf, self.rng)
        assert epoch.shape == (self.n_channels, self.n_samples)

    def test_all_classes_produce_output(self):
        for cls in ["rest", "left_hand", "right_hand", "feet", "tongue"]:
            epoch = generate_mi_epoch(cls, self.n_channels, self.n_samples, self.sf, self.rng)
            assert epoch.shape == (self.n_channels, self.n_samples)
            assert np.all(np.isfinite(epoch))

    def test_classes_differ(self):
        """Different classes should produce different patterns."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        left = generate_mi_epoch("left_hand", self.n_channels, self.n_samples, self.sf, rng1)
        right = generate_mi_epoch("right_hand", self.n_channels, self.n_samples, self.sf, rng2)
        # C3 (ch 0) and C4 (ch 1) should differ between left and right
        assert not np.allclose(left[0], right[0])

    def test_rest_has_no_class_pattern(self):
        """Rest class should be pure noise + background alpha."""
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        rest = generate_mi_epoch("rest", self.n_channels, self.n_samples, self.sf, rng1)
        # Rest should still have signal (noise + alpha), not zeros
        assert np.std(rest) > 1.0

    def test_left_hand_suppresses_c4(self):
        """Left hand MI should suppress mu at C4 (channel 1)."""
        epochs_left = []
        epochs_rest = []
        for seed in range(10):
            rng = np.random.RandomState(seed)
            epochs_left.append(generate_mi_epoch("left_hand", self.n_channels, self.n_samples, self.sf, rng))
            rng = np.random.RandomState(seed)
            epochs_rest.append(generate_mi_epoch("rest", self.n_channels, self.n_samples, self.sf, rng))
        # Average power at C4 should be lower for left_hand than rest
        power_left_c4 = np.mean([np.var(e[1]) for e in epochs_left])
        power_rest_c4 = np.mean([np.var(e[1]) for e in epochs_rest])
        assert power_left_c4 < power_rest_c4
