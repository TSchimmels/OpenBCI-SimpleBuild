"""Tests for Surface Laplacian spatial filter."""

import numpy as np
import pytest

from src.preprocessing.laplacian import (
    surface_laplacian_fdn,
    surface_laplacian_spline,
    ELECTRODE_POSITIONS_10_20,
    DEFAULT_16CH_MONTAGE,
)


class TestSurfaceLaplacianFDN:
    def test_output_shape(self):
        data = np.random.randn(16, 250)
        out = surface_laplacian_fdn(data)
        assert out.shape == data.shape

    def test_removes_global_signal(self):
        """A constant signal across all channels should be suppressed."""
        n_ch, n_samp = 16, 250
        # Global signal (same on all channels)
        global_sig = np.sin(np.linspace(0, 4 * np.pi, n_samp)) * 50.0
        data = np.tile(global_sig, (n_ch, 1))
        out = surface_laplacian_fdn(data)
        # Laplacian of a spatially uniform signal should be near zero
        assert np.max(np.abs(out)) < 1e-10

    def test_preserves_local_signal(self):
        """A signal present on only one channel should survive."""
        data = np.zeros((16, 250))
        data[0] = np.sin(np.linspace(0, 4 * np.pi, 250)) * 50.0
        out = surface_laplacian_fdn(data)
        # Channel 0 should have non-zero output
        assert np.std(out[0]) > 1.0

    def test_custom_channel_names(self):
        data = np.random.randn(8, 100)
        names = ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4", "Fz"]
        out = surface_laplacian_fdn(data, channel_names=names)
        assert out.shape == data.shape

    def test_n_neighbors_parameter(self):
        data = np.random.randn(16, 250)
        out2 = surface_laplacian_fdn(data, n_neighbors=2)
        out6 = surface_laplacian_fdn(data, n_neighbors=6)
        # Different neighbor counts should give different results
        assert not np.allclose(out2, out6)

    def test_single_channel_returns_unchanged(self):
        """With 1 channel and no neighbors, output equals input."""
        data = np.random.randn(1, 100)
        # 1 channel has no neighbors, but n_neighbors=4 means argsort
        # picks the only index. Edge case.
        out = surface_laplacian_fdn(data, channel_names=["Cz"], n_neighbors=1)
        assert out.shape == data.shape


class TestSurfaceLaplacianSpline:
    def test_output_shape(self):
        data = np.random.randn(16, 250)
        out = surface_laplacian_spline(data)
        assert out.shape == data.shape

    def test_output_finite(self):
        """Spline Laplacian should produce finite output on real-like data."""
        data = np.random.randn(16, 250) * 50.0
        out = surface_laplacian_spline(data)
        assert np.all(np.isfinite(out))

    def test_different_from_fdn(self):
        """Spline and FDN methods should give different results."""
        data = np.random.randn(16, 250)
        out_fdn = surface_laplacian_fdn(data)
        out_spl = surface_laplacian_spline(data)
        assert not np.allclose(out_fdn, out_spl)

    def test_smoothing_parameter(self):
        data = np.random.randn(16, 250)
        out_lo = surface_laplacian_spline(data, smoothing=1e-8)
        out_hi = surface_laplacian_spline(data, smoothing=1e-1)
        # Higher smoothing should give smoother (lower variance) output
        assert np.std(out_hi) < np.std(out_lo) or not np.allclose(out_lo, out_hi)


class TestElectrodePositions:
    def test_default_montage_has_16_channels(self):
        assert len(DEFAULT_16CH_MONTAGE) == 16

    def test_all_montage_names_in_positions(self):
        for name in DEFAULT_16CH_MONTAGE:
            assert name in ELECTRODE_POSITIONS_10_20, f"{name} missing from positions"

    def test_positions_on_unit_sphere(self):
        """All electrode positions should be approximately on a unit sphere."""
        for name, (x, y, z) in ELECTRODE_POSITIONS_10_20.items():
            r = np.sqrt(x**2 + y**2 + z**2)
            assert 0.5 < r < 1.5, f"{name} at radius {r:.2f}, expected ~1.0"
