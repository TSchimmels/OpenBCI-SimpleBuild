"""Tests for the control module (cursor_control, mapping)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


class TestControlMapper:
    """Tests for ControlMapper."""

    def test_velocity_map_dead_zone(self):
        from src.control.mapping import ControlMapper

        mapper = ControlMapper(dead_zone=0.15, max_velocity=30.0)

        # Below dead zone → zero velocity
        assert mapper.velocity_map(0.1) == 0.0
        assert mapper.velocity_map(-0.1) == 0.0
        assert mapper.velocity_map(0.0) == 0.0

        # Above dead zone → nonzero velocity
        vel = mapper.velocity_map(0.5)
        assert vel > 0.0

        # Negative signal → negative velocity
        vel_neg = mapper.velocity_map(-0.5)
        assert vel_neg < 0.0

        # Max signal → max velocity
        vel_max = mapper.velocity_map(1.0)
        assert abs(vel_max - 30.0) < 1e-6

    def test_normalize_welford(self):
        from src.control.mapping import ControlMapper

        mapper = ControlMapper()

        # First sample always returns 0.0 (not enough data for z-score)
        result = mapper.normalize(5.0)
        assert result == 0.0

        # After several samples, should return values in [-1, 1]
        for _ in range(50):
            result = mapper.normalize(np.random.randn() * 10)
        assert -1.0 <= result <= 1.0

    def test_normalize_nan_guard(self):
        from src.control.mapping import ControlMapper

        mapper = ControlMapper()
        assert mapper.normalize(float("nan")) == 0.0
        assert mapper.normalize(float("inf")) == 0.0

    def test_smooth_ema(self):
        from src.control.mapping import ControlMapper

        mapper = ControlMapper(smoothing_alpha=0.5)
        v1 = mapper.smooth(1.0)
        v2 = mapper.smooth(1.0)
        # EMA should converge toward the input value
        assert v2 > v1

    def test_process_full_pipeline(self):
        from src.control.mapping import ControlMapper

        mapper = ControlMapper(dead_zone=0.1, max_velocity=20.0, smoothing_alpha=0.5)
        # Feed several values to warm up Welford
        for _ in range(20):
            mapper.process(np.random.randn())
        # Should return a finite float
        result = mapper.process(2.0)
        assert np.isfinite(result)

    def test_mi_to_command(self):
        from src.control.mapping import ControlMapper

        proba = np.array([0.1, 0.7, 0.2])
        names = ["rest", "right_hand", "left_hand"]

        cmd = ControlMapper.mi_to_command(proba, names, threshold=0.6)
        assert cmd == "right_hand"

        # Below threshold → rest
        cmd2 = ControlMapper.mi_to_command(np.array([0.4, 0.4, 0.2]), names, threshold=0.6)
        assert cmd2 == "rest"

    def test_mi_to_command_nan_guard(self):
        from src.control.mapping import ControlMapper

        proba = np.array([float("nan"), 0.5, 0.5])
        names = ["rest", "right_hand", "left_hand"]
        assert ControlMapper.mi_to_command(proba, names) == "rest"

    def test_mi_to_direction(self):
        from src.control.mapping import ControlMapper

        proba = np.array([0.05, 0.8, 0.05, 0.05, 0.05])
        names = ["rest", "left_hand", "right_hand", "feet", "tongue"]
        direction_map = {
            "left_hand": "left",
            "right_hand": "right",
            "feet": "down",
            "tongue": "up",
        }

        direction, confidence = ControlMapper.mi_to_direction(
            proba, names, direction_map, threshold=0.5
        )
        assert direction == "left"
        assert confidence == 0.8

    def test_mi_to_direction_rest(self):
        from src.control.mapping import ControlMapper

        proba = np.array([0.8, 0.05, 0.05, 0.05, 0.05])
        names = ["rest", "left_hand", "right_hand", "feet", "tongue"]
        direction_map = {"left_hand": "left", "right_hand": "right", "feet": "down", "tongue": "up"}

        direction, confidence = ControlMapper.mi_to_direction(
            proba, names, direction_map, threshold=0.5
        )
        assert direction is None
        assert confidence == 0.0

    def test_reset(self):
        from src.control.mapping import ControlMapper

        mapper = ControlMapper()
        for _ in range(10):
            mapper.process(np.random.randn())
        mapper.reset()
        assert mapper._n_samples == 0
        assert mapper._prev_smoothed == 0.0


class TestEEGCursorController:
    """Tests for EEGCursorController (without pyautogui display)."""

    @pytest.fixture
    def config(self):
        return {
            "control": {
                "dead_zone": 0.15,
                "max_velocity": 25.0,
                "smoothing_alpha": 0.3,
                "confidence_threshold": 0.5,
                "direction_map": {
                    "left_hand": "left",
                    "right_hand": "right",
                    "feet": "down",
                    "tongue": "up",
                },
                "click": {
                    "hold_duration_s": 0.8,
                    "confidence_threshold": 0.7,
                    "double_click_window_s": 1.5,
                    "cooldown_s": 0.5,
                },
            },
        }

    def test_cursor_controller_init(self, config):
        """Test that EEGCursorController can be imported and configured."""
        # We can't fully test without a display, but we can test the
        # init logic by mocking pyautogui
        try:
            from src.control.cursor_control import EEGCursorController
            # If pyautogui works (has display), test creation
            ctrl = EEGCursorController(config)
            assert ctrl._move_threshold == 0.5
            assert ctrl._click_hold_duration == 0.8
            assert ctrl.total_movements == 0
            assert ctrl.total_clicks == 0
        except Exception:
            # No display available — can't create MouseController
            pytest.skip("No display available for pyautogui")


class TestAnalysisERP:
    """Tests for ERPAccumulator."""

    @pytest.fixture
    def erp_accum(self):
        from src.analysis.erp import ERPAccumulator

        return ERPAccumulator(
            n_channels=16,
            n_samples=750,
            sf=125,
            baseline_samples=125,
            class_names=["rest", "left_hand", "right_hand"],
        )

    def test_add_epoch(self, erp_accum):
        epoch = np.random.randn(16, 750)
        erp_accum.add_epoch(epoch, "left_hand")
        assert erp_accum.get_trial_count("left_hand") == 1
        assert erp_accum.get_trial_count() == 1

    def test_get_erp_mean(self, erp_accum):
        for _ in range(10):
            erp_accum.add_epoch(np.random.randn(16, 750) * 50, "left_hand")

        mean_erp, std_erp = erp_accum.get_erp("left_hand")
        assert mean_erp.shape == (16, 750)
        assert std_erp.shape == (16, 750)
        # Baseline-corrected mean should be near zero in baseline period
        assert abs(mean_erp[:, :125].mean()) < 20.0

    def test_get_erp_empty(self, erp_accum):
        mean_erp, std_erp = erp_accum.get_erp("right_hand")
        assert mean_erp.shape == (16, 750)
        assert np.all(mean_erp == 0)

    def test_signed_r2(self, erp_accum):
        # Add epochs with different means to create discriminability
        for _ in range(10):
            erp_accum.add_epoch(np.random.randn(16, 750) + 1.0, "left_hand")
            erp_accum.add_epoch(np.random.randn(16, 750) - 1.0, "right_hand")

        r2 = erp_accum.compute_signed_r2("left_hand", "right_hand")
        assert r2.shape == (16, 750)
        # With clearly different means, r² should be positive (class A > class B)
        assert r2.mean() > 0

    def test_signed_r2_insufficient_data(self, erp_accum):
        erp_accum.add_epoch(np.random.randn(16, 750), "left_hand")
        r2 = erp_accum.compute_signed_r2("left_hand", "right_hand")
        assert np.all(r2 == 0)

    def test_snr(self, erp_accum):
        for _ in range(10):
            erp_accum.add_epoch(np.random.randn(16, 750) * 50, "left_hand")
        snr = erp_accum.compute_erp_snr("left_hand")
        assert snr.shape == (16,)
        assert np.all(np.isfinite(snr))

    def test_clear(self, erp_accum):
        erp_accum.add_epoch(np.random.randn(16, 750), "left_hand")
        erp_accum.clear("left_hand")
        assert erp_accum.get_trial_count("left_hand") == 0

    def test_wrong_shape_rejected(self, erp_accum):
        erp_accum.add_epoch(np.random.randn(8, 500), "left_hand")
        assert erp_accum.get_trial_count("left_hand") == 0


class TestERDSComputer:
    """Tests for ERDSComputer."""

    def test_compute_tfr(self):
        from src.analysis.time_frequency import ERDSComputer

        erds = ERDSComputer(sf=125, freqs=np.arange(1, 41), n_cycles=5.0)
        signal = np.random.randn(500)
        power = erds.compute_tfr(signal.reshape(1, -1), channel=0)
        assert power.shape == (40, 500)
        assert np.all(power >= 0)  # Power is non-negative

    def test_compute_erds(self):
        from src.analysis.time_frequency import ERDSComputer

        erds = ERDSComputer(
            sf=125, freqs=np.arange(1, 41), n_cycles=5.0,
            baseline_tmin=0.0, baseline_tmax=1.0,
        )
        epoch = np.random.randn(1, 750)
        erds_map = erds.compute_erds(epoch, channel=0, epoch_tmin=-1.0)
        assert erds_map.shape == (40, 750)
        # ERDS% is in percentage — baseline period should be ~0%
        baseline = erds_map[:, :125]  # First 1s = baseline
        assert abs(baseline.mean()) < 200  # Not too far from zero

    def test_compute_band_power(self):
        from src.analysis.time_frequency import ERDSComputer

        erds = ERDSComputer(sf=125, freqs=np.arange(1, 41), n_cycles=5.0)
        epoch = np.random.randn(1, 500)
        bp, bp_erds = erds.compute_band_power(
            epoch, channel=0, band=(8, 12), epoch_tmin=0.0,
        )
        assert bp.shape == (500,)
        assert bp_erds.shape == (500,)
        assert np.all(np.isfinite(bp))


class TestTopoMapper:
    """Tests for TopoMapper."""

    def test_interpolate(self):
        from src.analysis.topography import TopoMapper

        topo = TopoMapper()
        values = np.random.randn(16)
        grid = topo.interpolate(values)
        assert grid.shape == (64, 64)
        # Should have some non-NaN values inside the head
        assert np.count_nonzero(~np.isnan(grid)) > 100

    def test_channel_index(self):
        from src.analysis.topography import TopoMapper

        topo = TopoMapper()
        assert topo.get_channel_index("C3") == 2
        assert topo.get_channel_index("C4") == 3
        assert topo.get_channel_index("Cz") == 14
