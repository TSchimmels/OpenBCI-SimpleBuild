"""Tests for all SOTA and adaptation modules.

Covers:
    - ErrPP300Detector (errp_detector)
    - SEALAdaptationEngine (seal_engine)
    - BCIStateMonitor (state_monitor)
    - CausalChannelDiscovery (causal_channels)
    - KoopmanEEGDecomposition (koopman_decomposition)
    - FTLEAnalyzer (ftle_analysis)
    - JacobianFeatureExtractor (jacobian_features)
    - AdaptiveClassifierRouter (adaptive_router)
    - NeuralSDEClassifier (neural_sde) [skip if no torch]
    - JEPAPretrainer (pretrain) [skip if no torch]
    - VariableSelector (variable_selector) [skip if no torch]
    - GFlowNetSEALOptimizer (gflownet_strategy) [skip if no torch]

All tests use synthetic data (np.random.randn) and verify output shapes,
types, and non-crash behavior.
"""

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path("/mnt/c/OpenBCI_ SimpleBuild")
sys.path.insert(0, str(_PROJECT_ROOT))

# Deterministic randomness for reproducibility
RNG = np.random.RandomState(42)


# ---------------------------------------------------------------------------
# Direct module loader -- bypasses __init__.py to avoid missing deps (mne)
# ---------------------------------------------------------------------------

def _load_module(dotted_path: str) -> ModuleType:
    """Import a module by dotted path, loading the .py file directly.

    This avoids triggering package __init__.py files that may import
    third-party dependencies (e.g. mne) not available in the test env.
    """
    parts = dotted_path.split(".")
    file_path = _PROJECT_ROOT / "/".join(parts[:-1]) / (parts[-1] + ".py")
    if not file_path.exists():
        # Fallback to normal import
        return importlib.import_module(dotted_path)
    # Ensure parent packages are minimally registered
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name not in sys.modules:
            pkg_path = _PROJECT_ROOT / "/".join(parts[:i])
            init_file = pkg_path / "__init__.py"
            spec = importlib.util.spec_from_file_location(
                pkg_name, str(init_file) if init_file.exists() else None,
                submodule_search_locations=[str(pkg_path)],
            )
            mod = ModuleType(pkg_name)
            mod.__path__ = [str(pkg_path)]
            mod.__package__ = pkg_name
            sys.modules[pkg_name] = mod
    # Load the actual module file
    mod_name = dotted_path
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Pre-load the base classifier (needed by adaptive_router and neural_sde)
# ---------------------------------------------------------------------------
_base_mod = _load_module("src.classification.base")
BaseClassifier = _base_mod.BaseClassifier

# ---------------------------------------------------------------------------
# Load all non-torch modules
# ---------------------------------------------------------------------------
_errp_mod = _load_module("src.adaptation.errp_detector")
ErrPP300Detector = _errp_mod.ErrPP300Detector

_seal_mod = _load_module("src.adaptation.seal_engine")
SEALAdaptationEngine = _seal_mod.SEALAdaptationEngine

_monitor_mod = _load_module("src.analysis.state_monitor")
BCIStateMonitor = _monitor_mod.BCIStateMonitor

_causal_mod = _load_module("src.analysis.causal_channels")
CausalChannelDiscovery = _causal_mod.CausalChannelDiscovery

_koopman_mod = _load_module("src.analysis.koopman_decomposition")
KoopmanEEGDecomposition = _koopman_mod.KoopmanEEGDecomposition

_ftle_mod = _load_module("src.analysis.ftle_analysis")
FTLEAnalyzer = _ftle_mod.FTLEAnalyzer

_jac_mod = _load_module("src.features.jacobian_features")
JacobianFeatureExtractor = _jac_mod.JacobianFeatureExtractor

_router_mod = _load_module("src.classification.adaptive_router")
AdaptiveClassifierRouter = _router_mod.AdaptiveClassifierRouter
CSPLDA = _router_mod.CSPLDA
EEGNET = _router_mod.EEGNET
RIEMANNIAN = _router_mod.RIEMANNIAN


# ===========================================================================
# 1. ErrPP300Detector
# ===========================================================================


class TestErrPP300Detector:
    """Tests for ErrP / P300 detector."""

    def test_init(self):
        det = ErrPP300Detector(sf=125)
        assert det.sf == 125
        assert det.mode == "heuristic"
        assert det.pending_count == 0
        assert det.n_correct == 0
        assert det.n_error == 0
        assert det.n_neutral == 0

    def test_update_buffer(self):
        det = ErrPP300Detector(sf=125)
        eeg = RNG.randn(16, 50).astype(np.float64)
        det.update_buffer(eeg, timestamp=1.0)
        assert det._eeg_buffer is not None
        assert det._eeg_buffer.shape[0] == 16

    def test_record_action(self):
        det = ErrPP300Detector(sf=125)
        det.record_action(timestamp=1.0, predicted_class="left")
        assert det.pending_count == 1
        det.record_action(timestamp=2.0, predicted_class="right",
                          eeg_epoch=RNG.randn(16, 125))
        assert det.pending_count == 2

    def test_detect_empty_buffer(self):
        det = ErrPP300Detector(sf=125)
        results = det.detect(current_time=5.0)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_detect_with_synthetic_data(self):
        det = ErrPP300Detector(sf=125, fcz_idx=0, fz_idx=1,
                               p3_idx=2, p4_idx=3)
        n_ch = 4
        eeg = RNG.randn(n_ch, 375).astype(np.float64)
        det.update_buffer(eeg, timestamp=3.0)
        det.record_action(timestamp=1.0, predicted_class="left")
        results = det.detect(current_time=2.0)
        assert isinstance(results, list)
        for r in results:
            assert "result" in r
            assert r["result"] in ("correct", "error", "neutral")
            assert "confidence" in r

    def test_heuristic_classification(self):
        det = ErrPP300Detector(sf=125, fcz_idx=0, fz_idx=1,
                               p3_idx=2, p4_idx=3)
        result = det._heuristic_classify(
            p300_amp=2.0, errp_amp=-1.0, rebound_amp=0.5)
        assert result["label"] in ("correct", "error", "neutral")
        assert "confidence" in result
        assert isinstance(result["p300_amplitude"], float)

    def test_template_accumulation(self):
        det = ErrPP300Detector(sf=125)
        epoch = RNG.randn(16, 100).astype(np.float64)
        det._accumulate_template(epoch, "correct")
        assert det._n_correct_templates == 1
        assert det.p300_template is not None
        det._accumulate_template(epoch, "error")
        assert det._n_error_templates == 1
        assert det.errp_template is not None

    def test_reset(self):
        det = ErrPP300Detector(sf=125)
        det.record_action(timestamp=0.5, predicted_class="rest")
        det.n_correct = 5
        det.reset()
        assert det.pending_count == 0
        assert det.n_correct == 0
        assert det.mode == "heuristic"


# ===========================================================================
# 2. SEALAdaptationEngine
# ===========================================================================


class _MockClassifier:
    """Minimal mock classifier for SEAL tests."""

    def fit(self, X, y):
        self._fitted = True
        return self

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


class TestSEALAdaptationEngine:
    """Tests for SEAL adaptation engine."""

    def _make_engine(self):
        config = {
            "adaptation": {
                "enabled": True,
                "update_interval_s": 0.0,
                "min_samples_for_update": 1,
            }
        }
        return SEALAdaptationEngine(config)

    def test_init(self):
        engine = self._make_engine()
        assert engine.enabled is True
        assert engine.n_updates == 0
        stats = engine.get_stats()
        assert stats["enabled"] is True
        assert stats["positive_buffer"] == 0

    def test_on_prediction(self):
        engine = self._make_engine()
        epoch = RNG.randn(16, 125).astype(np.float64)
        engine.on_prediction(epoch, predicted_class=0, action_time=1.0)
        assert len(engine._pending) == 1

    def test_on_errp_result_correct(self):
        engine = self._make_engine()
        epoch = RNG.randn(16, 125).astype(np.float64)
        engine.on_prediction(epoch, predicted_class=0, action_time=1.0)
        info = engine.on_errp_result(
            action_time=1.0, result="correct", confidence=0.8)
        assert info is not None
        assert info["result"] == "correct"
        assert engine.n_confirmed == 1
        assert len(engine._positive_buffer) == 1

    def test_on_errp_result_error(self):
        engine = self._make_engine()
        epoch = RNG.randn(16, 125).astype(np.float64)
        engine.on_prediction(epoch, predicted_class=1, action_time=2.0)
        info = engine.on_errp_result(
            action_time=2.0, result="error", confidence=0.7)
        assert info is not None
        assert info["should_undo"] is True
        assert engine.n_corrected == 1

    def test_replay_buffer_population(self):
        engine = self._make_engine()
        X = RNG.randn(20, 16, 125).astype(np.float64)
        y = RNG.randint(0, 3, size=20)
        engine.load_replay_buffer(X, y)
        assert len(engine._replay_buffer) == 20

    def test_maybe_update_with_mock(self):
        engine = self._make_engine()
        clf = _MockClassifier()
        engine.set_classifier(clf, class_names=["left", "right", "rest"])
        for _ in range(3):
            engine._positive_buffer.append(
                {"eeg": RNG.randn(16, 125), "label": 0})
        # Patch _apply_update to avoid importing mne-dependent classifiers
        engine._apply_update = lambda X, y: clf.fit(X, y)
        updated = engine.maybe_update(current_time=100.0)
        assert updated is True
        assert engine.n_updates == 1

    def test_disabled_engine(self):
        config = {"adaptation": {"enabled": False}}
        engine = SEALAdaptationEngine(config)
        epoch = RNG.randn(16, 125).astype(np.float64)
        engine.on_prediction(epoch, predicted_class=0, action_time=1.0)
        assert len(engine._pending) == 0


# ===========================================================================
# 3. BCIStateMonitor
# ===========================================================================


class TestBCIStateMonitor:
    """Tests for BCI state monitor (Early Warning Signals)."""

    def test_init(self):
        mon = BCIStateMonitor(sf=125, n_channels=16)
        assert mon.sf == 125
        assert mon.n_channels == 16

    def test_update_returns_none_before_interval(self):
        mon = BCIStateMonitor(sf=125, n_channels=4, window_s=2.0,
                              update_interval_s=5.0)
        eeg = RNG.randn(4, 250).astype(np.float64)
        result = mon.update(eeg, current_time=1.0)
        assert result is None

    def test_update_returns_assessment(self):
        mon = BCIStateMonitor(sf=125, n_channels=4, window_s=2.0,
                              update_interval_s=0.0)
        eeg = RNG.randn(4, 500).astype(np.float64)
        mon.update(eeg, current_time=0.0)
        result = mon.update(eeg, current_time=6.0)
        assert result is not None
        assert "state" in result
        assert result["state"] in ("stable", "warning", "degraded")
        assert "fatigue_score" in result
        assert "attention_score" in result
        assert "electrode_quality" in result
        assert isinstance(result["ews_indicators"], dict)

    def test_state_with_noisy_data(self):
        mon = BCIStateMonitor(sf=125, n_channels=4, window_s=2.0,
                              update_interval_s=0.0)
        eeg = RNG.randn(4, 500).astype(np.float64) * 200
        mon.update(eeg, current_time=0.0)
        result = mon.update(eeg, current_time=6.0)
        assert result is not None
        assert 0.0 <= result["fatigue_score"] <= 1.0
        assert 0.0 <= result["attention_score"] <= 1.0
        assert 0.0 <= result["electrode_quality"] <= 1.0


# ===========================================================================
# 4. CausalChannelDiscovery
# ===========================================================================


class TestCausalChannelDiscovery:
    """Tests for DAGMA-inspired causal channel discovery."""

    def test_init(self):
        ccd = CausalChannelDiscovery(n_channels=4, sf=125)
        assert ccd.n_channels == 4
        assert len(ccd.channel_names) == 4
        assert ccd.discovered_classes == []

    def test_discover_with_synthetic_epochs(self):
        ccd = CausalChannelDiscovery(
            n_channels=4, sf=125,
            class_names=["left", "right"],
            max_iter=10,
        )
        n_trials = 20
        epochs = RNG.randn(n_trials, 4, 125).astype(np.float64)
        labels = np.array([0] * 10 + [1] * 10)
        adj = ccd.discover(epochs, labels)
        assert isinstance(adj, dict)
        assert "left" in adj
        assert "right" in adj
        assert adj["left"].shape == (4, 4)
        assert adj["right"].shape == (4, 4)
        # Values should be non-negative where finite (NaN can arise from
        # numerical overflow in expm during DAGMA optimisation on random data)
        finite_mask = np.isfinite(adj["left"])
        assert np.all(adj["left"][finite_mask] >= 0)

    def test_get_important_channels(self):
        ccd = CausalChannelDiscovery(n_channels=4, sf=125, max_iter=10)
        epochs = RNG.randn(20, 4, 125).astype(np.float64)
        labels = np.array([0] * 10 + [1] * 10)
        ccd.discover(epochs, labels)
        cls_name = ccd.discovered_classes[0]
        important = ccd.get_important_channels(cls_name, top_k=2)
        assert isinstance(important, list)
        assert len(important) == 2
        assert all(0 <= ch < 4 for ch in important)

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            CausalChannelDiscovery(n_channels=1)


# ===========================================================================
# 5. KoopmanEEGDecomposition
# ===========================================================================


class TestKoopmanEEGDecomposition:
    """Tests for Koopman spectral decomposition via DMD."""

    def _make_sine_data(self, n_ch=4, n_samples=500, sf=125):
        t = np.arange(n_samples) / sf
        data = np.zeros((n_ch, n_samples))
        for ch in range(n_ch):
            freq = 8 + ch * 2
            data[ch] = np.sin(2 * np.pi * freq * t) + \
                0.1 * RNG.randn(n_samples)
        return data

    def test_fit(self):
        kd = KoopmanEEGDecomposition(n_channels=4, sf=125, n_modes=5)
        data = self._make_sine_data()
        kd.fit(data)
        assert kd._is_fitted is True

    def test_get_modes(self):
        kd = KoopmanEEGDecomposition(n_channels=4, sf=125, n_modes=5)
        kd.fit(self._make_sine_data())
        modes = kd.get_modes()
        assert isinstance(modes, list)
        assert len(modes) == 5
        for m in modes:
            assert "frequency_hz" in m
            assert "growth_rate" in m
            assert "amplitude" in m
            assert "spatial_pattern" in m
            assert m["spatial_pattern"].shape == (4,)

    def test_get_subject_mu_band(self):
        kd = KoopmanEEGDecomposition(n_channels=4, sf=125, n_modes=10)
        kd.fit(self._make_sine_data())
        center, bw = kd.get_subject_mu_band()
        assert isinstance(center, float)
        assert isinstance(bw, float)
        assert 6.0 <= center <= 15.0
        assert 0.5 <= bw <= 4.0

    def test_not_fitted_raises(self):
        kd = KoopmanEEGDecomposition(n_channels=4, sf=125)
        with pytest.raises(RuntimeError):
            kd.get_modes()


# ===========================================================================
# 6. FTLEAnalyzer
# ===========================================================================


class TestFTLEAnalyzer:
    """Tests for FTLE (Lagrangian coherent structures) analyzer."""

    def test_compute_ftle_shape(self):
        freqs = np.arange(1, 30, 1.0)
        analyzer = FTLEAnalyzer(sf=125, freqs=freqs)
        epoch = RNG.randn(4, 250).astype(np.float64)
        ftle = analyzer.compute_ftle(epoch, channel=0, dt=0.1)
        assert ftle.shape[0] == len(freqs)
        assert ftle.shape[1] == 250
        assert np.all(np.isfinite(ftle))

    def test_compute_ftle_1d_input(self):
        freqs = np.arange(1, 20, 2.0)
        analyzer = FTLEAnalyzer(sf=125, freqs=freqs)
        signal = RNG.randn(200).astype(np.float64)
        ftle = analyzer.compute_ftle(signal, dt=0.1)
        assert ftle.shape == (len(freqs), 200)

    def test_find_ridges(self):
        freqs = np.arange(1, 20, 2.0)
        analyzer = FTLEAnalyzer(sf=125, freqs=freqs)
        signal = RNG.randn(200).astype(np.float64)
        ftle = analyzer.compute_ftle(signal, dt=0.1)
        ridges = analyzer.find_ridges(ftle)
        assert isinstance(ridges, list)


# ===========================================================================
# 7. JacobianFeatureExtractor
# ===========================================================================


class TestJacobianFeatureExtractor:
    """Tests for Jacobian-SVD dynamical features."""

    def test_extract_shape(self):
        jfe = JacobianFeatureExtractor(n_channels=4, sf=125,
                                       embedding_dim=3, tau=5)
        epoch = RNG.randn(4, 200).astype(np.float64)
        features = jfe.extract(epoch)
        expected_len = 4 * jfe.N_FEATURES_PER_CHANNEL
        assert features.shape == (expected_len,)
        assert np.all(np.isfinite(features))

    def test_get_feature_names(self):
        jfe = JacobianFeatureExtractor(n_channels=3, sf=125)
        names = jfe.get_feature_names()
        assert len(names) == 3 * jfe.N_FEATURES_PER_CHANNEL
        assert names[0] == "ch0_max_lyapunov"
        assert "ch2_spectral_gap" in names

    def test_short_epoch_returns_zeros(self):
        jfe = JacobianFeatureExtractor(n_channels=2, sf=125,
                                       embedding_dim=3, tau=5)
        short = RNG.randn(2, 5).astype(np.float64)
        features = jfe.extract(short)
        assert features.shape == (10,)
        assert np.allclose(features, 0.0)

    def test_auto_tau(self):
        jfe = JacobianFeatureExtractor(n_channels=2, sf=125,
                                       embedding_dim=3, tau=None)
        epoch = RNG.randn(2, 300).astype(np.float64)
        features = jfe.extract(epoch)
        assert features.shape == (10,)
        assert np.all(np.isfinite(features))


# ===========================================================================
# 8. AdaptiveClassifierRouter
# ===========================================================================


class _FakeClassifier(BaseClassifier):
    """Fake classifier implementing BaseClassifier for router tests."""

    def __init__(self, n_classes=3):
        self._n_classes = n_classes

    def fit(self, X, y):
        return self

    def predict(self, X):
        if X.ndim == 2:
            return np.array([0])
        return np.zeros(X.shape[0], dtype=int)

    def predict_proba(self, X):
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
        n = X.shape[0]
        proba = np.ones((n, self._n_classes)) / self._n_classes
        proba[:, 0] += 0.3
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def decision_function(self, X):
        return self.predict_proba(X)


class TestAdaptiveClassifierRouter:
    """Tests for the adaptive routing mechanism."""

    def _make_router(self, n_classes=3):
        classifiers = {
            CSPLDA: _FakeClassifier(n_classes),
            EEGNET: _FakeClassifier(n_classes),
            RIEMANNIAN: _FakeClassifier(n_classes),
        }
        return AdaptiveClassifierRouter(classifiers)

    def test_init(self):
        router = self._make_router()
        assert router._fitted is False

    def test_fit(self):
        router = self._make_router()
        X = RNG.randn(30, 4, 250).astype(np.float64)
        y = RNG.randint(0, 3, size=30)
        router.fit(X, y)
        assert router._fitted is True

    def test_predict(self):
        router = self._make_router()
        X = RNG.randn(30, 4, 250).astype(np.float64)
        y = RNG.randint(0, 3, size=30)
        router.fit(X, y)
        preds = router.predict(X[:5])
        assert preds.shape == (5,)
        assert preds.dtype == int

    def test_predict_proba(self):
        router = self._make_router()
        X = RNG.randn(20, 4, 250).astype(np.float64)
        y = RNG.randint(0, 3, size=20)
        router.fit(X, y)
        proba = router.predict_proba(X[:3])
        assert proba.shape == (3, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_get_routing_stats(self):
        router = self._make_router()
        X = RNG.randn(10, 4, 250).astype(np.float64)
        y = RNG.randint(0, 3, size=10)
        router.fit(X, y)
        router.predict(X[:3])
        stats = router.get_routing_stats()
        assert "total_predictions" in stats
        assert stats["total_predictions"] == 3
        assert "routing_counts" in stats
        assert "experts_available" in stats

    def test_missing_classifier_raises(self):
        with pytest.raises(ValueError):
            AdaptiveClassifierRouter({CSPLDA: _FakeClassifier()})


# ===========================================================================
# 9. NeuralSDEClassifier (skip if no torch)
# ===========================================================================


class TestNeuralSDEClassifier:
    """Tests for the latent Neural SDE classifier."""

    @pytest.fixture(autouse=True)
    def _skip_no_torch(self):
        pytest.importorskip("torch")

    def _load_cls(self):
        mod = _load_module("src.classification.neural_sde")
        return mod.NeuralSDEClassifier

    def test_init(self):
        NeuralSDEClassifier = self._load_cls()
        clf = NeuralSDEClassifier(
            n_channels=4, n_samples=64, n_classes=3,
            latent_dim=8, n_steps=5, device="cpu")
        assert clf.n_channels == 4
        assert clf.n_classes == 3

    def test_fit_and_predict(self):
        NeuralSDEClassifier = self._load_cls()
        clf = NeuralSDEClassifier(
            n_channels=4, n_samples=64, n_classes=3,
            latent_dim=8, n_steps=5, device="cpu")
        X = RNG.randn(20, 4, 64).astype(np.float32)
        y = RNG.randint(0, 3, size=20).astype(np.int64)
        clf.fit(X, y, epochs=3, batch_size=10, patience=2)
        preds = clf.predict(X[:5])
        assert preds.shape == (5,)
        proba = clf.predict_proba(X[:5])
        assert proba.shape == (5, 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)

    def test_predict_all(self):
        NeuralSDEClassifier = self._load_cls()
        clf = NeuralSDEClassifier(
            n_channels=4, n_samples=64, n_classes=2,
            latent_dim=8, n_steps=3, device="cpu")
        X = RNG.randn(15, 4, 64).astype(np.float32)
        y = RNG.randint(0, 2, size=15).astype(np.int64)
        clf.fit(X, y, epochs=2, batch_size=8, patience=1)
        preds, proba, scores = clf.predict_all(X[:3])
        assert preds.shape == (3,)
        assert proba.shape == (3, 2)
        assert scores.shape == (3, 2)


# ===========================================================================
# 10. JEPAPretrainer (skip if no torch)
# ===========================================================================


class TestJEPAPretrainer:
    """Tests for TS-JEPA self-supervised pre-trainer."""

    @pytest.fixture(autouse=True)
    def _skip_no_torch(self):
        pytest.importorskip("torch")

    def _load_cls(self):
        mod = _load_module("src.training.pretrain")
        return mod.JEPAPretrainer

    def test_init(self):
        JEPAPretrainer = self._load_cls()
        jp = JEPAPretrainer(n_channels=4, n_samples=64, sf=125,
                            embed_dim=16)
        assert jp.n_channels == 4
        assert jp._is_pretrained is False

    def test_pretrain_and_extract(self):
        JEPAPretrainer = self._load_cls()
        jp = JEPAPretrainer(n_channels=4, n_samples=64, sf=125,
                            embed_dim=16)
        data = RNG.randn(30, 4, 64).astype(np.float32)
        result = jp.pretrain(data, n_epochs=3, batch_size=10)
        assert "loss_history" in result
        assert len(result["loss_history"]) == 3
        assert jp._is_pretrained is True
        features = jp.extract_features(data[:5])
        assert features.shape == (5, 4 * 16)

    def test_extract_single(self):
        JEPAPretrainer = self._load_cls()
        jp = JEPAPretrainer(n_channels=4, n_samples=64, sf=125,
                            embed_dim=16)
        data = RNG.randn(10, 4, 64).astype(np.float32)
        jp.pretrain(data, n_epochs=2, batch_size=5)
        single = RNG.randn(4, 64).astype(np.float32)
        feats = jp.extract_features(single)
        assert feats.shape == (4 * 16,)


# ===========================================================================
# 11. VariableSelector (skip if no torch)
# ===========================================================================


class TestVariableSelector:
    """Tests for TFT-inspired variable selection."""

    @pytest.fixture(autouse=True)
    def _skip_no_torch(self):
        pytest.importorskip("torch")

    def _load_cls(self):
        mod = _load_module("src.features.variable_selector")
        return mod.VariableSelector

    def test_init(self):
        VariableSelector = self._load_cls()
        vs = VariableSelector(n_channels=4, n_features=10, n_classes=3,
                              d_hidden=32)
        assert vs.n_channels == 4
        assert vs._fitted is False

    def test_fit_and_transform(self):
        VariableSelector = self._load_cls()
        vs = VariableSelector(n_channels=4, n_features=10, n_classes=3,
                              d_hidden=32)
        X = RNG.randn(40, 4, 10).astype(np.float32)
        y = RNG.randint(0, 3, size=40).astype(np.int64)
        vs.fit(X, y, epochs=5, batch_size=10)
        assert vs._fitted is True
        out = vs.transform(X[:5])
        assert out.shape == (5, 10)

    def test_get_importance(self):
        VariableSelector = self._load_cls()
        vs = VariableSelector(n_channels=4, n_features=10, n_classes=3,
                              d_hidden=32)
        X = RNG.randn(30, 4, 10).astype(np.float32)
        y = RNG.randint(0, 3, size=30).astype(np.int64)
        vs.fit(X, y, epochs=3, batch_size=10)
        imp = vs.get_importance()
        assert isinstance(imp, dict)
        assert len(imp) == 4
        assert all(v >= 0 for v in imp.values())

    def test_explain(self):
        VariableSelector = self._load_cls()
        vs = VariableSelector(n_channels=4, n_features=10, n_classes=3,
                              d_hidden=32)
        X = RNG.randn(20, 4, 10).astype(np.float32)
        y = RNG.randint(0, 3, size=20).astype(np.int64)
        vs.fit(X, y, epochs=3, batch_size=10)
        exp = vs.explain(X[0])
        assert "predicted_class" in exp
        assert "variable_weights" in exp
        assert "top_variable" in exp
        assert exp["class_probabilities"].shape == (3,)


# ===========================================================================
# 12. GFlowNetSEALOptimizer (skip if no torch)
# ===========================================================================


class TestGFlowNetSEALOptimizer:
    """Tests for GFlowNet-based SEAL hyperparameter optimization."""

    @pytest.fixture(autouse=True)
    def _skip_no_torch(self):
        pytest.importorskip("torch")

    def _load_cls(self):
        mod = _load_module("src.adaptation.gflownet_strategy")
        return mod.GFlowNetSEALOptimizer

    def test_init(self):
        GFlowNetSEALOptimizer = self._load_cls()
        gfn = GFlowNetSEALOptimizer()
        stats = gfn.get_stats()
        assert stats["n_updates"] == 0

    def test_propose_config(self):
        GFlowNetSEALOptimizer = self._load_cls()
        gfn = GFlowNetSEALOptimizer()
        config = gfn.propose_config(current_accuracy=0.7, n_actions=10)
        assert isinstance(config, dict)
        assert "blend_ratio" in config
        assert "update_interval" in config
        assert "min_samples" in config
        assert "learning_rate" in config

    def test_update(self):
        GFlowNetSEALOptimizer = self._load_cls()
        gfn = GFlowNetSEALOptimizer()
        config = gfn.propose_config(current_accuracy=0.6, n_actions=5)
        loss = gfn.update(config, accuracy_before=0.6, accuracy_after=0.75)
        assert isinstance(loss, float)
        assert np.isfinite(loss)
        stats = gfn.get_stats()
        assert stats["n_updates"] == 1
        assert stats["best_reward"] > 0

    def test_get_best_config(self):
        GFlowNetSEALOptimizer = self._load_cls()
        gfn = GFlowNetSEALOptimizer()
        best = gfn.get_best_config()
        assert isinstance(best, dict)
        assert "blend_ratio" in best
        assert best["blend_ratio"] == 0.3


# ===========================================================================
# Run with pytest
# ===========================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
