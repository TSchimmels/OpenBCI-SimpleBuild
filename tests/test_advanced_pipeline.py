"""Tests for the advanced training pipeline.

Tests SubjectProfiler, EEGAugmenter, FilterBankCSP, MultiModelTrainer,
EnsembleBuilder, and the full AdvancedTrainingPipeline using synthetic data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_epochs(rng):
    """Synthetic 5-class MI epochs with distinguishable spatial patterns.

    Returns (epochs, labels) with shape (100, 8, 250) and (100,).
    """
    n_channels = 8
    n_samples = 250
    n_classes = 5
    n_per_class = 20

    epochs = rng.standard_normal((n_per_class * n_classes, n_channels, n_samples))
    labels = np.repeat(np.arange(n_classes), n_per_class)

    # Add class-specific spatial variance patterns
    boost = 3.0
    for cls in range(n_classes):
        mask = labels == cls
        if cls > 0:
            ch_start = (cls - 1) * 2
            ch_end = ch_start + 2
            epochs[mask, ch_start:ch_end, :] *= boost

    # Shuffle
    perm = rng.permutation(len(labels))
    return epochs[perm], labels[perm]


@pytest.fixture
def config():
    """Minimal config for testing."""
    return {
        "board": {
            "board_id": -1,
            "channel_count": 8,
            "sampling_rate_override": 250,
        },
        "preprocessing": {
            "mi_bandpass_low": 8.0,
            "mi_bandpass_high": 30.0,
            "bandpass_order": 4,
            "artifact_threshold_uv": 200.0,
        },
        "features": {
            "csp_n_components": 4,
            "csp_regularization": "ledoit_wolf",
        },
        "classification": {
            "model_type": "csp_lda",
            "lda_shrinkage": "auto",
        },
        "training": {
            "n_classes": 5,
            "classes": ["rest", "left_hand", "right_hand", "feet", "tongue"],
            "classification_window_start": 1.5,
            "classification_window_end": 4.0,
        },
        "advanced": {
            "koopman_n_modes": 5,
            "koopman_delay_dim": 3,
            "causal_top_k": 4,
        },
    }


# ---------------------------------------------------------------------------
# SubjectProfile tests
# ---------------------------------------------------------------------------


class TestSubjectProfile:
    def test_create_default(self):
        from src.training.advanced_pipeline import SubjectProfile

        p = SubjectProfile()
        assert p.mu_band == (10.0, 2.0)
        assert p.optimal_window == (1.5, 4.0)
        assert p.bad_channels == []

    def test_to_dict_roundtrip(self, tmp_path):
        from src.training.advanced_pipeline import SubjectProfile

        p = SubjectProfile(
            mu_band=(9.5, 1.8),
            beta_band=(15.0, 28.0),
            optimal_window=(1.25, 3.5),
            important_channels={"left_hand": [0, 1, 2]},
            hub_channels={"left_hand": [0]},
            trial_weights=np.array([0.5, 0.3, 0.2]),
            bad_channels=[7],
            bad_trials=[3],
        )
        path = str(tmp_path / "profile.json")
        p.save(path)

        loaded = SubjectProfile.load(path)
        assert loaded.mu_band == [9.5, 1.8]  # JSON converts tuple to list
        assert loaded.bad_channels == [7]
        assert loaded.bad_trials == [3]
        assert loaded.trial_weights is not None
        np.testing.assert_allclose(loaded.trial_weights, [0.5, 0.3, 0.2])


# ---------------------------------------------------------------------------
# SubjectProfiler tests
# ---------------------------------------------------------------------------


class TestSubjectProfiler:
    def test_quality_profile(self, synthetic_epochs, config):
        from src.training.advanced_pipeline import SubjectProfiler

        epochs, labels = synthetic_epochs
        profiler = SubjectProfiler(config)
        weights, bad_ch, bad_tr = profiler._quality_profile(epochs)

        assert weights.shape[0] == epochs.shape[0]
        assert abs(weights.sum() - 1.0) < 1e-6 or weights.sum() == 0.0
        assert isinstance(bad_ch, list)
        assert isinstance(bad_tr, list)

    def test_temporal_profile(self, synthetic_epochs, config):
        from src.training.advanced_pipeline import SubjectProfiler

        epochs, labels = synthetic_epochs
        profiler = SubjectProfiler(config)
        window = profiler._temporal_profile(epochs, labels, sf=250)

        assert len(window) == 2
        assert window[0] < window[1]
        assert window[0] >= 0.5
        assert window[1] <= 5.0


# ---------------------------------------------------------------------------
# EEGAugmenter tests
# ---------------------------------------------------------------------------


class TestEEGAugmenter:
    def test_augment_expands_data(self, synthetic_epochs):
        from src.training.advanced_pipeline import EEGAugmenter

        epochs, labels = synthetic_epochs
        aug = EEGAugmenter(strength=0.5, sf=250)
        X_aug, y_aug = aug.augment(epochs, labels)

        assert X_aug.shape[0] > epochs.shape[0]
        assert y_aug.shape[0] == X_aug.shape[0]
        assert X_aug.shape[1] == epochs.shape[1]
        assert X_aug.shape[2] == epochs.shape[2]

    def test_zero_strength_no_change(self, synthetic_epochs):
        from src.training.advanced_pipeline import EEGAugmenter

        epochs, labels = synthetic_epochs
        aug = EEGAugmenter(strength=0.0, sf=250)
        X_aug, y_aug = aug.augment(epochs, labels)

        assert X_aug.shape[0] == epochs.shape[0]
        np.testing.assert_array_equal(y_aug, labels)

    def test_temporal_jitter(self, synthetic_epochs):
        from src.training.advanced_pipeline import EEGAugmenter

        aug = EEGAugmenter(strength=1.0, sf=250)
        trial = synthetic_epochs[0][0]
        jittered = aug._temporal_jitter(trial)
        assert jittered.shape == trial.shape

    def test_gaussian_noise(self, synthetic_epochs):
        from src.training.advanced_pipeline import EEGAugmenter

        aug = EEGAugmenter(strength=1.0, sf=250)
        trial = synthetic_epochs[0][0]
        noisy = aug._gaussian_noise(trial)
        assert noisy.shape == trial.shape
        assert not np.allclose(noisy, trial)

    def test_channel_dropout(self, synthetic_epochs):
        from src.training.advanced_pipeline import EEGAugmenter

        aug = EEGAugmenter(strength=1.0, sf=250)
        trial = synthetic_epochs[0][0]
        dropped = aug._channel_dropout(trial)
        # At least one channel should be zeroed
        zeroed = np.all(dropped == 0, axis=-1)
        assert zeroed.any()

    def test_mixup_within_class(self, synthetic_epochs):
        from src.training.advanced_pipeline import EEGAugmenter

        epochs, labels = synthetic_epochs
        aug = EEGAugmenter(strength=1.0, sf=250)
        mixed_X, mixed_y = aug._mixup(epochs, labels)
        assert mixed_X.shape[0] > 0
        # All mixed labels should be valid original labels
        assert set(mixed_y).issubset(set(labels))

    def test_preserves_class_balance(self, synthetic_epochs):
        from src.training.advanced_pipeline import EEGAugmenter

        epochs, labels = synthetic_epochs
        aug = EEGAugmenter(strength=0.5, sf=250)
        X_aug, y_aug = aug.augment(epochs, labels)

        # All original classes should still be present
        assert set(np.unique(labels)).issubset(set(np.unique(y_aug)))


# ---------------------------------------------------------------------------
# FilterBankCSP tests
# ---------------------------------------------------------------------------


class TestFilterBankCSP:
    @pytest.fixture
    def fbcsp_data(self, rng):
        """Small synthetic data for FBCSP testing (2 classes for simplicity)."""
        n_trials = 40
        n_channels = 8
        n_samples = 250
        X = rng.standard_normal((n_trials, n_channels, n_samples))
        y = np.repeat([0, 1], n_trials // 2)

        # Make class 0 have more power in channel 0
        X[y == 0, 0, :] *= 3.0
        X[y == 1, 2, :] *= 3.0
        return X, y

    @pytest.mark.skipif(
        not _can_import("mne"),
        reason="mne not installed",
    )
    def test_fit_transform_shape(self, fbcsp_data):
        from src.training.advanced_pipeline import FilterBankCSP

        X, y = fbcsp_data
        fbcsp = FilterBankCSP(n_components=4, sf=250)

        features = fbcsp.fit_transform(X, y)
        expected_features = fbcsp.n_bands * 4
        assert features.shape == (X.shape[0], expected_features)

    def test_subject_mu_band_adds_extra(self, fbcsp_data):
        from src.training.advanced_pipeline import FilterBankCSP

        X, y = fbcsp_data
        fbcsp_default = FilterBankCSP(n_components=4, sf=250)
        fbcsp_mu = FilterBankCSP(
            n_components=4, sf=250, subject_mu_band=(10.0, 2.0)
        )

        assert fbcsp_mu.n_bands == fbcsp_default.n_bands + 1

    def test_transform_requires_fit(self, fbcsp_data):
        from src.training.advanced_pipeline import FilterBankCSP

        X, y = fbcsp_data
        fbcsp = FilterBankCSP(n_components=4, sf=250)

        with pytest.raises(RuntimeError, match="not fitted"):
            fbcsp.transform(X)

    def test_bands_property(self):
        from src.training.advanced_pipeline import FilterBankCSP

        fbcsp = FilterBankCSP(n_components=6, sf=125)
        assert fbcsp.n_bands == 9
        assert fbcsp.n_features == 9 * 6


# ---------------------------------------------------------------------------
# MultiModelTrainer tests (requires full classification stack)
# ---------------------------------------------------------------------------


class TestMultiModelTrainer:
    @pytest.mark.skipif(
        not _can_import("mne"),
        reason="mne not installed",
    )
    def test_train_csp_lda(self, synthetic_epochs, config):
        from src.training.advanced_pipeline import MultiModelTrainer

        epochs, labels = synthetic_epochs
        trainer = MultiModelTrainer(
            config, n_splits=3, model_types=["csp_lda"]
        )
        results = trainer.train_all(epochs, labels)

        assert len(results) == 1
        assert results[0].name == "csp_lda"
        assert results[0].cv_accuracy > 0.0
        assert results[0].classifier is not None


# ---------------------------------------------------------------------------
# EnsembleBuilder tests
# ---------------------------------------------------------------------------


class TestEnsembleBuilder:
    def test_single_model_passthrough(self, synthetic_epochs):
        from src.training.advanced_pipeline import EnsembleBuilder, ModelResult

        epochs, labels = synthetic_epochs

        # Create a dummy model result with a simple predictor
        class DummyClassifier:
            def predict(self, X):
                return np.zeros(X.shape[0], dtype=int)

            def predict_proba(self, X):
                proba = np.zeros((X.shape[0], 5))
                proba[:, 0] = 1.0
                return proba

            def save(self, path):
                pass

        result = ModelResult(
            name="dummy",
            classifier=DummyClassifier(),
            cv_accuracy=0.5,
        )

        builder = EnsembleBuilder()
        ensemble = builder.build([result], epochs, labels)

        # With only 1 model, should return it directly
        assert ensemble.name == "dummy"

    def test_soft_voting_with_multiple(self, synthetic_epochs):
        from src.training.advanced_pipeline import EnsembleBuilder, ModelResult

        epochs, labels = synthetic_epochs

        class ConstClassifier:
            def __init__(self, cls):
                self._cls = cls

            def predict(self, X):
                return np.full(X.shape[0], self._cls, dtype=int)

            def predict_proba(self, X):
                proba = np.zeros((X.shape[0], 5))
                proba[:, self._cls] = 0.8
                proba[:, (self._cls + 1) % 5] = 0.2
                return proba

            def save(self, path):
                pass

        results = [
            ModelResult(name="m1", classifier=ConstClassifier(0), cv_accuracy=0.4),
            ModelResult(name="m2", classifier=ConstClassifier(1), cv_accuracy=0.3),
        ]

        builder = EnsembleBuilder()
        ensemble = builder.build(results, epochs, labels)
        assert ensemble.classifier is not None


# ---------------------------------------------------------------------------
# TrainingReport tests
# ---------------------------------------------------------------------------


class TestTrainingReport:
    def test_format_output(self):
        from src.training.advanced_pipeline import (
            SubjectProfile,
            ModelResult,
            TrainingReport,
        )

        profile = SubjectProfile(
            mu_band=(10.0, 2.0),
            important_channels={"left_hand": [0, 1]},
        )
        results = [
            ModelResult(
                name="csp_lda",
                classifier=None,
                cv_accuracy=0.65,
                cv_std=0.04,
                prediction_time_ms=0.3,
                n_params=200,
            )
        ]
        report = TrainingReport(
            profile=profile,
            augmentation_stats={"original": 100, "augmented": 500, "multiplier": 5.0},
            model_results=results,
            ensemble_result=None,
            best_model_name="csp_lda",
            best_accuracy=0.65,
        )

        text = report.format()
        assert "ADVANCED TRAINING REPORT" in text
        assert "csp_lda" in text
        assert "Mu band" in text

    def test_save_creates_files(self, tmp_path):
        from src.training.advanced_pipeline import (
            SubjectProfile,
            ModelResult,
            TrainingReport,
        )

        profile = SubjectProfile()
        report = TrainingReport(
            profile=profile,
            augmentation_stats={"original": 10, "augmented": 50, "multiplier": 5.0},
            model_results=[],
            ensemble_result=None,
            best_model_name="none",
            best_accuracy=0.0,
        )

        paths = report.save(str(tmp_path))
        assert "report" in paths
        assert Path(paths["report"]).exists()
        assert "profile" in paths
        assert Path(paths["profile"]).exists()


# ---------------------------------------------------------------------------
# Integration: AdvancedTrainingPipeline
# ---------------------------------------------------------------------------


class TestAdvancedTrainingPipeline:
    def test_pipeline_phases_run(self, synthetic_epochs, config):
        """Smoke test: pipeline runs all phases without crashing.

        Uses only the SubjectProfiler and EEGAugmenter phases
        since the full model training requires MNE.
        """
        from src.training.advanced_pipeline import (
            SubjectProfiler,
            EEGAugmenter,
        )

        epochs, labels = synthetic_epochs

        # Phase 1
        profiler = SubjectProfiler(config)
        profile = profiler.profile(epochs, labels, sf=250)
        assert profile.optimal_window[0] < profile.optimal_window[1]

        # Phase 2
        augmenter = EEGAugmenter(strength=0.5, sf=250)
        X_aug, y_aug = augmenter.augment(epochs, labels, profile)
        assert X_aug.shape[0] > epochs.shape[0]
