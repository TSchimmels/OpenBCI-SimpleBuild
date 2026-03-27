"""Tests for the classification module (CSP+LDA, pipeline factory, EEGNet).

Covers fitting, predicting, decision functions, save/load, the
ClassifierFactory, and (optionally) EEGNet forward pass.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.classification.csp_lda import CSPLDAClassifier
from src.classification.pipeline import ClassifierFactory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_mi_data(rng):
    """Synthetic 5-class motor imagery data with different spatial patterns.

    Class 0 (rest):       baseline noise, no boosted channels.
    Class 1 (left_hand):  higher variance in channels 0-1.
    Class 2 (right_hand): higher variance in channels 2-3.
    Class 3 (feet):       higher variance in channels 4-5.
    Class 4 (tongue):     higher variance in channels 6-7.

    Returns (X_train, y_train, X_test, y_test).
    """
    n_channels = 8
    n_samples = 250
    n_classes = 5
    class_names = ["rest", "left_hand", "right_hand", "feet", "tongue"]
    n_per_class_train = 20
    n_per_class_test = 5
    n_train = n_per_class_train * n_classes  # 100
    n_test = n_per_class_test * n_classes    # 25

    X_train = rng.standard_normal((n_train, n_channels, n_samples)).astype(np.float64)
    y_train = np.concatenate([np.full(n_per_class_train, c) for c in range(n_classes)])

    X_test = rng.standard_normal((n_test, n_channels, n_samples)).astype(np.float64)
    y_test = np.concatenate([np.full(n_per_class_test, c) for c in range(n_classes)])

    # Inject class-specific spatial patterns
    # Class 0 (rest): no boost — baseline
    # Class 1 (left_hand, train): boost channels 0-1
    start = n_per_class_train
    X_train[start : start + n_per_class_train, :2, :] *= 3.0
    # Class 2 (right_hand, train): boost channels 2-3
    start = 2 * n_per_class_train
    X_train[start : start + n_per_class_train, 2:4, :] *= 3.0
    # Class 3 (feet, train): boost channels 4-5
    start = 3 * n_per_class_train
    X_train[start : start + n_per_class_train, 4:6, :] *= 3.0
    # Class 4 (tongue, train): boost channels 6-7
    start = 4 * n_per_class_train
    X_train[start : start + n_per_class_train, 6:8, :] *= 3.0

    # Class 1 (left_hand, test): boost channels 0-1
    start = n_per_class_test
    X_test[start : start + n_per_class_test, :2, :] *= 3.0
    # Class 2 (right_hand, test): boost channels 2-3
    start = 2 * n_per_class_test
    X_test[start : start + n_per_class_test, 2:4, :] *= 3.0
    # Class 3 (feet, test): boost channels 4-5
    start = 3 * n_per_class_test
    X_test[start : start + n_per_class_test, 4:6, :] *= 3.0
    # Class 4 (tongue, test): boost channels 6-7
    start = 4 * n_per_class_test
    X_test[start : start + n_per_class_test, 6:8, :] *= 3.0

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# CSP+LDA classifier tests
# ---------------------------------------------------------------------------

class TestCSPLDAClassifier:

    def test_csp_lda_fit_predict(self, synthetic_mi_data):
        """Fit CSP+LDA on synthetic MI data; accuracy should exceed chance (30%)."""
        X_train, y_train, X_test, y_test = synthetic_mi_data

        clf = CSPLDAClassifier(n_components=4)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        accuracy = np.mean(predictions == y_test)
        assert accuracy > 0.30, (
            f"Accuracy {accuracy:.2%} should exceed 30% (chance=20%) "
            f"on well-separated synthetic data"
        )

    def test_csp_lda_decision_function(self, synthetic_mi_data):
        """decision_function should return continuous values, not discrete labels."""
        X_train, y_train, X_test, _ = synthetic_mi_data

        clf = CSPLDAClassifier(n_components=4)
        clf.fit(X_train, y_train)
        scores = clf.decision_function(X_test)

        # For 5-class classification, scores should be 2-D (n_samples, n_classes)
        assert scores.ndim in (1, 2), f"Expected 1-D or 2-D scores, got shape {scores.shape}"
        assert scores.shape[0] == X_test.shape[0]

        # Scores should not be purely integer (i.e., they are continuous)
        unique_values = np.unique(scores)
        assert len(unique_values) > 2, (
            f"Decision function should return continuous values, "
            f"but got only {len(unique_values)} unique values"
        )

    def test_csp_lda_single_trial(self, synthetic_mi_data):
        """A single trial (2D array) should work without error."""
        X_train, y_train, X_test, _ = synthetic_mi_data

        clf = CSPLDAClassifier(n_components=4)
        clf.fit(X_train, y_train)

        # Pass a single trial as 2D: (n_channels, n_samples)
        single_trial = X_test[0]
        assert single_trial.ndim == 2, "Single trial should be 2D"

        prediction = clf.predict(single_trial)
        assert prediction is not None
        # Should return an array (possibly length 1)
        assert isinstance(prediction, np.ndarray)

    def test_csp_lda_save_load(self, synthetic_mi_data):
        """Train, save, load, and verify predictions match."""
        X_train, y_train, X_test, _ = synthetic_mi_data

        clf = CSPLDAClassifier(n_components=4)
        clf.fit(X_train, y_train)
        original_predictions = clf.predict(X_test)
        original_scores = clf.decision_function(X_test)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name

        try:
            clf.save(tmp_path)

            # Load and verify
            loaded_clf = CSPLDAClassifier.load(tmp_path)
            loaded_predictions = loaded_clf.predict(X_test)
            loaded_scores = loaded_clf.decision_function(X_test)

            np.testing.assert_array_equal(
                original_predictions, loaded_predictions,
                err_msg="Predictions differ after save/load"
            )
            np.testing.assert_array_almost_equal(
                original_scores, loaded_scores, decimal=10,
                err_msg="Decision scores differ after save/load"
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# ClassifierFactory tests
# ---------------------------------------------------------------------------

class TestClassifierFactory:

    def test_classifier_factory_csp_lda(self):
        """Factory should return a CSPLDAClassifier for model_type='csp_lda'."""
        config = {
            "classification": {"model_type": "csp_lda"},
            "features": {"csp_n_components": 6},
        }
        clf = ClassifierFactory.create(config)

        assert isinstance(clf, CSPLDAClassifier), (
            f"Expected CSPLDAClassifier, got {type(clf).__name__}"
        )
        assert clf.n_components == 6, (
            f"Expected n_components=6, got {clf.n_components}"
        )


# ---------------------------------------------------------------------------
# EEGNet tests (optional — requires PyTorch)
# ---------------------------------------------------------------------------

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class TestEEGNet:

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_eegnet_if_available(self):
        """EEGNet should process a batch through a forward pass without error."""
        from src.classification.eegnet import EEGNetClassifier, EEGNetModel

        n_channels = 8
        n_samples = 128  # must be divisible by 32
        n_classes = 5
        batch_size = 4

        # Build the raw model and run a forward pass
        model = EEGNetModel(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            F1=8,
            D=2,
            F2=16,
            kernel_length=32,
            dropout=0.25,
        )
        model.eval()

        # Input shape: (batch, 1, n_channels, n_samples)
        x = torch.randn(batch_size, 1, n_channels, n_samples)

        with torch.no_grad():
            logits = model(x)

        assert logits.shape == (batch_size, n_classes), (
            f"Expected output shape ({batch_size}, {n_classes}), got {logits.shape}"
        )
        # Logits should be finite
        assert torch.all(torch.isfinite(logits)), "EEGNet output contains NaN or Inf"
