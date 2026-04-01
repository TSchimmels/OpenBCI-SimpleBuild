"""Abstract base classifier interface.

All classifiers (CSP+LDA, Riemannian, EEGNet) implement this interface
so they can be swapped without changing the rest of the pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class BaseClassifier(ABC):
    """Abstract base class for all EEG classifiers.

    Subclasses must implement fit, predict, predict_proba, and
    decision_function. Save/load use joblib by default but can
    be overridden (e.g., for PyTorch state_dict).
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseClassifier":
        """Train the classifier.

        Args:
            X: Training data, shape (n_trials, n_channels, n_samples)
            y: Labels, shape (n_trials,)

        Returns:
            self
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Input data, shape (n_trials, n_channels, n_samples)
                or (n_channels, n_samples) for single trial

        Returns:
            Predicted labels, shape (n_trials,) or scalar
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input data

        Returns:
            Probabilities, shape (n_trials, n_classes)
        """

    @abstractmethod
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Continuous output for proportional control.

        For 2-class: signed distance to decision boundary.
        For multi-class: per-class scores.

        Args:
            X: Input data

        Returns:
            Continuous scores
        """

    def predict_all(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels, probabilities, and decision scores.

        Default implementation calls predict(), predict_proba(), and
        decision_function() separately.  Subclasses (e.g. EEGNet) may
        override to share a single forward pass and avoid redundant
        data transfers.

        Args:
            X: Input data, shape (n_trials, n_channels, n_samples)
                or (n_channels, n_samples) for single trial

        Returns:
            (predictions, probabilities, decision_scores)
        """
        return self.predict(X), self.predict_proba(X), self.decision_function(X)

    def save(self, path: str) -> None:
        """Save trained model to disk.

        Args:
            path: File path (e.g., 'models/csp_lda.pkl')
        """
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "BaseClassifier":
        """Load trained model from disk.

        Dispatches to the correct loader based on file extension:
        - ``.pkl`` / ``.joblib`` — scikit-learn models (CSP+LDA, Riemannian)
        - ``.pt`` / ``.pth`` — PyTorch models (EEGNet, Neural SDE)

        Args:
            path: File path

        Returns:
            Loaded classifier instance
        """
        ext = Path(path).suffix.lower()
        if ext in (".pt", ".pth"):
            # Detect model type from checkpoint and dispatch
            import torch
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)

            if "n_steps" in checkpoint and "dt" in checkpoint:
                from .neural_sde import NeuralSDEClassifier
                return NeuralSDEClassifier.load(path)
            else:
                from .eegnet import EEGNetClassifier
                return EEGNetClassifier.load(path)

        import joblib
        return joblib.load(path)
