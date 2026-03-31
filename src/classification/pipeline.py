"""Classifier factory and pipeline orchestration.

Provides a single entry point — :meth:`ClassifierFactory.create` — that
reads a configuration dictionary and returns the appropriate classifier
instance, fully configured and ready for ``fit()``.

Supported model types:
    * ``csp_lda`` — :class:`CSPLDAClassifier` (CSP + shrinkage LDA)
    * ``eegnet``  — :class:`EEGNetClassifier` (compact CNN)
    * ``riemannian`` — :class:`RiemannianClassifier` (MDM on SPD manifold)
    * ``neural_sde`` — :class:`NeuralSDEClassifier` (latent Neural SDE)
    * ``adaptive_router`` — :class:`AdaptiveClassifierRouter` (MoE routing)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BaseClassifier

logger = logging.getLogger(__name__)


# ======================================================================
# Riemannian classifier (inline — thin wrapper around pyRiemann)
# ======================================================================

class RiemannianClassifier(BaseClassifier):
    """Minimum Distance to Mean (MDM) on the Riemannian manifold.

    Uses pyRiemann's :class:`Covariances` transformer to map EEG epochs
    to symmetric positive-definite (SPD) covariance matrices, then
    classifies with the :class:`MDM` classifier using a geodesic
    distance metric on the SPD manifold.

    This is a geometry-aware classifier that does not require explicit
    feature extraction and is naturally robust to non-stationarity.

    Args:
        metric: Riemannian metric for MDM (``'riemann'``, ``'logeuclid'``,
            ``'euclid'``).
        estimator: Covariance estimator (``'oas'``, ``'lwf'``, ``'scm'``).

    References:
        Barachant, A., Bonnet, S., Congedo, M., & Jutten, C. (2012).
        Multiclass brain-computer interface classification by Riemannian
        geometry. IEEE Trans. Biomed. Eng., 59(4), 920-928.
    """

    def __init__(
        self,
        metric: str = "riemann",
        estimator: str = "oas",
    ) -> None:
        self._check_pyriemann()
        self.metric = metric
        self.estimator = estimator
        self._pipeline: Optional[Any] = None  # sklearn Pipeline

    @staticmethod
    def _check_pyriemann() -> None:
        """Raise ImportError if pyriemann is not installed."""
        try:
            import pyriemann  # noqa: F401
        except ImportError:
            raise ImportError(
                "pyRiemann is required for the Riemannian classifier but "
                "was not found. Install it with:\n"
                "  pip install pyriemann\n"
                "See https://pyriemann.readthedocs.io/ for details."
            )

    @staticmethod
    def _ensure_3d(X: np.ndarray) -> np.ndarray:
        """Ensure input is 3-D (n_trials, n_channels, n_samples)."""
        if X.ndim == 2:
            return X[np.newaxis, :, :]
        if X.ndim == 3:
            return X
        raise ValueError(
            f"Expected 2-D or 3-D input, got {X.ndim}-D array with "
            f"shape {X.shape}"
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> RiemannianClassifier:
        """Fit the Riemannian MDM pipeline.

        Args:
            X: Training epochs, shape ``(n_trials, n_channels, n_samples)``.
            y: Integer class labels, shape ``(n_trials,)``.

        Returns:
            ``self``
        """
        from pyriemann.classification import MDM
        from pyriemann.estimation import Covariances
        from sklearn.pipeline import Pipeline as SklearnPipeline

        X = self._ensure_3d(X)

        self._pipeline = SklearnPipeline([
            ("cov", Covariances(estimator=self.estimator)),
            ("mdm", MDM(metric=self.metric)),
        ])
        self._pipeline.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Input epochs, shape ``(n_trials, n_channels, n_samples)``
                or ``(n_channels, n_samples)``.

        Returns:
            Predicted integer labels.
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")
        X = self._ensure_3d(X)
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        MDM uses softmax over negative geodesic distances to produce
        pseudo-probabilities.

        Args:
            X: Input epochs.

        Returns:
            Probabilities, shape ``(n_trials, n_classes)``.
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")
        X = self._ensure_3d(X)
        return self._pipeline.predict_proba(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return negative geodesic distances as continuous scores.

        MDM does not have a native ``decision_function``, so we compute
        the negative distance to each class mean. Closer (less negative)
        values indicate higher affinity for that class.

        Args:
            X: Input epochs.

        Returns:
            Scores, shape ``(n_trials, n_classes)``.
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")
        X = self._ensure_3d(X)

        from pyriemann.estimation import Covariances

        # Get the fitted pipeline stages
        cov_transform = self._pipeline.named_steps["cov"]
        mdm_clf = self._pipeline.named_steps["mdm"]

        # Transform to covariance matrices
        X_cov = cov_transform.transform(X)

        # MDM stores class means in covmeans_; compute distances
        # transform() returns negative distances when available
        if hasattr(mdm_clf, "transform"):
            # MDM.transform returns distances to each class mean
            distances = mdm_clf.transform(X_cov)
            # Return negative distances (closer = higher score)
            return -distances
        else:
            # Fallback: use predict_proba and log-transform
            return np.log(self._pipeline.predict_proba(X) + 1e-10)

    def __repr__(self) -> str:
        fitted = self._pipeline is not None
        return (
            f"RiemannianClassifier(metric='{self.metric}', "
            f"estimator='{self.estimator}', fitted={fitted})"
        )


# ======================================================================
# Classifier Factory
# ======================================================================

class ClassifierFactory:
    """Factory for creating classifier instances from configuration.

    Reads the ``classification`` section of the project config and
    returns the appropriate :class:`BaseClassifier` subclass, fully
    configured with the specified hyper-parameters.

    Example::

        config = load_config()
        clf = ClassifierFactory.create(config)
        clf.fit(X_train, y_train)
        labels = clf.predict(X_test)
        scores = clf.decision_function(X_test)

    The factory can also be used with a minimal dict::

        clf = ClassifierFactory.create({
            "classification": {"model_type": "csp_lda"},
            "features": {"csp_n_components": 8},
        })
    """

    # Registry of supported model types
    _SUPPORTED_TYPES = ("csp_lda", "eegnet", "riemannian", "neural_sde", "adaptive_router")

    @staticmethod
    def create(config: dict) -> BaseClassifier:
        """Create a classifier from a configuration dictionary.

        Args:
            config: Full project configuration dictionary.  Must contain
                a ``classification`` section with at least ``model_type``.
                Additional sections (``features``, ``board``, ``training``)
                are used to fill in defaults where relevant.

        Returns:
            A configured (but unfitted) :class:`BaseClassifier` instance.

        Raises:
            ValueError: If ``model_type`` is not one of the supported
                types.
            ImportError: If required libraries are not installed for the
                chosen model type.
        """
        cls_config = config.get("classification", {})
        model_type: str = cls_config.get("model_type", "csp_lda")

        if model_type == "csp_lda":
            return ClassifierFactory._create_csp_lda(config)
        elif model_type == "eegnet":
            return ClassifierFactory._create_eegnet(config)
        elif model_type == "riemannian":
            return ClassifierFactory._create_riemannian(config)
        elif model_type == "neural_sde":
            return ClassifierFactory._create_neural_sde(config)
        elif model_type == "adaptive_router":
            return ClassifierFactory._create_adaptive_router(config)
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Supported types: {ClassifierFactory._SUPPORTED_TYPES}"
            )

    @staticmethod
    def _create_csp_lda(config: dict) -> BaseClassifier:
        """Build a CSP+LDA classifier from config.

        Reads:
            - ``features.csp_n_components`` for CSP component count
            - ``features.csp_regularization`` for CSP regularisation
            - ``classification.lda_shrinkage`` for LDA shrinkage

        Args:
            config: Full project configuration dictionary.

        Returns:
            Configured :class:`CSPLDAClassifier`.
        """
        from .csp_lda import CSPLDAClassifier

        feat_config = config.get("features", {})
        cls_config = config.get("classification", {})

        n_components: int = feat_config.get("csp_n_components", 12)
        reg: str = feat_config.get("csp_regularization", "ledoit_wolf")
        shrinkage: str = cls_config.get("lda_shrinkage", "auto")

        clf = CSPLDAClassifier(
            n_components=n_components,
            reg=reg,
            shrinkage=shrinkage,
        )
        logger.info(
            "Created CSP+LDA classifier (n_components=%d, reg=%s, "
            "shrinkage=%s)",
            n_components, reg, shrinkage,
        )
        return clf

    @staticmethod
    def _create_eegnet(config: dict) -> BaseClassifier:
        """Build an EEGNet classifier from config.

        Reads:
            - ``board.channel_count`` for number of EEG channels
            - ``training`` section for class count and epoch sizing
            - ``classification.eegnet`` sub-section for architecture and
              training hyper-parameters

        Args:
            config: Full project configuration dictionary.

        Returns:
            Configured :class:`EEGNetClassifier`.
        """
        from .eegnet import EEGNetClassifier

        board_config = config.get("board", {})
        cls_config = config.get("classification", {})
        train_config = config.get("training", {})
        eegnet_config = cls_config.get("eegnet", {})

        n_channels: int = board_config.get("channel_count", 16)
        n_classes: int = train_config.get("n_classes", 3)

        # Compute n_samples from sampling rate and classification window.
        # Query the actual board sampling rate via BrainFlow if possible;
        # fall back to sampling_rate_override or 250 (synthetic board default).
        sampling_rate_raw = board_config.get("sampling_rate_override", None)
        if sampling_rate_raw is not None:
            sampling_rate: int = int(sampling_rate_raw)
        else:
            # Query BrainFlow for the actual board's native rate
            try:
                from brainflow.board_shim import BoardShim, BoardIds
                board_id = int(board_config.get("board_id", BoardIds.SYNTHETIC_BOARD))
                if board_id == -1:
                    board_id = BoardIds.SYNTHETIC_BOARD
                sampling_rate = int(BoardShim.get_sampling_rate(board_id))
            except Exception:
                sampling_rate = 250  # Safe default (synthetic board)
        window_start: float = train_config.get("classification_window_start", 1.5)
        window_end: float = train_config.get("classification_window_end", 4.0)
        n_samples: int = int(sampling_rate * (window_end - window_start))

        # Ensure n_samples is divisible by 32 (required by pooling layers)
        if n_samples % 32 != 0:
            adjusted = (n_samples // 32) * 32
            if adjusted == 0:
                adjusted = 32
            logger.warning(
                "n_samples=%d is not divisible by 32. Adjusting to %d. "
                "Ensure your epoch windowing produces %d samples.",
                n_samples, adjusted, adjusted,
            )
            n_samples = adjusted

        clf = EEGNetClassifier(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            F1=eegnet_config.get("F1", 8),
            D=eegnet_config.get("D", 2),
            F2=eegnet_config.get("F2", 16),
            kernel_length=eegnet_config.get("kernel_length", 64),
            dropout=eegnet_config.get("dropout", 0.5),
            epochs=eegnet_config.get("epochs", 300),
            batch_size=eegnet_config.get("batch_size", 32),
            learning_rate=eegnet_config.get("learning_rate", 1e-3),
            weight_decay=eegnet_config.get("weight_decay", 1e-3),
            patience=eegnet_config.get("patience", 50),
        )
        logger.info(
            "Created EEGNet classifier (n_channels=%d, n_samples=%d, "
            "n_classes=%d, F1=%d, D=%d, F2=%d)",
            n_channels, n_samples, n_classes,
            eegnet_config.get("F1", 8),
            eegnet_config.get("D", 2),
            eegnet_config.get("F2", 16),
        )
        return clf

    @staticmethod
    def _create_riemannian(config: dict) -> BaseClassifier:
        """Build a Riemannian MDM classifier from config.

        Reads:
            - ``classification.riemannian.metric`` for the geodesic metric
            - ``classification.riemannian.estimator`` for covariance
              estimation method

        Args:
            config: Full project configuration dictionary.

        Returns:
            Configured :class:`RiemannianClassifier`.
        """
        cls_config = config.get("classification", {})
        riemann_config = cls_config.get("riemannian", {})

        metric: str = riemann_config.get("metric", "riemann")
        estimator: str = riemann_config.get("estimator", "oas")

        clf = RiemannianClassifier(
            metric=metric,
            estimator=estimator,
        )
        logger.info(
            "Created Riemannian MDM classifier (metric=%s, estimator=%s)",
            metric, estimator,
        )
        return clf

    @staticmethod
    def _create_neural_sde(config: dict) -> BaseClassifier:
        """Build a Neural SDE classifier from config.

        Reads:
            - ``board.channel_count`` for number of EEG channels
            - ``training`` section for class count and epoch sizing
            - ``classification.neural_sde`` sub-section for SDE
              hyper-parameters (latent_dim, n_steps, dt, device)

        Args:
            config: Full project configuration dictionary.

        Returns:
            Configured :class:`NeuralSDEClassifier`.
        """
        from .neural_sde import NeuralSDEClassifier

        board_config = config.get("board", {})
        cls_config = config.get("classification", {})
        train_config = config.get("training", {})
        sde_config = cls_config.get("neural_sde", {})

        n_channels: int = board_config.get("channel_count", 16)
        n_classes: int = train_config.get("n_classes", 3)

        # Compute n_samples from sampling rate and classification window
        sampling_rate_raw = board_config.get("sampling_rate_override", None)
        if sampling_rate_raw is not None:
            sampling_rate: int = int(sampling_rate_raw)
        else:
            try:
                from brainflow.board_shim import BoardShim, BoardIds
                board_id = int(board_config.get("board_id", BoardIds.SYNTHETIC_BOARD))
                if board_id == -1:
                    board_id = BoardIds.SYNTHETIC_BOARD
                sampling_rate = int(BoardShim.get_sampling_rate(board_id))
            except Exception:
                sampling_rate = 250
        window_start: float = train_config.get("classification_window_start", 1.5)
        window_end: float = train_config.get("classification_window_end", 4.0)
        n_samples: int = int(sampling_rate * (window_end - window_start))

        clf = NeuralSDEClassifier(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            latent_dim=sde_config.get("latent_dim", 32),
            n_steps=sde_config.get("n_steps", 20),
            dt=sde_config.get("dt", 0.05),
            device=sde_config.get("device", "auto"),
        )
        logger.info(
            "Created NeuralSDE classifier (n_channels=%d, n_samples=%d, "
            "n_classes=%d, latent_dim=%d, n_steps=%d)",
            n_channels, n_samples, n_classes,
            sde_config.get("latent_dim", 32),
            sde_config.get("n_steps", 20),
        )
        return clf

    @staticmethod
    def _create_adaptive_router(config: dict) -> BaseClassifier:
        """Build an Adaptive Classifier Router from config.

        Creates unfitted CSP+LDA, EEGNet, and Riemannian classifiers via
        the existing factory methods, then wraps them in an
        :class:`AdaptiveClassifierRouter`.  The router's ``fit()`` will
        fit all three base classifiers.

        Reads:
            - All config sections used by the three base classifiers
            - ``classification.adaptive_router`` sub-section for routing
              thresholds and signal parameters

        Args:
            config: Full project configuration dictionary.

        Returns:
            Configured :class:`AdaptiveClassifierRouter`.
        """
        from .adaptive_router import AdaptiveClassifierRouter

        # Create the three base (unfitted) classifiers
        csp_lda = ClassifierFactory._create_csp_lda(config)
        eegnet = ClassifierFactory._create_eegnet(config)
        riemannian = ClassifierFactory._create_riemannian(config)

        classifiers = {
            "csp_lda": csp_lda,
            "eegnet": eegnet,
            "riemannian": riemannian,
        }

        cls_config = config.get("classification", {})
        router_config = cls_config.get("adaptive_router", {})

        clf = AdaptiveClassifierRouter(
            classifiers=classifiers,
            config=router_config,
        )
        logger.info(
            "Created AdaptiveClassifierRouter with experts: %s",
            list(classifiers.keys()),
        )
        return clf

    @staticmethod
    def list_available() -> list[str]:
        """Return list of supported model type strings.

        Returns:
            List of model type identifiers that can be passed to
            ``create()`` via ``config['classification']['model_type']``.
        """
        return list(ClassifierFactory._SUPPORTED_TYPES)
