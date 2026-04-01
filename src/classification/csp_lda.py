"""CSP + shrinkage-LDA classifier for motor imagery BCI.

Uses Common Spatial Patterns (CSP) for spatial filtering followed by
Linear Discriminant Analysis with automatic Ledoit-Wolf shrinkage.
The decision_function output provides continuous scores suitable for
proportional cursor control.

References:
    Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000).
    Optimal Spatial Filtering of Single Trial EEG During Imagined
    Hand Movement. IEEE Trans. Rehab. Eng., 8(4), 441-446.

    Blankertz, B., Tomioka, R., Lemm, S., Kawanabe, M., & Müller, K.-R.
    (2008). Optimizing Spatial filters for Robust EEG Single-Trial
    Analysis. IEEE Signal Processing Magazine, 25(1), 41-56.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from mne.decoding import CSP
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline as SklearnPipeline

from .base import BaseClassifier

logger = logging.getLogger(__name__)


class CSPLDAClassifier(BaseClassifier):
    """Common Spatial Patterns + shrinkage LDA classifier.

    This is the recommended starting classifier for motor imagery BCI.
    CSP extracts spatial filters that maximise variance differences
    between classes, and shrinkage LDA provides a regularised linear
    decision boundary that works well with small training sets.

    The ``decision_function`` method returns continuous signed distances
    to the decision boundary (2-class) or per-class scores (multi-class),
    which map directly to proportional cursor velocity.

    Args:
        n_components: Number of CSP components to extract. For *C*
            channels, at most *C* components are possible. A value of
            12 (6 pairs) is a good default for 16-channel Cyton+Daisy.
        reg: CSP regularisation estimator. ``'ledoit_wolf'`` provides
            automatic shrinkage and is robust to small sample sizes.
        shrinkage: LDA shrinkage parameter. ``'auto'`` uses the
            Ledoit-Wolf lemma to choose the optimal shrinkage intensity.
    """

    def __init__(
        self,
        n_components: int = 12,
        reg: str = "ledoit_wolf",
        shrinkage: str = "auto",
    ) -> None:
        self.n_components: int = n_components
        self.reg: str = reg
        self.shrinkage: str = shrinkage
        self._calibrator = None
        self._pipeline: SklearnPipeline = SklearnPipeline([
            ("csp", CSP(n_components=self.n_components, reg=self.reg, log=True, norm_trace=True)),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage=self.shrinkage)),
        ])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_3d(X: np.ndarray) -> np.ndarray:
        """Ensure input is 3-D (n_trials, n_channels, n_samples).

        If a single trial is passed as a 2-D array ``(n_channels,
        n_samples)``, it is reshaped to ``(1, n_channels, n_samples)``.

        Args:
            X: Input EEG array, either 2-D or 3-D.

        Returns:
            3-D array with shape ``(n_trials, n_channels, n_samples)``.

        Raises:
            ValueError: If *X* has fewer than 2 or more than 3
                dimensions.
        """
        if X.ndim == 2:
            return X[np.newaxis, :, :]
        if X.ndim == 3:
            return X
        raise ValueError(
            f"Expected 2-D or 3-D input, got {X.ndim}-D array with "
            f"shape {X.shape}"
        )

    # ------------------------------------------------------------------
    # BaseClassifier interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> CSPLDAClassifier:
        """Train the CSP+LDA pipeline.

        Builds a two-stage sklearn ``Pipeline``:

        1. **CSP** — learns spatial filters from the training data.
        2. **LDA** — fits a shrinkage-regularised linear discriminant.

        Args:
            X: Training epochs, shape ``(n_trials, n_channels,
                n_samples)``.
            y: Integer class labels, shape ``(n_trials,)``.

        Returns:
            ``self``, to allow chaining (e.g. ``clf.fit(X, y).predict(X)``).
        """
        X = self._ensure_3d(X)

        n_trials, n_channels, n_samples = X.shape
        n_classes = len(np.unique(y))

        if n_classes < 2:
            raise ValueError(
                f"Need at least 2 classes for CSP, got {n_classes}"
            )

        if n_trials < self.n_components:
            logger.warning(
                "Fewer trials (%d) than CSP components (%d). "
                "Reducing n_components.",
                n_trials,
                self.n_components,
            )
            self.n_components = max(2, n_trials - 1)
            # Rebuild pipeline with adjusted n_components
            self._pipeline = SklearnPipeline([
                ("csp", CSP(
                    n_components=self.n_components,
                    reg=self.reg,
                    log=True,
                    norm_trace=True,
                )),
                ("lda", LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage=self.shrinkage,
                )),
            ])

        min_trials_per_class = int(min(np.bincount(y)))
        if min_trials_per_class < 5:
            smallest_class = int(np.argmin(np.bincount(y)))
            logger.warning(
                "Class %d has only %d trials. Results may be unreliable.",
                smallest_class,
                min_trials_per_class,
            )

        self._pipeline.fit(X, y)

        # Probability calibration via cross-validation (sigmoid/Platt scaling).
        # LDA's predict_proba is based on class-conditional Gaussians which
        # can be poorly calibrated for small BCI datasets. This wraps the
        # fitted pipeline in a calibrator for more reliable confidence scores,
        # which improves adaptive routing and SEAL reward quality.
        n_cv = min(3, min_trials_per_class)
        if n_cv >= 2:
            try:
                self._calibrator = CalibratedClassifierCV(
                    estimator=self._pipeline,
                    method="sigmoid",
                    cv=n_cv,
                )
                self._calibrator.fit(X, y)
                logger.info(
                    "CSP+LDA probability calibration fitted (cv=%d, sigmoid)",
                    n_cv,
                )
            except Exception as exc:
                self._calibrator = None
                logger.debug("Calibration failed (non-critical): %s", exc)
        else:
            self._calibrator = None

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for one or more trials.

        Args:
            X: Input epochs, shape ``(n_trials, n_channels, n_samples)``
                or ``(n_channels, n_samples)`` for a single trial.

        Returns:
            Predicted integer labels.  Shape ``(n_trials,)`` for batch
            input, or scalar for a single trial.

        Raises:
            RuntimeError: If the classifier has not been fitted yet.
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")

        X = self._ensure_3d(X)
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input epochs, shape ``(n_trials, n_channels, n_samples)``
                or ``(n_channels, n_samples)`` for a single trial.

        Returns:
            Class probabilities, shape ``(n_trials, n_classes)``.

        Raises:
            RuntimeError: If the classifier has not been fitted yet.
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")

        X = self._ensure_3d(X)
        # Use calibrated probabilities when available
        if hasattr(self, '_calibrator') and self._calibrator is not None:
            return self._calibrator.predict_proba(X)
        return self._pipeline.predict_proba(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute continuous decision scores for proportional control.

        For 2-class problems this returns a signed scalar per trial
        (distance to the decision hyperplane).  For multi-class
        problems it returns per-class scores.

        These continuous values are the primary output used for
        proportional cursor velocity mapping.

        Args:
            X: Input epochs, shape ``(n_trials, n_channels, n_samples)``
                or ``(n_channels, n_samples)`` for a single trial.

        Returns:
            Decision scores.  Shape ``(n_trials,)`` for binary,
            ``(n_trials, n_classes)`` for multi-class.

        Raises:
            RuntimeError: If the classifier has not been fitted yet.
        """
        if self._pipeline is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")

        X = self._ensure_3d(X)
        return self._pipeline.decision_function(X)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        from sklearn.utils.validation import check_is_fitted
        try:
            check_is_fitted(self._pipeline)
            fitted = True
        except Exception:
            fitted = False
        return (
            f"CSPLDAClassifier(n_components={self.n_components}, "
            f"reg='{self.reg}', shrinkage='{self.shrinkage}', "
            f"fitted={fitted})"
        )
