"""Common Spatial Patterns (CSP) feature extraction.

Wraps MNE's CSP implementation for use in the Mental Mouse pipeline.
CSP learns spatial filters that maximize variance for one class while
minimizing it for another — the gold standard for MI-based BCIs.

References:
    Blankertz et al. (2008) "Optimizing Spatial Filters for Robust
    EEG Single-Trial Analysis." IEEE Signal Processing Magazine.
"""

import logging
from typing import Optional

import numpy as np
from mne.decoding import CSP

logger = logging.getLogger(__name__)


class CSPExtractor:
    """Common Spatial Patterns feature extractor.

    Learns spatial filters that maximize the difference in band-power
    between two (or more) classes. Works on epoched, band-pass filtered
    EEG data. The resulting log-variance features are highly discriminative
    for motor imagery tasks.

    Attributes:
        n_components: Number of CSP components (filters) to retain.
        reg: Regularization method for covariance estimation.
        log: Whether to apply log transform to the variance features.
        csp_: Fitted MNE CSP object (available after fit).
    """

    def __init__(
        self,
        n_components: int = 12,
        reg: str = "ledoit_wolf",
        log: bool = True,
    ) -> None:
        """Initialize CSP extractor.

        Args:
            n_components: Number of CSP components to extract. Should be
                even (pairs of most/least discriminative filters). For a
                16-channel setup, 12 is a reasonable default (6 pairs).
            reg: Covariance regularization. Options: 'ledoit_wolf',
                'oas', 'shrunk', 'empirical', or None. Ledoit-Wolf is
                recommended for small sample sizes typical of BCI.
            log: If True, apply log transformation to variance features.
                This improves the Gaussianity of features for LDA.
        """
        self.n_components = n_components
        self.reg = reg
        self.log = log

        self.csp_: Optional[CSP] = None
        self._is_fitted = False

        self._csp = CSP(
            n_components=self.n_components,
            reg=self.reg,
            log=self.log,
            norm_trace=True,
        )

        logger.info(
            "CSPExtractor initialized: n_components=%d, reg=%s, log=%s",
            n_components,
            reg,
            log,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CSPExtractor":
        """Fit CSP spatial filters from labeled training data.

        Computes spatial filters that maximize the ratio of class-conditional
        variances. The first n_components/2 filters maximize variance for
        class 1, the last n_components/2 maximize variance for class 2.

        Args:
            X: Training epochs, shape (n_trials, n_channels, n_samples).
                Should be band-pass filtered (e.g., 8-30 Hz for mu+beta).
            y: Class labels, shape (n_trials,). Binary or multi-class.

        Returns:
            self, for method chaining.

        Raises:
            ValueError: If X and y have incompatible shapes, or if
                n_components exceeds the number of channels.
        """
        if X.ndim != 3:
            raise ValueError(
                f"X must be 3D (n_trials, n_channels, n_samples), "
                f"got shape {X.shape}"
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} trials but y has {y.shape[0]} labels"
            )
        if self.n_components > X.shape[1]:
            raise ValueError(
                f"n_components ({self.n_components}) exceeds number of "
                f"channels ({X.shape[1]})"
            )

        # Guard: CSP needs at least 2 classes
        n_classes = len(np.unique(y))
        if n_classes < 2:
            raise ValueError(
                f"CSP requires at least 2 classes, but training data "
                f"contains only {n_classes} class(es): {np.unique(y).tolist()}. "
                f"Ensure the calibration session collected data for all classes."
            )

        # Guard: fewer trials than n_components (covariance will be rank-deficient)
        if X.shape[0] < self.n_components:
            logger.warning(
                "Fewer trials (%d) than n_components (%d). "
                "CSP may produce unreliable filters. Consider collecting "
                "more data or reducing n_components.",
                X.shape[0],
                self.n_components,
            )

        # Guard: NaN in input epochs
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning(
                "NaN/Inf detected in CSP training data; "
                "replacing with zeros."
            )
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(
            "Fitting CSP on %d trials, %d channels, %d samples",
            X.shape[0],
            X.shape[1],
            X.shape[2],
        )

        self._csp.fit(X, y)
        self.csp_ = self._csp
        self._is_fitted = True

        logger.info("CSP fitting complete. Filters shape: %s", self.csp_.filters_.shape)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply learned CSP filters and extract features.

        Projects epochs through the spatial filters and computes
        log-variance (if log=True) or variance features.

        Args:
            X: Epochs to transform, shape (n_trials, n_channels, n_samples).
                Must have the same number of channels as the training data.

        Returns:
            Feature matrix, shape (n_trials, n_components). Each column
            is the (log-)variance of one CSP-filtered signal.

        Raises:
            RuntimeError: If called before fit().
            ValueError: If X has wrong number of dimensions or channels.
        """
        if not self._is_fitted:
            raise RuntimeError("CSPExtractor has not been fitted. Call fit() first.")

        # Handle single trial: 2D input (n_channels, n_samples) -> expand
        _squeezed = False
        if X.ndim == 2:
            logger.debug(
                "CSP transform received 2D input (single trial); "
                "expanding to 3D."
            )
            X = X[np.newaxis, :, :]
            _squeezed = True

        if X.ndim != 3:
            raise ValueError(
                f"X must be 3D (n_trials, n_channels, n_samples), "
                f"got shape {X.shape}"
            )

        # Guard: NaN in input
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            logger.warning(
                "NaN/Inf detected in CSP transform input; "
                "replacing with zeros."
            )
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        features = self._csp.transform(X)

        # If we expanded a single trial, squeeze back
        if _squeezed:
            features = features[0]

        logger.debug(
            "CSP transform: %s trials -> features shape %s",
            X.shape[0],
            features.shape,
        )
        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit CSP filters and transform data in one step.

        Convenience method equivalent to fit(X, y).transform(X).

        Args:
            X: Training epochs, shape (n_trials, n_channels, n_samples).
            y: Class labels, shape (n_trials,).

        Returns:
            Feature matrix, shape (n_trials, n_components).
        """
        self.fit(X, y)
        return self.transform(X)

    def get_spatial_filters(self) -> np.ndarray:
        """Return the learned CSP spatial filters.

        The filters matrix can be used for topographic visualization
        (e.g., plotting the spatial pattern of each component on the
        scalp to verify that motor cortex regions are highlighted).

        Returns:
            Spatial filters, shape (n_components, n_channels). Each row
            is a spatial filter that can be applied to the channel data.

        Raises:
            RuntimeError: If called before fit().
        """
        if not self._is_fitted:
            raise RuntimeError("CSPExtractor has not been fitted. Call fit() first.")

        return self._csp.filters_[: self.n_components]

    def __repr__(self) -> str:
        fitted_str = "fitted" if self._is_fitted else "not fitted"
        return (
            f"CSPExtractor(n_components={self.n_components}, "
            f"reg='{self.reg}', log={self.log}, {fitted_str})"
        )
