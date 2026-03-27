"""Chaos / nonlinear feature extraction using the antropy library.

Extracts information-theoretic and fractal dimension features from EEG
signals. These features capture the complexity and regularity of neural
dynamics, complementing the variance-based CSP features for MI classification.

References:
    Lotte et al. (2018) "A review of classification algorithms for
    EEG-based brain-computer interfaces: a 10 year update."
    Journal of Neural Engineering.
"""

import logging
from typing import Dict, List, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Attempt to import antropy; degrade gracefully if unavailable.
try:
    import antropy as ant

    _ANTROPY_AVAILABLE = True
except ImportError:
    _ANTROPY_AVAILABLE = False
    logger.warning(
        "antropy is not installed. ChaosFeatureExtractor will return empty "
        "feature vectors. Install with: pip install antropy"
    )


class ChaosFeatureExtractor:
    """Nonlinear / complexity feature extractor for EEG signals.

    Wraps the antropy library to compute entropy measures, Hjorth
    parameters, and fractal dimension estimates on single-channel
    EEG windows. Designed for motor cortex channels where complexity
    changes during motor imagery.

    Attributes:
        features: List of feature names to compute.
        sf: Sampling frequency in Hz.
    """

    # Maps feature name -> (callable, number of output values).
    # Built once at class level, populated per-instance to capture sf.

    def __init__(
        self,
        features: List[str],
        sf: int = 125,
    ) -> None:
        """Initialize chaos feature extractor.

        Args:
            features: Which features to compute. Supported names:
                "hjorth"           - Hjorth mobility & complexity (2 values)
                "perm_entropy"     - Permutation entropy (1 value)
                "spectral_entropy" - Spectral entropy via Welch (1 value)
                "sample_entropy"   - Sample entropy (1 value)
                "higuchi_fd"       - Higuchi fractal dimension (1 value)
                "petrosian_fd"     - Petrosian fractal dimension (1 value)
                "katz_fd"          - Katz fractal dimension (1 value)
                "svd_entropy"      - SVD entropy (1 value)
                "dfa"              - Detrended fluctuation analysis (1 value)
            sf: Sampling frequency in Hz. Must match the data's actual
                sampling rate (125 Hz for Cyton+Daisy after downsampling).

        Raises:
            ValueError: If an unsupported feature name is provided.
        """
        self.sf = sf
        self.features = list(features)

        # Validate feature names and build the dispatch table.
        self._dispatch: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        self._feature_sizes: Dict[str, int] = {}
        self._build_dispatch()

        for feat in self.features:
            if feat not in self._dispatch:
                supported = sorted(self._dispatch.keys())
                raise ValueError(
                    f"Unknown feature '{feat}'. Supported: {supported}"
                )

        total = sum(self._feature_sizes[f] for f in self.features)
        logger.info(
            "ChaosFeatureExtractor initialized: %d features -> %d values per channel",
            len(self.features),
            total,
        )

    def _build_dispatch(self) -> None:
        """Build the feature name -> extraction function mapping.

        Each function takes a 1D signal and returns a 1D numpy array
        of feature values. We capture self.sf in closures where needed.
        """
        if not _ANTROPY_AVAILABLE:
            return

        sf = self.sf

        def _hjorth(signal: np.ndarray) -> np.ndarray:
            mobility, complexity = ant.hjorth_params(signal)
            return np.array([mobility, complexity], dtype=np.float64)

        def _perm_entropy(signal: np.ndarray) -> np.ndarray:
            pe = ant.perm_entropy(signal, order=3, normalize=True)
            return np.array([pe], dtype=np.float64)

        def _spectral_entropy(signal: np.ndarray) -> np.ndarray:
            se = ant.spectral_entropy(signal, sf=sf, method="welch", normalize=True)
            return np.array([se], dtype=np.float64)

        def _sample_entropy(signal: np.ndarray) -> np.ndarray:
            sampen = ant.sample_entropy(signal)
            return np.array([sampen], dtype=np.float64)

        def _higuchi_fd(signal: np.ndarray) -> np.ndarray:
            hfd = ant.higuchi_fd(signal)
            return np.array([hfd], dtype=np.float64)

        def _petrosian_fd(signal: np.ndarray) -> np.ndarray:
            pfd = ant.petrosian_fd(signal)
            return np.array([pfd], dtype=np.float64)

        def _katz_fd(signal: np.ndarray) -> np.ndarray:
            kfd = ant.katz_fd(signal)
            return np.array([kfd], dtype=np.float64)

        def _svd_entropy(signal: np.ndarray) -> np.ndarray:
            svd = ant.svd_entropy(signal, order=3, normalize=True)
            return np.array([svd], dtype=np.float64)

        def _dfa(signal: np.ndarray) -> np.ndarray:
            dfa_val = ant.detrended_fluctuation(signal)
            return np.array([dfa_val], dtype=np.float64)

        self._dispatch = {
            "hjorth": _hjorth,
            "perm_entropy": _perm_entropy,
            "spectral_entropy": _spectral_entropy,
            "sample_entropy": _sample_entropy,
            "higuchi_fd": _higuchi_fd,
            "petrosian_fd": _petrosian_fd,
            "katz_fd": _katz_fd,
            "svd_entropy": _svd_entropy,
            "dfa": _dfa,
        }

        self._feature_sizes = {
            "hjorth": 2,
            "perm_entropy": 1,
            "spectral_entropy": 1,
            "sample_entropy": 1,
            "higuchi_fd": 1,
            "petrosian_fd": 1,
            "katz_fd": 1,
            "svd_entropy": 1,
            "dfa": 1,
        }

    @property
    def n_features_per_channel(self) -> int:
        """Number of feature values produced per channel."""
        if not _ANTROPY_AVAILABLE:
            return 0
        return sum(self._feature_sizes.get(f, 0) for f in self.features)

    def extract_single_channel(self, window: np.ndarray) -> np.ndarray:
        """Extract chaos/nonlinear features from a single-channel window.

        Args:
            window: 1D EEG signal, shape (n_samples,). Should be
                pre-filtered (e.g., 1-40 Hz bandpass).

        Returns:
            Feature vector, shape (n_features,). Order follows
            self.features list. Returns empty array if antropy
            is not available.

        Raises:
            ValueError: If window is not 1D.
        """
        if window.ndim != 1:
            raise ValueError(
                f"window must be 1D (n_samples,), got shape {window.shape}"
            )

        if not _ANTROPY_AVAILABLE:
            logger.warning("antropy unavailable, returning empty features.")
            return np.array([], dtype=np.float64)

        n_total = self.n_features_per_channel

        # Guard: empty window
        if window.shape[0] == 0:
            logger.warning(
                "extract_single_channel received empty window; "
                "returning zeros."
            )
            return np.zeros(n_total, dtype=np.float64)

        # Guard: NaN / Inf input — antropy will produce NaN
        if np.any(np.isnan(window)) or np.any(np.isinf(window)):
            logger.warning(
                "NaN/Inf detected in chaos feature input; "
                "replacing with zeros."
            )
            window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

        # Guard: very short window — entropy measures are unreliable
        _MIN_SAMPLES_CHAOS = 10
        if window.shape[0] < _MIN_SAMPLES_CHAOS:
            logger.warning(
                "Window too short for reliable entropy estimation "
                "(%d samples, need >= %d). Returning zeros.",
                window.shape[0],
                _MIN_SAMPLES_CHAOS,
            )
            return np.zeros(n_total, dtype=np.float64)

        # Guard: constant signal (zero variance) — entropy is undefined
        if np.ptp(window) == 0:
            logger.warning(
                "Constant signal (zero variance) detected; "
                "entropy is undefined. Returning zeros."
            )
            return np.zeros(n_total, dtype=np.float64)

        parts: List[np.ndarray] = []
        for feat_name in self.features:
            try:
                values = self._dispatch[feat_name](window)
                # Replace any NaN/Inf in output with 0.0
                if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                    logger.warning(
                        "Feature '%s' produced NaN/Inf; replacing with 0.0.",
                        feat_name,
                    )
                    values = np.nan_to_num(
                        values, nan=0.0, posinf=0.0, neginf=0.0
                    )
                parts.append(values)
            except Exception as exc:
                # If a single feature computation fails (e.g., signal too
                # short for sample_entropy), fill with 0.0 and log warning.
                n_vals = self._feature_sizes[feat_name]
                logger.warning(
                    "Feature '%s' failed: %s. Filling %d value(s) with 0.0.",
                    feat_name,
                    exc,
                    n_vals,
                )
                parts.append(np.zeros(n_vals, dtype=np.float64))

        return np.concatenate(parts)

    def extract_multi_channel(
        self,
        data: np.ndarray,
        channel_indices: List[int],
    ) -> np.ndarray:
        """Extract chaos features from multiple EEG channels.

        Applies extract_single_channel to each specified channel and
        concatenates the results into a flat feature vector.

        Args:
            data: Multi-channel EEG data, shape (n_channels, n_samples).
            channel_indices: Which channels to extract features from.
                Typically motor cortex channels (C3, C4, Cz, etc.).

        Returns:
            Flat feature vector, shape (n_channels_selected * n_features,).
            Order: [ch0_feat0, ch0_feat1, ..., ch1_feat0, ch1_feat1, ...].
            Returns empty array if antropy is not available.

        Raises:
            ValueError: If data is not 2D.
            IndexError: If any channel index is out of bounds.
        """
        if data.ndim != 2:
            raise ValueError(
                f"data must be 2D (n_channels, n_samples), got shape {data.shape}"
            )

        if not _ANTROPY_AVAILABLE:
            logger.warning("antropy unavailable, returning empty features.")
            return np.array([], dtype=np.float64)

        channel_features: List[np.ndarray] = []
        for ch_idx in channel_indices:
            if ch_idx < 0 or ch_idx >= data.shape[0]:
                raise IndexError(
                    f"Channel index {ch_idx} out of bounds for data "
                    f"with {data.shape[0]} channels"
                )
            ch_feats = self.extract_single_channel(data[ch_idx])
            channel_features.append(ch_feats)

        return np.concatenate(channel_features)

    def get_feature_names(self, channel_indices: Optional[List[int]] = None) -> List[str]:
        """Get human-readable names for each output feature.

        Useful for debugging, feature importance analysis, and logging.

        Args:
            channel_indices: If provided, generates names for multi-channel
                output (e.g., "ch0_hjorth_mobility"). If None, generates
                names for single-channel output.

        Returns:
            List of feature name strings.
        """
        single_names: List[str] = []
        for feat_name in self.features:
            if feat_name == "hjorth":
                single_names.extend(["hjorth_mobility", "hjorth_complexity"])
            else:
                single_names.append(feat_name)

        if channel_indices is None:
            return single_names

        multi_names: List[str] = []
        for ch_idx in channel_indices:
            for name in single_names:
                multi_names.append(f"ch{ch_idx}_{name}")
        return multi_names

    def __repr__(self) -> str:
        avail = "available" if _ANTROPY_AVAILABLE else "NOT available"
        return (
            f"ChaosFeatureExtractor(features={self.features}, "
            f"sf={self.sf}, antropy={avail})"
        )
