"""Jacobian-SVD dynamical features for EEG signals.

Computes local dynamical properties of the EEG trajectory by estimating
the Jacobian matrix of the reconstructed state-space dynamics and
extracting features from its singular value decomposition.

The pipeline for each channel:
    1. Time-delay embedding: reconstruct attractor from scalar time series
    2. Local Jacobian estimation: linearise dynamics at each time point
    3. SVD of Jacobian: extract singular values
    4. Feature computation: Lyapunov exponents, fractal dimension, etc.

These features characterise the local stability, complexity, and
dimensionality of neural dynamics -- properties that change during
motor imagery as the brain transitions between dynamical regimes.

References:
    Eckmann, J.-P., & Ruelle, D. (1985). "Ergodic theory of chaos and
    strange attractors." Reviews of Modern Physics, 57(3), 617-656.

    Sauer, T., Yorke, J. A., & Casdagli, M. (1991). "Embedology."
    Journal of Statistical Physics, 65(3-4), 579-616.

    Kantz, H., & Schreiber, T. (2004). "Nonlinear Time Series Analysis."
    Cambridge University Press.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from scipy import linalg as la

logger = logging.getLogger(__name__)


class JacobianFeatureExtractor:
    """Extract dynamical features from EEG via Jacobian-SVD analysis.

    For each EEG channel, the signal is embedded into a reconstructed
    state space using time-delay embedding (Takens' theorem). Local
    Jacobian matrices are estimated via least squares, and their singular
    values yield features describing the local dynamics.

    Features per channel (5 values):
        - Local Lyapunov exponent (max): log(sigma_1), largest expansion rate
        - Kaplan-Yorke dimension: D_KY, effective attractor dimension
        - Condition number: sigma_1 / sigma_d, sensitivity of dynamics
        - Trace: sum(sigma_i), volume expansion/contraction rate
        - Spectral gap: sigma_1 - sigma_2, dominant direction strength

    Attributes:
        n_channels: Number of EEG channels to process.
        sf: Sampling frequency in Hz.
        embedding_dim: Dimension of the time-delay embedding.
        tau: Time delay for embedding (samples). If None, computed
            automatically via first minimum of mutual information.
    """

    # Number of features extracted per channel
    N_FEATURES_PER_CHANNEL = 5

    # Feature names in output order
    _FEATURE_NAMES = [
        "max_lyapunov",
        "kaplan_yorke_dim",
        "condition_number",
        "trace",
        "spectral_gap",
    ]

    def __init__(
        self,
        n_channels: int,
        sf: int = 125,
        embedding_dim: int = 3,
        tau: Optional[int] = None,
    ) -> None:
        """Initialise the Jacobian feature extractor.

        Args:
            n_channels: Number of EEG channels to extract features from.
            sf: Sampling frequency in Hz.
            embedding_dim: Dimension of the time-delay embedding space.
                Typically 3-7 for EEG (per Takens' theorem, should be
                > 2 * attractor_dimension).
            tau: Time delay in samples for embedding. If None, the
                optimal delay is estimated per channel using the first
                minimum of the auto-mutual information function.
        """
        if embedding_dim < 2:
            raise ValueError(
                f"embedding_dim must be >= 2, got {embedding_dim}"
            )
        if n_channels < 1:
            raise ValueError(
                f"n_channels must be >= 1, got {n_channels}"
            )

        self.n_channels = n_channels
        self.sf = sf
        self.embedding_dim = embedding_dim
        self.tau = tau

        logger.info(
            "JacobianFeatureExtractor: %d channels, dim=%d, tau=%s, "
            "%d features/channel",
            n_channels,
            embedding_dim,
            tau if tau is not None else "auto",
            self.N_FEATURES_PER_CHANNEL,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, epoch: np.ndarray) -> np.ndarray:
        """Extract Jacobian-SVD features from a multi-channel EEG epoch.

        Args:
            epoch: EEG data, shape (n_channels, n_samples). Each row is
                a single channel's time series.

        Returns:
            Feature vector, shape (n_channels * N_FEATURES_PER_CHANNEL,).
            Order: [ch0_max_lyap, ch0_ky_dim, ch0_cond, ch0_trace,
            ch0_gap, ch1_max_lyap, ...].

        Raises:
            ValueError: If epoch shape does not match expected n_channels.
        """
        if epoch.ndim != 2:
            raise ValueError(
                f"epoch must be 2D (n_channels, n_samples), "
                f"got shape {epoch.shape}"
            )
        if epoch.shape[0] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, "
                f"got {epoch.shape[0]}"
            )

        n_samples = epoch.shape[1]
        n_total = self.n_channels * self.N_FEATURES_PER_CHANNEL

        # Minimum samples needed for meaningful embedding + Jacobian
        min_samples = (self.embedding_dim + 1) * max(self.tau or 1, 1) + 10
        if n_samples < min_samples:
            logger.warning(
                "Epoch too short (%d samples) for embedding dim=%d. "
                "Need >= %d. Returning zeros.",
                n_samples, self.embedding_dim, min_samples,
            )
            return np.zeros(n_total, dtype=np.float64)

        all_features: List[np.ndarray] = []

        for ch_idx in range(self.n_channels):
            signal = epoch[ch_idx]

            # Handle degenerate signals
            if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                logger.warning(
                    "NaN/Inf in channel %d; returning zeros for this channel.",
                    ch_idx,
                )
                all_features.append(
                    np.zeros(self.N_FEATURES_PER_CHANNEL, dtype=np.float64)
                )
                continue

            if np.ptp(signal) == 0:
                logger.warning(
                    "Constant signal in channel %d; returning zeros.", ch_idx
                )
                all_features.append(
                    np.zeros(self.N_FEATURES_PER_CHANNEL, dtype=np.float64)
                )
                continue

            try:
                ch_features = self._extract_single_channel(signal)
                all_features.append(ch_features)
            except Exception as exc:
                logger.warning(
                    "Jacobian feature extraction failed for channel %d: %s. "
                    "Filling with zeros.",
                    ch_idx, exc,
                )
                all_features.append(
                    np.zeros(self.N_FEATURES_PER_CHANNEL, dtype=np.float64)
                )

        return np.concatenate(all_features)

    def get_feature_names(self) -> List[str]:
        """Get human-readable names for each output feature.

        Returns:
            List of feature name strings, length
            n_channels * N_FEATURES_PER_CHANNEL.
        """
        names: List[str] = []
        for ch_idx in range(self.n_channels):
            for feat_name in self._FEATURE_NAMES:
                names.append(f"ch{ch_idx}_{feat_name}")
        return names

    # ------------------------------------------------------------------
    # Private: per-channel extraction pipeline
    # ------------------------------------------------------------------

    def _extract_single_channel(self, signal: np.ndarray) -> np.ndarray:
        """Full pipeline for a single channel.

        Args:
            signal: 1D time series, shape (n_samples,).

        Returns:
            Feature vector, shape (N_FEATURES_PER_CHANNEL,).
        """
        # 1. Determine time delay
        tau = self.tau if self.tau is not None else self._optimal_tau(signal)
        tau = max(1, tau)  # ensure at least 1

        # 2. Time-delay embedding
        embedded = self._time_delay_embed(signal, self.embedding_dim, tau)

        # 3. Estimate local Jacobians and collect SVD features
        jacobians = self._estimate_jacobian(embedded, window_size=10)

        # 4. Aggregate SVD features across time points
        svd_feats_list = []
        for J in jacobians:
            svd_feats_list.append(self._svd_features(J))

        if len(svd_feats_list) == 0:
            return np.zeros(self.N_FEATURES_PER_CHANNEL, dtype=np.float64)

        # Average features over all time points for a stable estimate
        svd_feats_array = np.array(svd_feats_list)

        # Use median for robustness against outlier Jacobians
        return np.median(svd_feats_array, axis=0)

    # ------------------------------------------------------------------
    # Private: time-delay embedding
    # ------------------------------------------------------------------

    def _time_delay_embed(
        self, signal: np.ndarray, dim: int, tau: int
    ) -> np.ndarray:
        """Construct time-delay embedding matrix.

        Given a scalar time series x(t), constructs vectors:
            X(t) = [x(t), x(t - tau), x(t - 2*tau), ..., x(t - (dim-1)*tau)]

        Args:
            signal: 1D input, shape (n_samples,).
            dim: Embedding dimension.
            tau: Time delay in samples.

        Returns:
            Embedded matrix, shape (n_vectors, dim), where
            n_vectors = n_samples - (dim - 1) * tau.

        Raises:
            ValueError: If signal is too short for the given dim and tau.
        """
        n = len(signal)
        n_vectors = n - (dim - 1) * tau

        if n_vectors < 2:
            raise ValueError(
                f"Signal too short ({n} samples) for embedding "
                f"dim={dim}, tau={tau}. Need at least "
                f"{(dim - 1) * tau + 2} samples."
            )

        # Build embedding matrix efficiently using stride tricks
        embedded = np.empty((n_vectors, dim), dtype=np.float64)
        for d in range(dim):
            offset = d * tau
            embedded[:, d] = signal[offset: offset + n_vectors]

        return embedded

    # ------------------------------------------------------------------
    # Private: local Jacobian estimation
    # ------------------------------------------------------------------

    def _estimate_jacobian(
        self, embedded: np.ndarray, window_size: int = 10
    ) -> np.ndarray:
        """Estimate local Jacobian matrices via least squares.

        At each time point t, fits a linear map J such that:
            X(t+1) ~ J * X(t)
        using data from a local window [t - w, t + w].

        Args:
            embedded: Time-delay embedded data, shape (n_vectors, dim).
            window_size: Half-width of the local fitting window.

        Returns:
            Array of Jacobian matrices, shape (n_jacobians, dim, dim).
        """
        n_vectors, dim = embedded.shape
        jacobians = []

        # We need at least window_size + 1 points on each side, plus
        # the successor point
        start = window_size
        end = n_vectors - window_size - 1

        if start >= end:
            # Not enough data for local estimation; use global Jacobian
            logger.debug(
                "Window too large for signal length; using global Jacobian."
            )
            X_curr = embedded[:-1]
            X_next = embedded[1:]
            try:
                J, _, _, _ = la.lstsq(X_curr, X_next)
                jacobians.append(J.T)
            except la.LinAlgError:
                jacobians.append(np.eye(dim))
            return np.array(jacobians)

        # Subsample time points for efficiency (every 5th point)
        step = max(1, (end - start) // 50)

        for t in range(start, end, step):
            # Local window
            lo = max(0, t - window_size)
            hi = min(n_vectors - 1, t + window_size)

            X_curr = embedded[lo:hi]
            X_next = embedded[lo + 1: hi + 1]

            if X_curr.shape[0] < dim:
                # Underdetermined system, skip
                continue

            try:
                # Solve X_next = X_curr @ J^T  =>  J^T = lstsq(X_curr, X_next)
                J_T, _, _, _ = la.lstsq(X_curr, X_next)
                J = J_T.T
                jacobians.append(J)
            except la.LinAlgError:
                # Singular matrix, skip this time point
                continue

        if len(jacobians) == 0:
            jacobians.append(np.eye(dim))

        return np.array(jacobians)

    # ------------------------------------------------------------------
    # Private: SVD-based features from a single Jacobian
    # ------------------------------------------------------------------

    def _svd_features(self, jacobian: np.ndarray) -> np.ndarray:
        """Extract dynamical features from the SVD of a Jacobian matrix.

        Args:
            jacobian: Square matrix, shape (dim, dim).

        Returns:
            Feature vector of length N_FEATURES_PER_CHANNEL:
                [max_lyapunov, kaplan_yorke_dim, condition_number,
                 trace, spectral_gap]
        """
        try:
            sigma = la.svd(jacobian, compute_uv=False)
        except la.LinAlgError:
            return np.zeros(self.N_FEATURES_PER_CHANNEL, dtype=np.float64)

        # Ensure sorted descending
        sigma = np.sort(sigma)[::-1]

        # Clamp very small singular values to avoid log(0)
        sigma_safe = np.maximum(sigma, 1e-12)

        # 1. Local Lyapunov exponents: lambda_i = log(sigma_i)
        lyapunov_exponents = np.log(sigma_safe)
        max_lyapunov = lyapunov_exponents[0]

        # 2. Kaplan-Yorke dimension
        kaplan_yorke_dim = self._kaplan_yorke_dimension(lyapunov_exponents)

        # 3. Condition number: sigma_1 / sigma_d
        condition_number = sigma_safe[0] / sigma_safe[-1]
        # Cap at a reasonable value to avoid infinities
        condition_number = min(condition_number, 1e6)

        # 4. Trace: sum of singular values (related to volume change)
        trace = np.sum(sigma)

        # 5. Spectral gap: sigma_1 - sigma_2 (dominant direction strength)
        if len(sigma) >= 2:
            spectral_gap = sigma[0] - sigma[1]
        else:
            spectral_gap = sigma[0]

        features = np.array(
            [max_lyapunov, kaplan_yorke_dim, condition_number,
             trace, spectral_gap],
            dtype=np.float64,
        )

        # Replace any NaN/Inf with 0
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        return features

    @staticmethod
    def _kaplan_yorke_dimension(lyapunov_exponents: np.ndarray) -> float:
        """Compute the Kaplan-Yorke dimension from Lyapunov exponents.

        D_KY = j + (sum_{i=1}^{j} lambda_i) / |lambda_{j+1}|

        where j is the largest index such that the sum of the first j
        exponents is non-negative.

        Args:
            lyapunov_exponents: Sorted (descending) Lyapunov exponents.

        Returns:
            Kaplan-Yorke dimension (float).
        """
        n = len(lyapunov_exponents)
        cumsum = np.cumsum(lyapunov_exponents)

        # Find j: largest index where cumulative sum >= 0
        j = 0
        for i in range(n):
            if cumsum[i] >= 0:
                j = i + 1
            else:
                break

        if j == 0:
            # All exponents negative: dimension is 0
            return 0.0

        if j >= n:
            # All sums positive: dimension equals embedding dimension
            return float(n)

        # D_KY = j + sum_{i=0}^{j-1} lambda_i / |lambda_j|
        denom = abs(lyapunov_exponents[j])
        if denom < 1e-12:
            return float(j)

        d_ky = j + cumsum[j - 1] / denom
        return float(d_ky)

    # ------------------------------------------------------------------
    # Private: optimal time delay via mutual information
    # ------------------------------------------------------------------

    def _optimal_tau(self, signal: np.ndarray, max_tau: int = 50) -> int:
        """Estimate optimal time delay via first minimum of auto-MI.

        Uses a histogram-based estimator of mutual information between
        x(t) and x(t + tau) for increasing tau values. The first local
        minimum indicates decorrelation without loss of dynamical
        coupling (Fraser & Swinney, 1986).

        Args:
            signal: 1D time series.
            max_tau: Maximum delay to search.

        Returns:
            Optimal time delay in samples (>= 1).
        """
        n = len(signal)
        max_tau = min(max_tau, n // 4)  # don't exceed quarter of signal

        if max_tau < 2:
            return 1

        n_bins = max(10, int(np.sqrt(n / 5)))

        mi_values = np.zeros(max_tau)

        for tau_candidate in range(1, max_tau):
            x = signal[: n - tau_candidate]
            y = signal[tau_candidate:]

            # 2D histogram for joint distribution
            try:
                hist_2d, _, _ = np.histogram2d(x, y, bins=n_bins)
                # Normalise to joint probability
                p_xy = hist_2d / hist_2d.sum()
                # Marginals
                p_x = p_xy.sum(axis=1)
                p_y = p_xy.sum(axis=0)

                # Mutual information: sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
                mi = 0.0
                for i in range(n_bins):
                    for j in range(n_bins):
                        if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                            mi += p_xy[i, j] * np.log(
                                p_xy[i, j] / (p_x[i] * p_y[j])
                            )
                mi_values[tau_candidate] = mi
            except Exception:
                mi_values[tau_candidate] = 0.0

        # Find first local minimum (tau >= 1)
        for tau_candidate in range(2, max_tau - 1):
            if (
                mi_values[tau_candidate] < mi_values[tau_candidate - 1]
                and mi_values[tau_candidate] <= mi_values[tau_candidate + 1]
            ):
                return tau_candidate

        # Fallback: use the tau with smallest MI
        best = np.argmin(mi_values[1:]) + 1
        return int(best)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"JacobianFeatureExtractor(n_channels={self.n_channels}, "
            f"sf={self.sf}, embedding_dim={self.embedding_dim}, "
            f"tau={self.tau})"
        )
