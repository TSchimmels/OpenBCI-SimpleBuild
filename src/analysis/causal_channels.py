"""DAGMA-inspired causal discovery for EEG channel interactions.

Instead of hardcoding which channels are important for motor imagery,
this module discovers per-subject causal structure from calibration data.
Each EEG channel is a node in a directed acyclic graph (DAG), and edges
represent causal influence (e.g., C3 -> P3 means C3 drives P3 during
left-hand imagery). Different MI classes produce different causal
structures, and the discovered graph tells us which channels to focus
on per class.

The causal discovery uses a simplified variant of the DAGMA algorithm:
    1. Compute time-lagged cross-correlation between all channel pairs
    2. Apply thresholding to get candidate edges
    3. Enforce DAG constraint via DAGMA's acyclicity penalty:
       h(W) = tr(e^(W o W)) - d = 0
    4. Optimize: min ||X - XW||^2 + lambda_1 * |W| + alpha * h(W)

Reference:
    Bello, K., Aragam, B., & Ravikumar, P. (2022). DAGMA: Learning
    DAGs via M-matrices and a Log-Determinant Acyclicity
    Characterization. NeurIPS 2022.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_EPOCHS_PER_CLASS = 5
_DEFAULT_MAX_LAG_SEC = 0.1
_DEFAULT_LAMBDA1 = 0.05
_DEFAULT_ALPHA_INIT = 1.0
_DEFAULT_ALPHA_MULT = 10.0
_DEFAULT_LR = 0.003
_DEFAULT_MAX_ITER = 300
_DEFAULT_H_TOL = 1e-8
_DEFAULT_THRESHOLD = 0.10


class CausalChannelDiscovery:
    """Discover causal DAG structure among EEG channels per MI class.

    For each motor-imagery class, the algorithm discovers a weighted
    directed acyclic graph whose edges encode which channels causally
    drive which other channels. This replaces hard-coded channel
    selection with a data-driven, subject-specific approach.

    Args:
        n_channels: Number of EEG channels.
        sf: Sampling frequency in Hz.
        channel_names: Human-readable channel names (e.g. ``["C3", "C4", ...]``).
            If *None*, channels are labelled ``ch0, ch1, ...``.
        class_names: Names of MI classes (e.g. ``["left", "right", "rest"]``).
            If *None*, classes are labelled ``class0, class1, ...``.
        max_lag_sec: Maximum time lag (seconds) for cross-correlation.
        lambda1: L1 sparsity penalty weight.
        alpha_init: Initial weight on the acyclicity penalty.
        alpha_mult: Multiplier applied to *alpha* each outer iteration.
        lr: Learning rate for gradient descent.
        max_iter: Maximum gradient-descent steps per outer iteration.
        h_tol: Convergence tolerance for the acyclicity constraint h(W).
        threshold: Absolute-value threshold applied to the final weight
            matrix; edges with ``|w_ij| < threshold`` are pruned.
    """

    def __init__(
        self,
        n_channels: int,
        sf: int = 125,
        channel_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        max_lag_sec: float = _DEFAULT_MAX_LAG_SEC,
        lambda1: float = _DEFAULT_LAMBDA1,
        alpha_init: float = _DEFAULT_ALPHA_INIT,
        alpha_mult: float = _DEFAULT_ALPHA_MULT,
        lr: float = _DEFAULT_LR,
        max_iter: int = _DEFAULT_MAX_ITER,
        h_tol: float = _DEFAULT_H_TOL,
        threshold: float = _DEFAULT_THRESHOLD,
    ) -> None:
        if n_channels < 2:
            raise ValueError("Need at least 2 channels for causal discovery.")

        self.n_channels = n_channels
        self.sf = sf

        if channel_names is not None:
            if len(channel_names) != n_channels:
                raise ValueError(
                    f"channel_names length ({len(channel_names)}) "
                    f"!= n_channels ({n_channels})"
                )
            self.channel_names = list(channel_names)
        else:
            self.channel_names = [f"ch{i}" for i in range(n_channels)]

        self.class_names = list(class_names) if class_names is not None else []
        self.max_lag = max(1, int(max_lag_sec * sf))
        self.lambda1 = lambda1
        self.alpha_init = alpha_init
        self.alpha_mult = alpha_mult
        self.lr = lr
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.threshold = threshold

        # Stores discovered adjacency matrices keyed by class name.
        # Each matrix has shape (n_channels, n_channels); entry W[i, j] > 0
        # means channel i causally influences channel j.
        self._adjacency: Dict[str, np.ndarray] = {}

        logger.info(
            "CausalChannelDiscovery: %d channels, sf=%d Hz, max_lag=%d samples, "
            "lambda1=%.4f, threshold=%.3f",
            n_channels, sf, self.max_lag, lambda1, threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover(
        self, epochs: np.ndarray, labels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Discover a causal adjacency matrix for each MI class.

        Args:
            epochs: Epoched EEG data, shape ``(n_trials, n_channels, n_samples)``.
            labels: Integer class labels, shape ``(n_trials,)``.

        Returns:
            Dictionary mapping class name to a weight matrix of shape
            ``(n_channels, n_channels)``. Entry ``W[i, j]`` is the
            estimated causal strength from channel *i* to channel *j*.
            Only positive (thresholded) values are retained.

        Raises:
            ValueError: If shapes are inconsistent or no valid classes found.
        """
        epochs = np.asarray(epochs, dtype=np.float64)
        labels = np.asarray(labels).ravel()

        if epochs.ndim != 3:
            raise ValueError(
                f"epochs must be 3-D (n_trials, n_channels, n_samples), "
                f"got shape {epochs.shape}"
            )
        if epochs.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Mismatch: {epochs.shape[0]} epochs vs {labels.shape[0]} labels."
            )
        if epochs.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {epochs.shape[1]}."
            )

        unique_labels = np.unique(labels)

        # Assign class names if not provided at init
        if not self.class_names:
            self.class_names = [f"class{int(c)}" for c in unique_labels]

        if len(self.class_names) < len(unique_labels):
            # Extend with generic names for any extra labels
            for c in unique_labels:
                cname = f"class{int(c)}"
                if cname not in self.class_names:
                    self.class_names.append(cname)

        self._adjacency.clear()

        for idx, label_val in enumerate(unique_labels):
            class_name = (
                self.class_names[idx]
                if idx < len(self.class_names)
                else f"class{int(label_val)}"
            )
            class_epochs = epochs[labels == label_val]
            n_trials = class_epochs.shape[0]

            if n_trials < _MIN_EPOCHS_PER_CLASS:
                logger.warning(
                    "Class '%s': only %d epochs (need >= %d). "
                    "Skipping causal discovery — returning zero matrix.",
                    class_name, n_trials, _MIN_EPOCHS_PER_CLASS,
                )
                self._adjacency[class_name] = np.zeros(
                    (self.n_channels, self.n_channels)
                )
                continue

            logger.info(
                "Discovering causal structure for class '%s' (%d epochs)...",
                class_name, n_trials,
            )

            W = self._discover_single_class(class_epochs)
            self._adjacency[class_name] = W

            n_edges = int(np.sum(W > 0))
            logger.info(
                "Class '%s': discovered %d causal edges.", class_name, n_edges
            )

        return dict(self._adjacency)

    def get_important_channels(
        self, class_name: str, top_k: int = 6
    ) -> List[int]:
        """Return the most causally important channels for a class.

        Importance is measured by total causal influence: the sum of
        outgoing (driving) and incoming (driven) edge weights.

        Args:
            class_name: Name of the MI class.
            top_k: Number of channels to return.

        Returns:
            List of channel indices sorted by descending importance.

        Raises:
            KeyError: If *class_name* has not been discovered yet.
        """
        W = self._get_adjacency(class_name)
        importance = W.sum(axis=1) + W.sum(axis=0)
        top_k = min(top_k, self.n_channels)
        return list(np.argsort(importance)[::-1][:top_k])

    def get_channel_pairs(
        self, class_name: str, top_k: int = 10
    ) -> List[Tuple[int, int]]:
        """Return the strongest causal channel pairs.

        Args:
            class_name: Name of the MI class.
            top_k: Maximum number of pairs to return.

        Returns:
            List of ``(source, target)`` tuples sorted by descending
            edge weight.
        """
        W = self._get_adjacency(class_name)
        # Flatten, sort, pick top-k nonzero
        flat = W.ravel()
        order = np.argsort(flat)[::-1]

        pairs: List[Tuple[int, int]] = []
        for idx in order:
            if flat[idx] <= 0:
                break
            src = int(idx // self.n_channels)
            tgt = int(idx % self.n_channels)
            pairs.append((src, tgt))
            if len(pairs) >= top_k:
                break

        return pairs

    def get_hub_channels(
        self, class_name: str, top_k: int = 3
    ) -> List[int]:
        """Return hub channels — those with the most outgoing causal edges.

        Hub channels are strong *drivers* of other channels during a
        particular MI class.

        Args:
            class_name: Name of the MI class.
            top_k: Number of hub channels to return.

        Returns:
            List of channel indices sorted by descending out-degree.
        """
        W = self._get_adjacency(class_name)
        out_degree = np.sum(W > 0, axis=1) + W.sum(axis=1)
        top_k = min(top_k, self.n_channels)
        return list(np.argsort(out_degree)[::-1][:top_k])

    def adjacency_to_networkx(self, class_name: str):
        """Convert the causal adjacency matrix to a NetworkX DiGraph.

        Args:
            class_name: Name of the MI class.

        Returns:
            ``networkx.DiGraph`` with channels as nodes and causal
            weights as edge attributes. Returns *None* if NetworkX is
            not installed.
        """
        try:
            import networkx as nx  # type: ignore
        except ImportError:
            logger.warning("networkx not installed — returning None.")
            return None

        W = self._get_adjacency(class_name)
        G = nx.DiGraph()

        for i in range(self.n_channels):
            G.add_node(i, name=self.channel_names[i])

        for i in range(self.n_channels):
            for j in range(self.n_channels):
                if W[i, j] > 0:
                    G.add_edge(i, j, weight=float(W[i, j]))

        return G

    @property
    def discovered_classes(self) -> List[str]:
        """Names of classes for which causal structure has been discovered."""
        return list(self._adjacency.keys())

    # ------------------------------------------------------------------
    # Core DAGMA-inspired optimisation
    # ------------------------------------------------------------------

    def _discover_single_class(self, class_epochs: np.ndarray) -> np.ndarray:
        """Run DAGMA-inspired causal discovery on epochs of one class.

        Args:
            class_epochs: shape ``(n_trials, n_channels, n_samples)``.

        Returns:
            Thresholded adjacency matrix, shape ``(n_channels, n_channels)``.
        """
        d = self.n_channels

        # Step 1: Build a design matrix from time-lagged cross-correlations
        X = self._build_design_matrix(class_epochs)

        # Step 2: Standardize columns for numerical stability
        col_std = X.std(axis=0)
        col_std[col_std < 1e-12] = 1.0
        X = X / col_std

        # Step 3: Initialise W near zero
        rng = np.random.RandomState(42)
        W = rng.randn(d, d) * 0.01
        np.fill_diagonal(W, 0.0)

        # Step 4: Augmented-Lagrangian-style loop
        alpha = self.alpha_init
        n_outer = 20  # max outer iterations

        for outer in range(n_outer):
            W = self._gradient_descent(X, W, alpha)
            h_val = self._h_acyclicity(W)

            if h_val < self.h_tol:
                logger.debug(
                    "Acyclicity converged at outer iter %d (h=%.2e).",
                    outer, h_val,
                )
                break

            alpha *= self.alpha_mult

        # Step 5: Threshold and ensure non-negative
        W_out = np.abs(W)
        np.fill_diagonal(W_out, 0.0)
        W_out[W_out < self.threshold] = 0.0

        return W_out

    def _gradient_descent(
        self, X: np.ndarray, W: np.ndarray, alpha: float
    ) -> np.ndarray:
        """Run gradient descent on the DAGMA objective.

        Minimises::

            L(W) = 0.5 * ||X - X W||_F^2 / n  +  lambda1 * |W|_1  +  alpha * h(W)

        Args:
            X: Design matrix, shape ``(n_obs, d)``.
            W: Current weight matrix, shape ``(d, d)``.
            alpha: Acyclicity penalty weight.

        Returns:
            Updated weight matrix.
        """
        d = W.shape[0]
        n = X.shape[0]
        W = W.copy()
        lr = self.lr

        for step in range(self.max_iter):
            # Clip W to prevent overflow in matmul and matrix exponential
            W = np.clip(W, -5.0, 5.0)

            # --- Reconstruction loss gradient ---
            residual = X - X @ W  # (n, d)
            grad_loss = -(X.T @ residual) / n  # (d, d)

            # --- L1 sub-gradient ---
            grad_l1 = self.lambda1 * np.sign(W)

            # --- Acyclicity gradient ---
            # h(W) = tr(expm(W o W)) - d
            # grad_h = 2 * W o expm(W o W)^T
            W_sq = W * W
            E = expm(W_sq)
            # Guard against NaN/inf from numerical overflow
            if not np.all(np.isfinite(E)):
                break
            grad_h = 2.0 * W * E.T

            # --- Combined gradient ---
            grad = grad_loss + grad_l1 + alpha * grad_h

            # Zero out diagonal (no self-loops)
            np.fill_diagonal(grad, 0.0)

            # Update
            W -= lr * grad
            np.fill_diagonal(W, 0.0)

            # Early stopping if gradient is small
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-6:
                break

        return W

    # ------------------------------------------------------------------
    # Acyclicity constraint
    # ------------------------------------------------------------------

    @staticmethod
    def _h_acyclicity(W: np.ndarray) -> float:
        """Compute the DAGMA acyclicity penalty h(W).

        .. math::

            h(W) = \\text{tr}(e^{W \\circ W}) - d

        When h(W) = 0 the graph encoded by W is a DAG.

        Args:
            W: Weight matrix, shape ``(d, d)``.

        Returns:
            Non-negative scalar. Zero iff W encodes a DAG.
        """
        d = W.shape[0]
        W_clipped = np.clip(W, -5.0, 5.0)
        W_sq = W_clipped * W_clipped
        try:
            E = expm(W_sq)
            if not np.all(np.isfinite(E)):
                return float(d * 100.0)  # large penalty to push W down
            h = np.trace(E) - d
        except (np.linalg.LinAlgError, ValueError):
            # Fallback for singular / ill-conditioned matrices:
            # Use the power-series approximation (first 10 terms)
            h = 0.0
            M = np.eye(d)
            for k in range(1, 11):
                M = M @ W_sq / k
                h += np.trace(M)
        return float(max(h, 0.0))

    # ------------------------------------------------------------------
    # Design matrix construction
    # ------------------------------------------------------------------

    def _build_design_matrix(self, class_epochs: np.ndarray) -> np.ndarray:
        """Build a design matrix from time-lagged cross-correlations.

        For each pair of channels (i, j), the maximum absolute cross-
        correlation across lags ``1..max_lag`` is computed per trial.
        The resulting feature vectors (one per trial per time-window)
        are stacked into a matrix suitable for the linear SEM
        ``X = X W + noise``.

        Args:
            class_epochs: shape ``(n_trials, n_channels, n_samples)``.

        Returns:
            Design matrix of shape ``(n_observations, n_channels)``.
        """
        n_trials, d, n_samples = class_epochs.shape

        # We segment each trial into overlapping windows to get more
        # observations for the regression.
        win_len = max(self.max_lag * 4, int(0.5 * self.sf))
        win_step = max(1, win_len // 2)
        n_windows = max(1, (n_samples - win_len) // win_step + 1)

        observations = []
        for t in range(n_trials):
            for w in range(n_windows):
                start = w * win_step
                end = start + win_len
                if end > n_samples:
                    break

                segment = class_epochs[t, :, start:end]  # (d, win_len)

                # Feature: per-channel power in this window (log-scale)
                power = np.log1p(np.mean(segment ** 2, axis=1))  # (d,)
                observations.append(power)

        if not observations:
            # Fallback: use trial-level mean power
            for t in range(n_trials):
                power = np.log1p(np.mean(class_epochs[t] ** 2, axis=1))
                observations.append(power)

        X = np.stack(observations, axis=0)  # (n_obs, d)

        # Safety check for degenerate data
        if X.shape[0] < d:
            logger.warning(
                "Fewer observations (%d) than channels (%d). "
                "Results may be unreliable.",
                X.shape[0], d,
            )

        return X

    # ------------------------------------------------------------------
    # Time-lagged cross-correlation (used for initial warm-start)
    # ------------------------------------------------------------------

    def _time_lagged_xcorr(
        self, signal_a: np.ndarray, signal_b: np.ndarray
    ) -> float:
        """Maximum absolute cross-correlation from *a* to *b* over positive lags.

        A positive lag means *a* leads *b*, consistent with the
        interpretation that *a* causally influences *b*.

        Args:
            signal_a: 1-D source signal.
            signal_b: 1-D target signal.

        Returns:
            Maximum absolute normalised cross-correlation (0 to 1).
        """
        a = signal_a - signal_a.mean()
        b = signal_b - signal_b.mean()

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0

        a = a / norm_a
        b = b / norm_b

        max_xcorr = 0.0
        n = len(a)
        for lag in range(1, self.max_lag + 1):
            if lag >= n:
                break
            corr = np.abs(np.dot(a[:n - lag], b[lag:]))
            if corr > max_xcorr:
                max_xcorr = corr

        return float(max_xcorr)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_adjacency(self, class_name: str) -> np.ndarray:
        """Retrieve the adjacency matrix for a class, raising on miss."""
        if class_name not in self._adjacency:
            raise KeyError(
                f"No causal structure discovered for class '{class_name}'. "
                f"Available: {list(self._adjacency.keys())}. "
                f"Call discover() first."
            )
        return self._adjacency[class_name]

    def summary(self, class_name: str) -> str:
        """Return a human-readable summary of the causal structure.

        Args:
            class_name: Name of the MI class.

        Returns:
            Multi-line string describing top edges and hub channels.
        """
        W = self._get_adjacency(class_name)
        n_edges = int(np.sum(W > 0))
        hubs = self.get_hub_channels(class_name, top_k=3)
        top_pairs = self.get_channel_pairs(class_name, top_k=5)

        lines = [
            f"Causal structure for class '{class_name}':",
            f"  Channels: {self.n_channels}",
            f"  Edges:    {n_edges}",
            f"  Density:  {n_edges / max(1, self.n_channels * (self.n_channels - 1)):.1%}",
            f"  Hub channels (top drivers):",
        ]
        for ch in hubs:
            out_w = W[ch].sum()
            lines.append(f"    {self.channel_names[ch]} (out-weight={out_w:.3f})")

        lines.append("  Strongest causal pairs:")
        for src, tgt in top_pairs:
            lines.append(
                f"    {self.channel_names[src]} -> "
                f"{self.channel_names[tgt]} (w={W[src, tgt]:.3f})"
            )

        return "\n".join(lines)

    def __repr__(self) -> str:
        discovered = ", ".join(self._adjacency.keys()) or "none"
        return (
            f"CausalChannelDiscovery(n_channels={self.n_channels}, "
            f"sf={self.sf}, discovered=[{discovered}])"
        )
