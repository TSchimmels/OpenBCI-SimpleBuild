"""MODE-inspired Adaptive Classifier Router for motor imagery BCI.

Dynamically routes each EEG classification window to the best-suited
classifier based on real-time signal characteristics extracted from the
window itself.  The routing mechanism is inspired by the Mixture of
Experts (MoE) gating framework, where a lightweight gating network
selects which expert (classifier) processes each input.

Available experts:
    * **CSP+LDA** -- fast, reliable with clean high-SNR data
    * **EEGNet** -- deep learning, captures complex nonlinear patterns
    * **Riemannian MDM** -- geometry-aware, robust to noise and
      non-stationarity

The router computes five signal-quality features per window:

    1. **SNR estimate** -- signal RMS vs. noise-floor RMS after bandpass
    2. **Artifact density** -- fraction of samples exceeding an amplitude
       threshold (likely EMG/EOG contamination)
    3. **Mu power ratio** -- power in the 8--12 Hz mu band relative to
       total power (motor imagery discriminability proxy)
    4. **Stationarity index** -- variance of rolling variance (high
       values indicate non-stationary segments)
    5. **Channel correlation** -- mean pairwise Pearson correlation
       (high values suggest volume-conducted artifact)

Routing policy:
    * High SNR + low artifacts + strong mu  -->  CSP+LDA
    * Low SNR + high artifacts              -->  Riemannian MDM
    * Ambiguous / complex pattern           -->  EEGNet
    * Below confidence threshold            -->  weighted ensemble

References:
    Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q.,
    Hinton, G., & Dean, J. (2017). Outrageously Large Neural Networks:
    The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.

    Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E.
    (1991). Adaptive Mixtures of Local Experts. Neural Computation,
    3(1), 79-87.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch

from .base import BaseClassifier

logger = logging.getLogger(__name__)

# Classifier name constants used as dictionary keys throughout
CSPLDA = "csp_lda"
EEGNET = "eegnet"
RIEMANNIAN = "riemannian"
ENSEMBLE = "ensemble"

# Default routing thresholds (can be overridden via config)
_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "snr_high": 5.0,           # dB -- above this, signal is clean
    "snr_low": 1.5,            # dB -- below this, signal is noisy
    "artifact_low": 0.05,      # fraction -- below 5 % is clean
    "artifact_high": 0.20,     # fraction -- above 20 % is contaminated
    "mu_power_strong": 0.15,   # ratio -- strong mu presence
    "stationarity_high": 2.0,  # z-score -- above this is non-stationary
    "correlation_high": 0.70,  # mean r -- above this is volume conduction
    "confidence_min": 0.55,    # min predicted proba for single-expert trust
}

# Default signal parameters
_DEFAULT_SIGNAL_PARAMS: Dict[str, Any] = {
    "sampling_rate": 125,           # Hz (Cyton+Daisy default)
    "artifact_threshold_uv": 100,   # micro-volts peak threshold
    "mu_band": (8.0, 12.0),        # Hz -- mu rhythm
    "signal_band": (1.0, 40.0),    # Hz -- broadband EEG
    "rolling_var_window": 25,       # samples for stationarity check
}


# =====================================================================
# Signal quality feature extraction
# =====================================================================

def _compute_snr(epoch: np.ndarray, fs: int,
                 signal_band: Tuple[float, float],
                 broadband: Tuple[float, float] = (1.0, 40.0)) -> float:
    """Estimate signal-to-noise ratio in dB.

    SNR is computed as the ratio of mean power in the signal band
    (typically 8-30 Hz for motor imagery) to power outside that band
    but within the broadband range.

    Args:
        epoch: EEG data, shape ``(n_channels, n_samples)``.
        fs: Sampling frequency in Hz.
        signal_band: ``(low_hz, high_hz)`` for the signal of interest.
        broadband: ``(low_hz, high_hz)`` total frequency range.

    Returns:
        SNR estimate in dB.  Returns 0.0 on computation failure.
    """
    try:
        n_channels = epoch.shape[0]
        snr_per_channel = np.zeros(n_channels)

        for ch in range(n_channels):
            freqs, psd = welch(epoch[ch], fs=fs, nperseg=min(fs, epoch.shape[1]))

            # Power in signal band
            sig_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
            sig_power = np.mean(psd[sig_mask]) if np.any(sig_mask) else 1e-12

            # Power outside signal band (noise floor)
            broad_mask = (freqs >= broadband[0]) & (freqs <= broadband[1])
            noise_mask = broad_mask & ~sig_mask
            noise_power = np.mean(psd[noise_mask]) if np.any(noise_mask) else 1e-12

            snr_per_channel[ch] = sig_power / max(noise_power, 1e-12)

        mean_snr = np.mean(snr_per_channel)
        return float(10.0 * np.log10(max(mean_snr, 1e-12)))
    except Exception:
        logger.debug("SNR computation failed, returning 0.0 dB")
        return 0.0


def _compute_artifact_density(epoch: np.ndarray,
                              threshold_uv: float) -> float:
    """Compute fraction of samples exceeding an amplitude threshold.

    High artifact density indicates EMG, EOG, or electrode-pop
    contamination that degrades linear classifiers.

    Args:
        epoch: EEG data, shape ``(n_channels, n_samples)``.
        threshold_uv: Amplitude threshold in micro-volts.

    Returns:
        Fraction in ``[0, 1]`` of total samples above threshold.
    """
    total_samples = epoch.shape[0] * epoch.shape[1]
    if total_samples == 0:
        return 0.0
    artifact_count = int(np.sum(np.abs(epoch) > threshold_uv))
    return artifact_count / total_samples


def _compute_mu_power_ratio(epoch: np.ndarray, fs: int,
                            mu_band: Tuple[float, float]) -> float:
    """Compute the ratio of mu-band power to total power.

    A strong mu rhythm indicates clear sensorimotor engagement, which
    is exactly what CSP is designed to exploit.

    Args:
        epoch: EEG data, shape ``(n_channels, n_samples)``.
        fs: Sampling frequency in Hz.
        mu_band: ``(low_hz, high_hz)`` of the mu rhythm.

    Returns:
        Power ratio in ``[0, 1]``.  Returns 0.0 on failure.
    """
    try:
        ratios = np.zeros(epoch.shape[0])
        for ch in range(epoch.shape[0]):
            freqs, psd = welch(epoch[ch], fs=fs, nperseg=min(fs, epoch.shape[1]))
            mu_mask = (freqs >= mu_band[0]) & (freqs <= mu_band[1])
            mu_power = np.sum(psd[mu_mask])
            total_power = np.sum(psd)
            ratios[ch] = mu_power / max(total_power, 1e-12)
        return float(np.mean(ratios))
    except Exception:
        logger.debug("Mu power ratio computation failed, returning 0.0")
        return 0.0


def _compute_stationarity_index(epoch: np.ndarray,
                                window: int) -> float:
    """Compute a stationarity index from rolling variance.

    Non-stationary segments have high variance-of-variance.  The index
    is the coefficient of variation (std / mean) of the rolling window
    variance, averaged across channels.

    Args:
        epoch: EEG data, shape ``(n_channels, n_samples)``.
        window: Rolling window size in samples.

    Returns:
        Stationarity index (higher = less stationary).
    """
    n_channels, n_samples = epoch.shape
    if n_samples < window * 2:
        return 0.0

    indices = np.zeros(n_channels)
    n_windows = n_samples - window + 1
    for ch in range(n_channels):
        # Efficient rolling variance via cumulative sums
        x = epoch[ch]
        cumsum = np.cumsum(x)
        cumsum2 = np.cumsum(x ** 2)

        # Windowed mean and variance
        window_sum = cumsum[window - 1:] - np.concatenate(([0.0], cumsum[:n_windows - 1]))
        window_sum2 = cumsum2[window - 1:] - np.concatenate(([0.0], cumsum2[:n_windows - 1]))
        window_mean = window_sum / window
        window_var = window_sum2 / window - window_mean ** 2
        window_var = np.maximum(window_var, 0.0)  # numerical floor

        var_mean = np.mean(window_var)
        var_std = np.std(window_var)
        indices[ch] = var_std / max(var_mean, 1e-12)

    return float(np.mean(indices))


def _compute_channel_correlation(epoch: np.ndarray) -> float:
    """Compute mean pairwise Pearson correlation across channels.

    High correlation indicates volume conduction or a common artifact
    source, which violates the independence assumptions that CSP and
    EEGNet rely on.  Riemannian methods handle this more gracefully
    through the covariance manifold representation.

    Args:
        epoch: EEG data, shape ``(n_channels, n_samples)``.

    Returns:
        Mean absolute pairwise correlation in ``[0, 1]``.
    """
    n_channels = epoch.shape[0]
    if n_channels < 2:
        return 0.0
    try:
        corr_matrix = np.corrcoef(epoch)
        # Extract upper triangle (excluding diagonal)
        upper_indices = np.triu_indices(n_channels, k=1)
        mean_corr = float(np.mean(np.abs(corr_matrix[upper_indices])))
        return mean_corr
    except Exception:
        return 0.0


def extract_signal_features(epoch: np.ndarray,
                            params: Dict[str, Any]) -> Dict[str, float]:
    """Extract all five signal quality features from a single epoch.

    This is the feature vector that the gating network uses to decide
    which expert classifier should handle the current window.

    Args:
        epoch: EEG data, shape ``(n_channels, n_samples)``.
        params: Signal parameters (sampling_rate, thresholds, bands).

    Returns:
        Dictionary with keys: ``snr_db``, ``artifact_density``,
        ``mu_power_ratio``, ``stationarity_index``,
        ``channel_correlation``.
    """
    fs = params.get("sampling_rate", 250)
    mu_band = params.get("mu_band", (8.0, 12.0))
    signal_band = params.get("signal_band", (1.0, 40.0))
    artifact_thresh = params.get("artifact_threshold_uv", 100.0)
    rolling_window = params.get("rolling_var_window", 25)

    return {
        "snr_db": _compute_snr(epoch, fs, signal_band=mu_band,
                                broadband=signal_band),
        "artifact_density": _compute_artifact_density(epoch, artifact_thresh),
        "mu_power_ratio": _compute_mu_power_ratio(epoch, fs, mu_band),
        "stationarity_index": _compute_stationarity_index(epoch, rolling_window),
        "channel_correlation": _compute_channel_correlation(epoch),
    }


# =====================================================================
# Routing logic
# =====================================================================

def _select_expert(features: Dict[str, float],
                   thresholds: Dict[str, float]) -> str:
    """Deterministic rule-based expert selection.

    Implements a decision tree inspired by the sparse gating mechanism
    in Shazeer et al. (2017), but using interpretable signal-quality
    rules instead of a learned gating network.  This is preferred for
    BCI applications where explainability and safety matter.

    The decision priority is:

    1. If signal is clean (high SNR, low artifacts, strong mu) then
       CSP+LDA is the fastest and most interpretable choice.
    2. If signal is noisy (low SNR, high artifacts, or non-stationary)
       then Riemannian MDM is the most robust.
    3. Otherwise, EEGNet can learn nonlinear discriminative features
       that the other two miss.

    Args:
        features: Signal quality features from
            :func:`extract_signal_features`.
        thresholds: Routing thresholds.

    Returns:
        Classifier key: one of ``'csp_lda'``, ``'eegnet'``, or
        ``'riemannian'``.
    """
    snr = features["snr_db"]
    artifact = features["artifact_density"]
    mu = features["mu_power_ratio"]
    stationarity = features["stationarity_index"]
    correlation = features["channel_correlation"]

    # Rule 1: Clean signal with strong motor imagery features --> CSP+LDA
    if (snr >= thresholds["snr_high"]
            and artifact <= thresholds["artifact_low"]
            and mu >= thresholds["mu_power_strong"]
            and stationarity <= thresholds["stationarity_high"]):
        return CSPLDA

    # Rule 2: Noisy / non-stationary signal --> Riemannian MDM
    if (snr <= thresholds["snr_low"]
            or artifact >= thresholds["artifact_high"]
            or stationarity > thresholds["stationarity_high"]
            or correlation >= thresholds["correlation_high"]):
        return RIEMANNIAN

    # Rule 3: Ambiguous -- let the deep network handle it
    return EEGNET


# =====================================================================
# Adaptive Classifier Router
# =====================================================================

class AdaptiveClassifierRouter(BaseClassifier):
    """MODE-inspired adaptive router that selects the best classifier
    per EEG window based on real-time signal characteristics.

    The router wraps three specialist classifiers and acts as a
    transparent drop-in replacement that implements the full
    :class:`BaseClassifier` interface.  At inference time, each trial
    is independently routed to the most suitable expert based on
    signal quality features extracted from the EEG data itself.

    When no single expert exceeds the minimum confidence threshold,
    the router falls back to a weighted ensemble that blends the
    probability outputs of all three experts, weighted by their
    predicted confidence (maximum class probability).  This follows
    the adaptive mixture formulation from Jacobs et al. (1991).

    Args:
        classifiers: Dictionary mapping classifier names to instances.
            Expected keys: ``'csp_lda'``, ``'eegnet'``, ``'riemannian'``.
        config: Configuration dictionary with optional sub-keys:

            * ``'thresholds'`` -- routing thresholds (see
              ``_DEFAULT_THRESHOLDS``)
            * ``'signal_params'`` -- signal processing parameters (see
              ``_DEFAULT_SIGNAL_PARAMS``)
            * ``'enable_gating_network'`` -- if ``True``, enable
              online learning of the routing policy (default ``False``)
    """

    def __init__(
        self,
        classifiers: Dict[str, BaseClassifier],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        config = config or {}

        # --- Validate classifiers ---
        self._classifiers: Dict[str, BaseClassifier] = {}
        for key in (CSPLDA, EEGNET, RIEMANNIAN):
            if key not in classifiers:
                raise ValueError(
                    f"Missing required classifier '{key}'. Provide all "
                    f"three: {CSPLDA}, {EEGNET}, {RIEMANNIAN}."
                )
            self._classifiers[key] = classifiers[key]

        # --- Routing configuration ---
        self._thresholds: Dict[str, float] = {
            **_DEFAULT_THRESHOLDS,
            **config.get("thresholds", {}),
        }
        self._signal_params: Dict[str, Any] = {
            **_DEFAULT_SIGNAL_PARAMS,
            **config.get("signal_params", {}),
        }
        self._confidence_min: float = self._thresholds["confidence_min"]

        # --- Routing statistics ---
        self._routing_counts: Dict[str, int] = defaultdict(int)
        self._routing_latencies: Dict[str, List[float]] = defaultdict(list)
        self._total_predictions: int = 0
        self._fitted: bool = False

        # --- Optional gating network for online adaptation ---
        self._enable_gating: bool = config.get("enable_gating_network", False)
        self._gating_network: Optional[Any] = None
        if self._enable_gating:
            self._init_gating_network()

        logger.info(
            "AdaptiveClassifierRouter initialised with experts: %s | "
            "gating_network=%s",
            list(self._classifiers.keys()),
            self._enable_gating,
        )

    # ------------------------------------------------------------------
    # Private: gating network (optional online adaptation)
    # ------------------------------------------------------------------

    def _init_gating_network(self) -> None:
        """Initialise a small MLP gating network for learned routing.

        The gating network maps the 5-D signal feature vector to a
        softmax distribution over the 3 experts, following Shazeer et
        al. (2017) Eq. 3:

            G(x) = Softmax( W_g * x + noise )

        where noise provides load-balancing exploration.  The network
        is updated online using cross-entropy between the gating
        decision and the expert that actually achieved the highest
        classification confidence.

        This is optional and disabled by default because the
        deterministic rule-based routing is more interpretable and
        sufficient for typical motor imagery BCI use.
        """
        try:
            import torch
            import torch.nn as nn

            class GatingMLP(nn.Module):
                """Lightweight gating network (5 features -> 3 experts)."""

                def __init__(self) -> None:
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(5, 16),
                        nn.ReLU(),
                        nn.Linear(16, 3),
                        # No Softmax here — F.cross_entropy expects raw logits
                    )

                def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                    return self.net(x)

            self._gating_network = GatingMLP()
            self._gating_optimizer = torch.optim.Adam(
                self._gating_network.parameters(), lr=1e-3,
            )
            logger.info("Gating network initialised (MLP: 5->16->3)")
        except ImportError:
            logger.warning(
                "PyTorch not available -- gating network disabled. "
                "Falling back to rule-based routing."
            )
            self._enable_gating = False
            self._gating_network = None

    def _gating_predict(self, features: Dict[str, float]) -> str:
        """Use the learned gating network to select an expert.

        Args:
            features: Signal quality features.

        Returns:
            Classifier key selected by the gating network.
        """
        import torch

        expert_names = [CSPLDA, EEGNET, RIEMANNIAN]
        feature_vec = torch.tensor([
            features["snr_db"],
            features["artifact_density"],
            features["mu_power_ratio"],
            features["stationarity_index"],
            features["channel_correlation"],
        ], dtype=torch.float32).unsqueeze(0)

        self._gating_network.eval()
        with torch.no_grad():
            weights = self._gating_network(feature_vec)
        idx = int(torch.argmax(weights, dim=1).item())
        return expert_names[idx]

    def _gating_update(self, features: Dict[str, float],
                       best_expert_idx: int) -> None:
        """Online update of the gating network toward the best expert.

        After all experts produce predictions, the one with the
        highest confidence (max predicted probability) is treated as
        the ground-truth routing target.  The gating network is
        updated via one step of cross-entropy loss.

        Args:
            features: Signal quality features for the current window.
            best_expert_idx: Index of the expert with highest confidence
                (0=csp_lda, 1=eegnet, 2=riemannian).
        """
        import torch
        import torch.nn.functional as F

        feature_vec = torch.tensor([
            features["snr_db"],
            features["artifact_density"],
            features["mu_power_ratio"],
            features["stationarity_index"],
            features["channel_correlation"],
        ], dtype=torch.float32).unsqueeze(0)
        target = torch.tensor([best_expert_idx], dtype=torch.long)

        self._gating_network.train()
        self._gating_optimizer.zero_grad()
        pred = self._gating_network(feature_vec)
        ce_loss = F.cross_entropy(pred, target)
        # Router z-loss (ST-MoE, Zoph et al. 2022): prevents logit explosion
        z_loss = torch.logsumexp(pred, dim=-1).square().mean()
        loss = ce_loss + 1e-3 * z_loss
        loss.backward()
        self._gating_optimizer.step()

    # ------------------------------------------------------------------
    # Private: core routing
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_3d(X: np.ndarray) -> np.ndarray:
        """Ensure input is 3-D ``(n_trials, n_channels, n_samples)``."""
        if X.ndim == 2:
            return X[np.newaxis, :, :]
        if X.ndim == 3:
            return X
        raise ValueError(
            f"Expected 2-D or 3-D input, got {X.ndim}-D with shape {X.shape}"
        )

    def _route_single(self, epoch: np.ndarray) -> str:
        """Route a single epoch to the best expert.

        Args:
            epoch: Single EEG window, shape ``(n_channels, n_samples)``.

        Returns:
            Classifier key for the selected expert.
        """
        features = extract_signal_features(epoch, self._signal_params)

        if self._enable_gating and self._gating_network is not None:
            return self._gating_predict(features)

        return _select_expert(features, self._thresholds)

    def _ensemble_predict_proba(self, epoch_3d: np.ndarray) -> np.ndarray:
        """Weighted ensemble prediction using confidence-based weights.

        Each expert classifies the trial independently.  The final
        probability is a convex combination weighted by each expert's
        maximum predicted probability (confidence), following the
        adaptive mixture formulation from Jacobs et al. (1991):

            p(y|x) = sum_i  g_i(x) * p_i(y|x)

        where g_i is the normalised confidence weight for expert i.

        Args:
            epoch_3d: Single trial, shape ``(1, n_channels, n_samples)``.

        Returns:
            Blended probability array, shape ``(1, n_classes)``.
        """
        probas = {}
        confidences = {}
        for name, clf in self._classifiers.items():
            try:
                proba = clf.predict_proba(epoch_3d)
                probas[name] = proba
                confidences[name] = float(np.max(proba))
            except Exception as exc:
                logger.debug("Ensemble: %s failed (%s), skipping", name, exc)
                continue

        if not probas:
            raise RuntimeError(
                "All classifiers failed during ensemble prediction."
            )

        # Normalise confidence weights to sum to 1
        total_conf = sum(confidences.values())
        if total_conf < 1e-12:
            # Equal weighting as ultimate fallback
            weights = {name: 1.0 / len(probas) for name in probas}
        else:
            weights = {name: conf / total_conf
                       for name, conf in confidences.items()}

        # Weighted combination
        result = np.zeros_like(next(iter(probas.values())))
        for name, proba in probas.items():
            result += weights[name] * proba

        return result

    # ------------------------------------------------------------------
    # BaseClassifier interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdaptiveClassifierRouter":
        """Fit all expert classifiers on the training data.

        Each expert receives the full training set.  In a future
        version, the gating network could be trained here via
        cross-validation to learn which expert works best for which
        signal regime.

        Args:
            X: Training epochs, shape ``(n_trials, n_channels,
                n_samples)``.
            y: Integer class labels, shape ``(n_trials,)``.

        Returns:
            ``self``
        """
        X = self._ensure_3d(X)
        n_trials = X.shape[0]
        logger.info(
            "Fitting %d expert classifiers on %d trials...",
            len(self._classifiers), n_trials,
        )

        for name, clf in self._classifiers.items():
            t0 = time.perf_counter()
            try:
                clf.fit(X, y)
                elapsed = time.perf_counter() - t0
                logger.info(
                    "  [%s] fitted in %.2f s", name, elapsed,
                )
            except Exception as exc:
                logger.error(
                    "  [%s] fitting FAILED: %s", name, exc,
                )
                raise

        # Reset routing statistics
        self._routing_counts = defaultdict(int)
        self._routing_latencies = defaultdict(list)
        self._total_predictions = 0
        self._fitted = True

        logger.info("AdaptiveClassifierRouter fit complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Route each trial to the best expert and predict labels.

        Args:
            X: Input epochs, shape ``(n_trials, n_channels, n_samples)``
                or ``(n_channels, n_samples)`` for a single trial.

        Returns:
            Predicted integer labels, shape ``(n_trials,)``.
        """
        if not self._fitted:
            raise RuntimeError(
                "Router has not been fitted. Call fit() first."
            )
        X = self._ensure_3d(X)
        predictions = np.zeros(X.shape[0], dtype=int)

        for i in range(X.shape[0]):
            epoch = X[i]  # (n_channels, n_samples)
            epoch_3d = epoch[np.newaxis, :, :]  # (1, C, T)

            t0 = time.perf_counter()
            expert_name = self._route_single(epoch)
            clf = self._classifiers[expert_name]

            try:
                proba = clf.predict_proba(epoch_3d)
                confidence = float(np.max(proba))

                if confidence < self._confidence_min:
                    # Confidence too low -- fall back to ensemble
                    proba = self._ensemble_predict_proba(epoch_3d)
                    expert_name = ENSEMBLE

                predictions[i] = int(np.argmax(proba, axis=1)[0])
            except Exception as exc:
                logger.warning(
                    "Expert '%s' failed on trial %d (%s), "
                    "falling back to ensemble.",
                    expert_name, i, exc,
                )
                proba = self._ensemble_predict_proba(epoch_3d)
                expert_name = ENSEMBLE
                predictions[i] = int(np.argmax(proba, axis=1)[0])

            elapsed = time.perf_counter() - t0
            self._routing_counts[expert_name] += 1
            self._routing_latencies[expert_name].append(elapsed)
            self._total_predictions += 1

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Route each trial and return class probabilities.

        Args:
            X: Input epochs, shape ``(n_trials, n_channels, n_samples)``
                or ``(n_channels, n_samples)`` for a single trial.

        Returns:
            Probabilities, shape ``(n_trials, n_classes)``.
        """
        if not self._fitted:
            raise RuntimeError(
                "Router has not been fitted. Call fit() first."
            )
        X = self._ensure_3d(X)
        all_proba: List[np.ndarray] = []

        for i in range(X.shape[0]):
            epoch = X[i]
            epoch_3d = epoch[np.newaxis, :, :]

            expert_name = self._route_single(epoch)
            clf = self._classifiers[expert_name]

            try:
                proba = clf.predict_proba(epoch_3d)
                confidence = float(np.max(proba))

                if confidence < self._confidence_min:
                    proba = self._ensemble_predict_proba(epoch_3d)
                    expert_name = ENSEMBLE
            except Exception:
                proba = self._ensemble_predict_proba(epoch_3d)
                expert_name = ENSEMBLE

            all_proba.append(proba)
            self._routing_counts[expert_name] += 1
            self._total_predictions += 1

        return np.vstack(all_proba)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Route each trial and return continuous decision scores.

        Args:
            X: Input epochs.

        Returns:
            Decision scores from the routed expert.
        """
        if not self._fitted:
            raise RuntimeError(
                "Router has not been fitted. Call fit() first."
            )
        X = self._ensure_3d(X)
        all_scores: List[np.ndarray] = []

        for i in range(X.shape[0]):
            epoch = X[i]
            epoch_3d = epoch[np.newaxis, :, :]

            expert_name = self._route_single(epoch)
            clf = self._classifiers[expert_name]

            try:
                scores = clf.decision_function(epoch_3d)
            except Exception:
                # Fallback: use log-probabilities from ensemble
                proba = self._ensemble_predict_proba(epoch_3d)
                scores = np.log(proba + 1e-10)

            all_scores.append(scores)

        # Stack and handle variable-shape scores (binary vs multi-class)
        if all_scores[0].ndim > 1:
            return np.vstack(all_scores)
        return np.concatenate(all_scores)

    def predict_all(
        self, X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Route each trial, returning predictions, probabilities, and
        decision scores in a single pass.

        This avoids redundant routing and feature extraction compared
        to calling ``predict()``, ``predict_proba()``, and
        ``decision_function()`` separately.

        Args:
            X: Input epochs, shape ``(n_trials, n_channels, n_samples)``
                or ``(n_channels, n_samples)`` for a single trial.

        Returns:
            3-tuple ``(predictions, probabilities, decision_scores)``.
        """
        if not self._fitted:
            raise RuntimeError(
                "Router has not been fitted. Call fit() first."
            )
        X = self._ensure_3d(X)

        predictions = np.zeros(X.shape[0], dtype=int)
        all_proba: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []

        for i in range(X.shape[0]):
            epoch = X[i]
            epoch_3d = epoch[np.newaxis, :, :]

            t0 = time.perf_counter()
            features = extract_signal_features(epoch, self._signal_params)

            if self._enable_gating and self._gating_network is not None:
                expert_name = self._gating_predict(features)
            else:
                expert_name = _select_expert(features, self._thresholds)

            clf = self._classifiers[expert_name]

            try:
                pred, proba, scores = clf.predict_all(epoch_3d)
                confidence = float(np.max(proba))

                if confidence < self._confidence_min:
                    proba = self._ensemble_predict_proba(epoch_3d)
                    pred = np.argmax(proba, axis=1)
                    # Recompute scores from ensemble proba
                    scores = np.log(proba + 1e-10)
                    expert_name = ENSEMBLE
            except Exception as exc:
                logger.warning(
                    "Expert '%s' failed on trial %d (%s), using ensemble.",
                    expert_name, i, exc,
                )
                proba = self._ensemble_predict_proba(epoch_3d)
                pred = np.argmax(proba, axis=1)
                scores = np.log(proba + 1e-10)
                expert_name = ENSEMBLE

            predictions[i] = int(pred[0]) if pred.ndim > 0 else int(pred)
            all_proba.append(proba)
            all_scores.append(scores)

            elapsed = time.perf_counter() - t0
            self._routing_counts[expert_name] += 1
            self._routing_latencies[expert_name].append(elapsed)
            self._total_predictions += 1

            # Online gating update: train toward whichever expert was
            # most confident (only when gating network is enabled)
            if self._enable_gating and self._gating_network is not None:
                expert_confs = []
                expert_names = [CSPLDA, EEGNET, RIEMANNIAN]
                for name in expert_names:
                    try:
                        p = self._classifiers[name].predict_proba(epoch_3d)
                        expert_confs.append(float(np.max(p)))
                    except Exception:
                        expert_confs.append(0.0)
                best_idx = int(np.argmax(expert_confs))
                self._gating_update(features, best_idx)

        probabilities = np.vstack(all_proba)
        if all_scores[0].ndim > 1:
            decision_scores = np.vstack(all_scores)
        else:
            decision_scores = np.concatenate(all_scores)

        return predictions, probabilities, decision_scores

    # ------------------------------------------------------------------
    # Routing statistics
    # ------------------------------------------------------------------

    def get_routing_stats(self) -> Dict[str, Any]:
        """Return routing statistics collected during inference.

        Returns:
            Dictionary with:

            - ``'total_predictions'`` -- total trials processed
            - ``'routing_counts'`` -- per-expert selection counts
            - ``'routing_fractions'`` -- per-expert selection fractions
            - ``'mean_latencies_ms'`` -- per-expert mean latency in ms
            - ``'experts_available'`` -- list of expert names
        """
        total = max(self._total_predictions, 1)
        fractions = {
            name: count / total
            for name, count in self._routing_counts.items()
        }
        mean_latencies = {
            name: float(np.mean(lats)) * 1000.0 if lats else 0.0
            for name, lats in self._routing_latencies.items()
        }

        return {
            "total_predictions": self._total_predictions,
            "routing_counts": dict(self._routing_counts),
            "routing_fractions": fractions,
            "mean_latencies_ms": mean_latencies,
            "experts_available": list(self._classifiers.keys()),
        }

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AdaptiveClassifierRouter("
            f"experts={list(self._classifiers.keys())}, "
            f"fitted={self._fitted}, "
            f"gating={'MLP' if self._enable_gating else 'rules'}, "
            f"predictions={self._total_predictions})"
        )
