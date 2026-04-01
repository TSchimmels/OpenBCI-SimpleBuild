"""Advanced 5-phase GPU-accelerated training pipeline.

Implements the training program from TRAINING_DESIGN.md:
  Phase 1: Subject profiling (Koopman + causal channels)
  Phase 2: Data preparation (FBCSP + augmentation)
  Phase 3: Multi-model training (4 classifier types)
  Phase 4: Ensemble construction
  Phase 5: Training report
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SubjectProfile:
    """Phase 1 output: discovered subject-specific parameters."""

    mu_band: Tuple[float, float] = (10.0, 2.0)
    beta_band: Tuple[float, float] = (15.0, 28.0)
    optimal_window: Tuple[float, float] = (1.5, 4.0)
    important_channels: Dict[str, List[int]] = field(default_factory=dict)
    hub_channels: Dict[str, List[int]] = field(default_factory=dict)
    trial_weights: Optional[np.ndarray] = None
    bad_channels: List[int] = field(default_factory=list)
    bad_trials: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.trial_weights is not None:
            d["trial_weights"] = self.trial_weights.tolist()
        return d

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("SubjectProfile saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "SubjectProfile":
        with open(path) as f:
            d = json.load(f)
        tw = d.pop("trial_weights", None)
        profile = cls(**d)
        if tw is not None:
            profile.trial_weights = np.array(tw)
        return profile


@dataclass
class ModelResult:
    """Result from training one model type."""

    name: str
    classifier: Any  # BaseClassifier instance
    cv_accuracy: float = 0.0
    cv_std: float = 0.0
    cv_scores: List[float] = field(default_factory=list)
    train_accuracy: float = 0.0
    prediction_time_ms: float = 0.0
    n_params: int = 0


# ---------------------------------------------------------------------------
# Phase 1: Subject Profiling
# ---------------------------------------------------------------------------


class SubjectProfiler:
    """Discover subject-specific parameters from calibration data.

    Uses :class:`KoopmanEEGDecomposition` for spectral profiling and
    :class:`CausalChannelDiscovery` for spatial profiling.
    """

    def __init__(self, config: Dict) -> None:
        self._config = config
        board_cfg = config.get("board", {})
        train_cfg = config.get("training", {})
        preproc_cfg = config.get("preprocessing", {})
        adv_cfg = config.get("advanced", {})

        self._n_channels: int = board_cfg.get("channel_count", 16)
        self._sf: int = board_cfg.get("sampling_rate_override") or 125
        self._n_classes: int = train_cfg.get("n_classes", 5)
        self._class_names: List[str] = train_cfg.get(
            "classes", ["rest", "left_hand", "right_hand", "feet", "tongue"]
        )
        self._artifact_threshold: float = preproc_cfg.get(
            "artifact_threshold_uv", 100.0
        )

        # Koopman params
        self._koopman_n_modes: int = adv_cfg.get("koopman_n_modes", 10)
        self._koopman_delay: int = adv_cfg.get("koopman_delay_dim", 5)

        # Causal params
        self._causal_top_k: int = adv_cfg.get("causal_top_k", 6)

    def profile(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        continuous_data: Optional[np.ndarray] = None,
        sf: Optional[int] = None,
    ) -> SubjectProfile:
        """Run all profiling phases and return a SubjectProfile.

        Args:
            epochs: Shape ``(n_trials, n_channels, n_samples)``.
            labels: Shape ``(n_trials,)``.
            continuous_data: Optional continuous recording for Koopman.
                If ``None``, epochs are concatenated.
            sf: Sampling frequency override.

        Returns:
            Populated :class:`SubjectProfile`.
        """
        sf = sf or self._sf
        profile = SubjectProfile()

        # Phase 1.1 — spectral profile (Koopman)
        mu_band = self._spectral_profile(epochs, continuous_data, sf)
        profile.mu_band = mu_band
        profile.beta_band = (15.0, 28.0)

        # Phase 1.2 — spatial profile (causal channels)
        imp_ch, hub_ch = self._spatial_profile(epochs, labels, sf)
        profile.important_channels = imp_ch
        profile.hub_channels = hub_ch

        # Phase 1.3 — temporal profile (window grid search)
        opt_window = self._temporal_profile(epochs, labels, sf)
        profile.optimal_window = opt_window

        # Phase 1.4 — quality profile
        weights, bad_ch, bad_tr = self._quality_profile(epochs)
        profile.trial_weights = weights
        profile.bad_channels = bad_ch
        profile.bad_trials = bad_tr

        logger.info(
            "SubjectProfile complete: mu=%.1f-%.1f Hz, window=%.2f-%.2f s, "
            "%d bad channels, %d bad trials",
            mu_band[0] - mu_band[1],
            mu_band[0] + mu_band[1],
            opt_window[0],
            opt_window[1],
            len(bad_ch),
            len(bad_tr),
        )
        return profile

    # -- internal methods --

    def _spectral_profile(
        self,
        epochs: np.ndarray,
        continuous_data: Optional[np.ndarray],
        sf: int,
    ) -> Tuple[float, float]:
        """Use Koopman decomposition to find subject-specific mu band."""
        try:
            from ..analysis.koopman_decomposition import KoopmanEEGDecomposition
        except ImportError:
            logger.warning("Koopman module unavailable; using default mu band.")
            return (10.0, 2.0)

        if continuous_data is None:
            continuous_data = epochs.reshape(epochs.shape[1], -1)

        n_ch = continuous_data.shape[0]
        koopman = KoopmanEEGDecomposition(
            n_channels=n_ch,
            sf=sf,
            n_modes=self._koopman_n_modes,
            delay_embedding_dim=self._koopman_delay,
        )
        try:
            koopman.fit(continuous_data)
            center, bw = koopman.get_subject_mu_band()
            logger.info(
                "Koopman spectral profile: mu center=%.2f Hz, bw=%.2f Hz",
                center, bw,
            )
            return (center, bw)
        except Exception as exc:
            logger.warning("Koopman fitting failed (%s); using defaults.", exc)
            return (10.0, 2.0)

    def _spatial_profile(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        sf: int,
    ) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Use causal channel discovery to find important channels."""
        try:
            from ..analysis.causal_channels import CausalChannelDiscovery
        except ImportError:
            logger.warning("CausalChannelDiscovery unavailable; skipping.")
            return {}, {}

        n_ch = epochs.shape[1]
        ccd = CausalChannelDiscovery(
            n_channels=n_ch,
            sf=sf,
            class_names=self._class_names[: self._n_classes],
        )
        try:
            ccd.discover(epochs, labels)
        except Exception as exc:
            logger.warning("Causal discovery failed (%s); skipping.", exc)
            return {}, {}

        important: Dict[str, List[int]] = {}
        hubs: Dict[str, List[int]] = {}
        for cls_name in ccd.discovered_classes:
            important[cls_name] = ccd.get_important_channels(
                cls_name, top_k=self._causal_top_k
            )
            hubs[cls_name] = ccd.get_hub_channels(cls_name, top_k=3)

        return important, hubs

    def _temporal_profile(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        sf: int,
    ) -> Tuple[float, float]:
        """Grid search over classification window start/end times."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        starts = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        ends = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

        # Determine the full trial length in seconds
        n_samples_total = epochs.shape[2]
        trial_duration = n_samples_total / sf

        best_acc = -1.0
        best_window = (1.5, 4.0)

        for start in starts:
            for end in ends:
                if end <= start + 0.5:
                    continue
                if end > trial_duration:
                    continue

                s_idx = int(start * sf)
                e_idx = int(end * sf)
                if e_idx > n_samples_total:
                    continue

                windowed = epochs[:, :, s_idx:e_idx]

                # Quick CSP+LDA: compute log-variance of each channel as features
                log_var = np.log(
                    np.var(windowed, axis=2) + 1e-12
                )  # (n_trials, n_channels)

                _, class_counts = np.unique(labels.astype(int), return_counts=True)
                n_unique = len(class_counts)
                n_splits = min(3, int(min(class_counts)))
                if n_splits < 2 or n_unique < 2:
                    continue

                cv = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=42
                )
                lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                try:
                    scores = cross_val_score(
                        lda, log_var, labels, cv=cv, scoring="accuracy"
                    )
                    mean_acc = float(np.mean(scores))
                except Exception:
                    continue

                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_window = (start, end)

        logger.info(
            "Temporal profile: optimal window=%.2f-%.2f s (acc=%.1f%%)",
            best_window[0],
            best_window[1],
            best_acc * 100,
        )
        return best_window

    def _quality_profile(
        self, epochs: np.ndarray
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """Estimate per-trial SNR and flag bad channels/trials."""
        # Per-trial RMS
        trial_rms = np.sqrt(np.mean(epochs ** 2, axis=(1, 2)))

        # Per-channel mean amplitude
        ch_rms = np.sqrt(np.mean(epochs ** 2, axis=(0, 2)))

        # Flag bad trials: peak-to-peak > threshold
        ptp = np.ptp(epochs, axis=2).max(axis=1)  # max ptp across channels
        bad_trials = list(np.where(ptp > self._artifact_threshold)[0])

        # Flag bad channels: those with RMS > 3 * median
        median_rms = np.median(ch_rms)
        bad_channels = list(np.where(ch_rms > 3.0 * median_rms)[0])

        # Quality weights: inverse of RMS, normalized
        weights = 1.0 / (trial_rms + 1e-12)
        weights[bad_trials] = 0.0
        total = weights.sum()
        if total > 0:
            weights = weights / total

        logger.info(
            "Quality profile: %d bad trials, %d bad channels",
            len(bad_trials),
            len(bad_channels),
        )
        return weights, bad_channels, bad_trials


# ---------------------------------------------------------------------------
# Phase 2: Data Preparation
# ---------------------------------------------------------------------------


class FilterBankCSP:
    """Filter Bank Common Spatial Patterns (FBCSP).

    Decomposes EEG into multiple frequency sub-bands and applies CSP
    independently on each. Wraps the existing :class:`CSPExtractor`.

    The standard sub-bands are 9 overlapping 4 Hz bands from 4-40 Hz,
    plus an optional subject-specific mu band.
    """

    # Default sub-bands: (low_hz, high_hz)
    DEFAULT_BANDS = [
        (4, 8),
        (8, 12),
        (12, 16),
        (16, 20),
        (20, 24),
        (24, 28),
        (28, 32),
        (32, 36),
        (36, 40),
    ]

    def __init__(
        self,
        n_components: int = 6,
        sf: int = 125,
        bands: Optional[List[Tuple[float, float]]] = None,
        subject_mu_band: Optional[Tuple[float, float]] = None,
    ) -> None:
        self._n_components = n_components
        self._sf = sf
        self._bands = list(bands or self.DEFAULT_BANDS)

        if subject_mu_band is not None:
            center, bw = subject_mu_band
            mu_low = max(1.0, center - bw)
            mu_high = min(sf / 2 - 1, center + bw)
            self._bands.append((mu_low, mu_high))

        self._csp_list: List[Any] = []  # one CSPExtractor per band
        self._sos_filters: List[np.ndarray] = []
        self._fitted = False

        # Pre-compute SOS filters
        nyq = sf / 2.0
        for low, high in self._bands:
            # Clamp to valid range
            lo = max(low / nyq, 0.001)
            hi = min(high / nyq, 0.999)
            if lo >= hi:
                lo = max(0.001, hi - 0.01)
            sos = butter(4, [lo, hi], btype="bandpass", output="sos")
            self._sos_filters.append(sos)

    @property
    def n_bands(self) -> int:
        return len(self._bands)

    @property
    def n_features(self) -> int:
        return self.n_bands * self._n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FilterBankCSP":
        """Fit CSP on each sub-band independently.

        Args:
            X: Epochs, shape ``(n_trials, n_channels, n_samples)``.
            y: Labels, shape ``(n_trials,)``.

        Returns:
            self
        """
        from ..features.csp import CSPExtractor

        self._csp_list = []
        for idx, sos in enumerate(self._sos_filters):
            X_band = self._filter_band(X, sos)
            csp = CSPExtractor(
                n_components=self._n_components, reg="ledoit_wolf", log=True
            )
            csp.fit(X_band, y)
            self._csp_list.append(csp)
            logger.debug(
                "FBCSP band %d (%.0f-%.0f Hz) fitted.",
                idx,
                self._bands[idx][0],
                self._bands[idx][1],
            )

        self._fitted = True
        logger.info(
            "FilterBankCSP fitted: %d bands x %d components = %d features",
            self.n_bands,
            self._n_components,
            self.n_features,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract FBCSP features.

        Args:
            X: Epochs, shape ``(n_trials, n_channels, n_samples)``.

        Returns:
            Features, shape ``(n_trials, n_bands * n_components)``.
        """
        if not self._fitted:
            raise RuntimeError("FilterBankCSP not fitted. Call fit() first.")

        features = []
        for sos, csp in zip(self._sos_filters, self._csp_list):
            X_band = self._filter_band(X, sos)
            feat = csp.transform(X_band)
            features.append(feat)

        return np.hstack(features)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    @staticmethod
    def _filter_band(X: np.ndarray, sos: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to each trial and channel."""
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                out[i, j] = sosfiltfilt(sos, X[i, j])
        return out


class EEGAugmenter:
    """EEG-specific data augmentation for motor imagery.

    Augmentations are designed to simulate real sources of EEG
    variability while preserving physiological plausibility.
    """

    def __init__(self, strength: float = 0.5, sf: int = 125) -> None:
        self._strength = np.clip(strength, 0.0, 1.0)
        self._sf = sf
        self._rng = np.random.RandomState(42)

    def augment(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        profile: Optional[SubjectProfile] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentations and return expanded dataset.

        Args:
            epochs: Shape ``(n_trials, n_channels, n_samples)``.
            labels: Shape ``(n_trials,)``.
            profile: Optional subject profile for quality weighting.

        Returns:
            ``(augmented_epochs, augmented_labels)`` with more trials.
        """
        if self._strength == 0.0:
            return epochs.copy(), labels.copy()

        aug_X = [epochs.copy()]
        aug_y = [labels.copy()]

        # Sliding window (always applied — most impactful)
        sw_X, sw_y = self._sliding_window(epochs, labels)
        aug_X.append(sw_X)
        aug_y.append(sw_y)

        # Stochastic augmentations
        n_trials = epochs.shape[0]
        for trial_idx in range(n_trials):
            trial = epochs[trial_idx]
            label = labels[trial_idx]
            augmented = []

            if self._rng.random() < 0.5 * self._strength:
                augmented.append(self._temporal_jitter(trial))

            if self._rng.random() < 0.3 * self._strength:
                augmented.append(self._gaussian_noise(trial))

            if self._rng.random() < 0.2 * self._strength:
                augmented.append(self._channel_dropout(trial))

            if self._rng.random() < 0.3 * self._strength:
                augmented.append(self._amplitude_scaling(trial))

            if self._rng.random() < 0.2 * self._strength:
                augmented.append(self._time_warp(trial))

            for aug_trial in augmented:
                aug_X.append(aug_trial[np.newaxis])
                aug_y.append(np.array([label]))

        # Within-class mixup
        mixup_X, mixup_y = self._mixup(epochs, labels)
        if mixup_X.shape[0] > 0:
            aug_X.append(mixup_X)
            aug_y.append(mixup_y)

        all_X = np.concatenate(aug_X, axis=0)
        all_y = np.concatenate(aug_y, axis=0)

        # Shuffle
        perm = self._rng.permutation(all_X.shape[0])
        all_X = all_X[perm]
        all_y = all_y[perm]

        logger.info(
            "Augmentation: %d -> %d trials (%.1fx), strength=%.2f",
            epochs.shape[0],
            all_X.shape[0],
            all_X.shape[0] / max(1, epochs.shape[0]),
            self._strength,
        )
        return all_X, all_y

    # -- augmentation methods --

    def _sliding_window(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract overlapping sub-windows from each trial."""
        n_trials, n_ch, n_samp = X.shape
        win_size = int(n_samp * 0.8)
        stride = int(n_samp * 0.4)

        windows = []
        win_labels = []
        for i in range(n_trials):
            start = 0
            while start + win_size <= n_samp:
                window = X[i, :, start : start + win_size]
                # Pad or resize to original length
                padded = np.zeros((n_ch, n_samp))
                padded[:, :win_size] = window
                # Extrapolate with last values
                if win_size < n_samp:
                    padded[:, win_size:] = window[:, -1:]
                windows.append(padded)
                win_labels.append(y[i])
                start += stride

        if not windows:
            return np.empty((0, n_ch, n_samp)), np.empty(0)
        return np.array(windows), np.array(win_labels)

    def _temporal_jitter(self, trial: np.ndarray) -> np.ndarray:
        """Shift trial by +/-100ms."""
        shift = self._rng.randint(
            -int(0.1 * self._sf), int(0.1 * self._sf) + 1
        )
        return np.roll(trial, shift, axis=-1)

    def _gaussian_noise(self, trial: np.ndarray) -> np.ndarray:
        """Add Gaussian noise at 5% of channel std."""
        std = trial.std(axis=-1, keepdims=True)
        noise = self._rng.randn(*trial.shape) * std * 0.05
        return trial + noise

    def _channel_dropout(self, trial: np.ndarray) -> np.ndarray:
        """Zero out 1-2 random channels."""
        result = trial.copy()
        n_drop = self._rng.randint(1, 3)
        drop_idx = self._rng.choice(trial.shape[0], n_drop, replace=False)
        result[drop_idx] = 0.0
        return result

    def _amplitude_scaling(self, trial: np.ndarray) -> np.ndarray:
        """Scale amplitude by 0.8-1.2."""
        scale = self._rng.uniform(0.8, 1.2)
        return trial * scale

    def _time_warp(self, trial: np.ndarray) -> np.ndarray:
        """Stretch/compress time by +/-10%."""
        factor = self._rng.uniform(0.9, 1.1)
        n_samp = trial.shape[-1]
        new_len = int(n_samp * factor)
        indices = np.linspace(0, n_samp - 1, new_len).astype(int)
        warped = trial[:, indices]

        # Resize back to original length
        result = np.zeros_like(trial)
        if new_len >= n_samp:
            result = warped[:, :n_samp]
        else:
            result[:, :new_len] = warped
            result[:, new_len:] = warped[:, -1:]
        return result

    def _mixup(
        self, X: np.ndarray, y: np.ndarray, alpha: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Within-class mixup with probability scaled by strength."""
        mixed_X = []
        mixed_y = []
        unique_labels = np.unique(y)

        for label in unique_labels:
            mask = y == label
            class_X = X[mask]
            n = class_X.shape[0]
            if n < 2:
                continue

            n_mixup = max(1, int(n * 0.2 * self._strength))
            for _ in range(n_mixup):
                i, j = self._rng.choice(n, 2, replace=False)
                lam = self._rng.beta(alpha, alpha)
                mixed = lam * class_X[i] + (1 - lam) * class_X[j]
                mixed_X.append(mixed)
                mixed_y.append(label)

        if not mixed_X:
            return np.empty((0, X.shape[1], X.shape[2])), np.empty(0)
        return np.array(mixed_X), np.array(mixed_y)


# ---------------------------------------------------------------------------
# Phase 3: Multi-Model Training
# ---------------------------------------------------------------------------


class MultiModelTrainer:
    """Train all supported classifier types and cross-validate.

    Uses :class:`ClassifierFactory` to create each model and
    :class:`ModelTrainer` for training and CV.
    """

    # Models to train (must match ClassifierFactory._SUPPORTED_TYPES)
    MODEL_TYPES = ["csp_lda", "eegnet", "riemannian", "neural_sde"]

    def __init__(
        self,
        config: Dict,
        n_splits: int = 5,
        model_types: Optional[List[str]] = None,
    ) -> None:
        self._config = config
        self._n_splits = n_splits
        self._model_types = model_types or self.MODEL_TYPES

    def train_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        profile: Optional[SubjectProfile] = None,
    ) -> List[ModelResult]:
        """Train and cross-validate all model types.

        Args:
            X: Epochs, shape ``(n_trials, n_channels, n_samples)``.
            y: Labels, shape ``(n_trials,)``.
            profile: Optional subject profile (used to tune config).

        Returns:
            List of :class:`ModelResult`, one per model type.
        """
        from .trainer import ModelTrainer

        # Build a config that incorporates profile discoveries
        config = self._build_profile_config(profile)
        trainer = ModelTrainer(config)
        results = []

        for model_type in self._model_types:
            logger.info("Training model: %s", model_type)
            try:
                result = self._train_single(
                    model_type, config, trainer, X, y
                )
                results.append(result)
                logger.info(
                    "%s: CV=%.2f%% +/- %.2f%%",
                    model_type,
                    result.cv_accuracy * 100,
                    result.cv_std * 100,
                )
            except Exception as exc:
                logger.error("Failed to train %s: %s", model_type, exc)
                results.append(
                    ModelResult(name=model_type, classifier=None)
                )

        # Sort by CV accuracy (best first)
        results.sort(key=lambda r: r.cv_accuracy, reverse=True)
        return results

    def _train_single(
        self,
        model_type: str,
        config: Dict,
        trainer: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> ModelResult:
        """Train and evaluate a single model type."""
        from ..classification.pipeline import ClassifierFactory

        # Create classifier (deepcopy to avoid mutating shared config)
        import copy
        model_config = copy.deepcopy(config)
        model_config.setdefault("classification", {})
        model_config["classification"]["model_type"] = model_type
        classifier = ClassifierFactory.create(model_config)

        # Cross-validate
        cv_result = trainer.cross_validate(
            classifier, X, y, n_splits=self._n_splits
        )

        # Re-create and train on full data
        classifier = ClassifierFactory.create(model_config)
        classifier, train_metrics = trainer.train(classifier, X, y)

        # Measure prediction time
        t0 = time.perf_counter()
        for _ in range(10):
            classifier.predict(X[:1])
        pred_time_ms = (time.perf_counter() - t0) / 10 * 1000

        # Estimate parameter count
        n_params = self._count_params(classifier)

        return ModelResult(
            name=model_type,
            classifier=classifier,
            cv_accuracy=cv_result["mean_accuracy"],
            cv_std=cv_result["std_accuracy"],
            cv_scores=cv_result["scores"],
            train_accuracy=train_metrics["accuracy"],
            prediction_time_ms=pred_time_ms,
            n_params=n_params,
        )

    def _build_profile_config(
        self, profile: Optional[SubjectProfile]
    ) -> Dict:
        """Merge profile discoveries into config."""
        import copy

        config = copy.deepcopy(self._config)
        if profile is None:
            return config

        # Apply optimal window
        config.setdefault("training", {})
        config["training"]["classification_window_start"] = profile.optimal_window[0]
        config["training"]["classification_window_end"] = profile.optimal_window[1]

        # Apply discovered mu band for preprocessing
        config.setdefault("preprocessing", {})
        center, bw = profile.mu_band
        config["preprocessing"]["mi_bandpass_low"] = max(1.0, center - bw - 2)
        config["preprocessing"]["mi_bandpass_high"] = min(
            60.0, center + bw + 16
        )  # Include beta

        return config

    @staticmethod
    def _count_params(classifier: Any) -> int:
        """Estimate the number of trainable parameters."""
        model = getattr(classifier, "_model", None)
        if model is not None and hasattr(model, "parameters"):
            try:
                return sum(p.numel() for p in model.parameters())
            except Exception:
                pass

        # Fallback: estimate from sklearn
        pipeline = getattr(classifier, "_pipeline", None)
        if pipeline is not None:
            try:
                lda = pipeline.named_steps.get(
                    "lda", pipeline.named_steps.get("classifier", None)
                )
                if lda is not None and hasattr(lda, "coef_"):
                    return lda.coef_.size + lda.intercept_.size
            except Exception:
                pass

        return 0


# ---------------------------------------------------------------------------
# Phase 4: Ensemble Construction
# ---------------------------------------------------------------------------


class EnsembleBuilder:
    """Build an ensemble from trained model results."""

    def build(
        self,
        model_results: List[ModelResult],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ModelResult:
        """Construct the best ensemble strategy.

        Tries best-single and soft-voting, returns whichever
        has higher validation accuracy.

        Args:
            model_results: Trained models from :class:`MultiModelTrainer`.
            X_val: Validation epochs.
            y_val: Validation labels.

        Returns:
            :class:`ModelResult` for the chosen ensemble.
        """
        # Filter out failed models
        valid = [r for r in model_results if r.classifier is not None]
        if not valid:
            raise ValueError("No valid trained models for ensemble.")

        # Strategy 1: Best single model
        best_single = max(valid, key=lambda r: r.cv_accuracy)

        if len(valid) < 2:
            logger.info(
                "Only 1 valid model; using %s as ensemble.",
                best_single.name,
            )
            return best_single

        # Strategy 2: Soft voting
        ensemble_result = self._soft_voting(valid, X_val, y_val)

        # Pick the better strategy
        if ensemble_result.cv_accuracy > best_single.cv_accuracy:
            logger.info(
                "Soft voting ensemble (%.2f%%) beats best single %s (%.2f%%)",
                ensemble_result.cv_accuracy * 100,
                best_single.name,
                best_single.cv_accuracy * 100,
            )
            return ensemble_result
        else:
            logger.info(
                "Best single model %s (%.2f%%) beats ensemble (%.2f%%)",
                best_single.name,
                best_single.cv_accuracy * 100,
                ensemble_result.cv_accuracy * 100,
            )
            return best_single

    def _soft_voting(
        self,
        model_results: List[ModelResult],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> ModelResult:
        """Soft voting: average class probabilities across models."""
        # Compute weights via CV accuracy
        accs = np.array([r.cv_accuracy for r in model_results])
        weights = accs / accs.sum()

        # Collect predictions on validation set
        all_proba = []
        for r in model_results:
            try:
                proba = r.classifier.predict_proba(X_val)
                all_proba.append(proba)
            except Exception:
                # If predict_proba fails, use uniform
                n_classes = len(np.unique(y_val))
                all_proba.append(
                    np.ones((X_val.shape[0], n_classes)) / n_classes
                )

        # Weighted average
        avg_proba = np.zeros_like(all_proba[0])
        for w, p in zip(weights, all_proba):
            avg_proba += w * p

        # Evaluate
        from sklearn.metrics import accuracy_score

        ensemble_preds = np.argmax(avg_proba, axis=1)
        val_acc = float(accuracy_score(y_val, ensemble_preds))

        # Create a wrapper classifier for the ensemble
        ensemble_clf = _SoftVotingClassifier(model_results, weights)

        return ModelResult(
            name="ensemble_soft_voting",
            classifier=ensemble_clf,
            cv_accuracy=val_acc,
            cv_std=0.0,
            cv_scores=[val_acc],
            train_accuracy=val_acc,
            prediction_time_ms=sum(r.prediction_time_ms for r in model_results),
            n_params=sum(r.n_params for r in model_results),
        )


class _SoftVotingClassifier:
    """Lightweight wrapper for soft-voting ensemble inference."""

    def __init__(
        self,
        model_results: List[ModelResult],
        weights: np.ndarray,
    ) -> None:
        self._models = [r.classifier for r in model_results]
        self._weights = weights
        self._model_names = [r.name for r in model_results]

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        all_proba = []
        n_classes = None
        for m in self._models:
            try:
                p = m.predict_proba(X)
                if n_classes is None:
                    n_classes = p.shape[1]
                all_proba.append(p)
            except Exception:
                # Use first successful model's n_classes for fallback shape
                nc = n_classes if n_classes is not None else 5
                all_proba.append(np.ones((X.shape[0], nc)) / nc)

        avg = np.zeros_like(all_proba[0])
        for w, p in zip(self._weights, all_proba):
            avg += w * p
        return avg

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)

    def save(self, path: str) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Ensemble saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "_SoftVotingClassifier":
        import joblib

        return joblib.load(path)


# ---------------------------------------------------------------------------
# Phase 5: Training Report
# ---------------------------------------------------------------------------


class TrainingReport:
    """Format and save the training report."""

    def __init__(
        self,
        profile: SubjectProfile,
        augmentation_stats: Dict[str, Any],
        model_results: List[ModelResult],
        ensemble_result: Optional[ModelResult],
        best_model_name: str,
        best_accuracy: float,
    ) -> None:
        self.profile = profile
        self.augmentation_stats = augmentation_stats
        self.model_results = model_results
        self.ensemble_result = ensemble_result
        self.best_model_name = best_model_name
        self.best_accuracy = best_accuracy
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def format(self) -> str:
        lines = [
            "=" * 48,
            "ADVANCED TRAINING REPORT",
            "=" * 48,
            "",
            "Subject Profile:",
        ]

        center, bw = self.profile.mu_band
        lines.append(
            f"  Mu band:          {center - bw:.1f} - {center + bw:.1f} Hz "
            f"(center={center:.1f}, bw={bw:.1f})"
        )
        lines.append(
            f"  Beta band:        {self.profile.beta_band[0]:.1f} - "
            f"{self.profile.beta_band[1]:.1f} Hz"
        )
        lines.append(
            f"  Optimal window:   {self.profile.optimal_window[0]:.2f} - "
            f"{self.profile.optimal_window[1]:.2f} s"
        )
        lines.append(
            f"  Bad channels:     {len(self.profile.bad_channels)}"
        )
        lines.append(
            f"  Bad trials:       {len(self.profile.bad_trials)}"
        )

        lines.append("")
        lines.append("Data Augmentation:")
        lines.append(
            f"  Original trials:  {self.augmentation_stats.get('original', 0)}"
        )
        lines.append(
            f"  After augment:    {self.augmentation_stats.get('augmented', 0)}"
        )
        lines.append(
            f"  Multiplier:       {self.augmentation_stats.get('multiplier', 1.0):.1f}x"
        )

        lines.append("")
        lines.append("Model Comparison:")
        lines.append(
            f"  {'Model':<20} {'CV Acc':>8} {'Std':>8} "
            f"{'Pred(ms)':>10} {'Params':>8}"
        )
        lines.append(f"  {'-' * 18:<20} {'-' * 6:>8} {'-' * 6:>8} "
                      f"{'-' * 8:>10} {'-' * 6:>8}")

        for r in self.model_results:
            if r.classifier is None:
                lines.append(f"  {r.name:<20} {'FAILED':>8}")
                continue
            lines.append(
                f"  {r.name:<20} {r.cv_accuracy * 100:>7.1f}% "
                f"{r.cv_std * 100:>7.1f}% {r.prediction_time_ms:>9.1f} "
                f"{r.n_params:>8}"
            )

        if self.ensemble_result and self.ensemble_result.classifier is not None:
            r = self.ensemble_result
            lines.append(
                f"  {r.name:<20} {r.cv_accuracy * 100:>7.1f}% "
                f"{r.cv_std * 100:>7.1f}% {r.prediction_time_ms:>9.1f} "
                f"{r.n_params:>8}"
            )

        lines.append("")
        lines.append(
            f"  Best: {self.best_model_name} -- "
            f"{self.best_accuracy * 100:.1f}%"
        )
        n_classes = len(set(
            self.profile.important_channels.keys()
        )) or 5
        lines.append(f"  Chance: {100.0 / n_classes:.1f}%")

        lines.append("")
        lines.append("=" * 48)
        return "\n".join(lines)

    def save(self, output_dir: str) -> Dict[str, str]:
        """Save report, profile, and best model to output_dir."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: Dict[str, str] = {}

        # Report text
        report_path = out / f"training_report_{self.timestamp}.txt"
        report_path.write_text(self.format())
        paths["report"] = str(report_path)

        # Profile JSON
        profile_path = out / f"subject_profile_{self.timestamp}.json"
        self.profile.save(str(profile_path))
        paths["profile"] = str(profile_path)

        # Best model
        best = self.ensemble_result or (
            self.model_results[0] if self.model_results else None
        )
        if best and best.classifier is not None:
            model_path = out / f"{best.name}_{self.timestamp}.pkl"
            try:
                best.classifier.save(str(model_path))
                paths["model"] = str(model_path)
            except Exception as exc:
                logger.warning("Failed to save model: %s", exc)

        logger.info("Training artifacts saved to %s", output_dir)
        return paths


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------


class AdvancedTrainingPipeline:
    """Orchestrate the full 5-phase advanced training pipeline.

    Args:
        config: Project configuration dictionary.
        augmentation: Augmentation strength 0.0-1.0.
        skip_hyperopt: Skip Optuna optimization.
        output_dir: Where to save results.
    """

    def __init__(
        self,
        config: Dict,
        augmentation: float = 0.5,
        skip_hyperopt: bool = True,
        output_dir: str = "models",
        model_types: Optional[List[str]] = None,
    ) -> None:
        self._config = config
        self._augmentation = augmentation
        self._skip_hyperopt = skip_hyperopt
        self._output_dir = output_dir
        self._model_types = model_types

    def run(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        continuous_data: Optional[np.ndarray] = None,
        sf: Optional[int] = None,
    ) -> TrainingReport:
        """Execute the full pipeline.

        Args:
            epochs: Preprocessed epochs ``(n_trials, n_channels, n_samples)``.
            labels: Integer labels ``(n_trials,)``.
            continuous_data: Optional continuous data for Koopman.
            sf: Sampling frequency override.

        Returns:
            :class:`TrainingReport` with full results.
        """
        t_start = time.time()
        sf = sf or self._config.get("board", {}).get(
            "sampling_rate_override"
        ) or 125

        # ---- Phase 1: Subject Profiling ----
        logger.info("=" * 40)
        logger.info("PHASE 1: Subject Profiling")
        logger.info("=" * 40)
        profiler = SubjectProfiler(self._config)
        profile = profiler.profile(
            epochs, labels, continuous_data=continuous_data, sf=sf
        )

        # Remove bad trials from training data
        good_mask = np.ones(epochs.shape[0], dtype=bool)
        for idx in profile.bad_trials:
            if 0 <= idx < epochs.shape[0]:
                good_mask[idx] = False
        epochs_clean = epochs[good_mask]
        labels_clean = labels[good_mask]
        logger.info(
            "After quality filtering: %d -> %d trials",
            epochs.shape[0],
            epochs_clean.shape[0],
        )

        # ---- Phase 2: Data Preparation (FBCSP + Augmentation) ----
        logger.info("=" * 40)
        logger.info("PHASE 2: Data Preparation (FBCSP + Augmentation)")
        logger.info("=" * 40)

        # Phase 2a: FilterBankCSP feature extraction
        fbcsp = None
        fbcsp_features = None
        try:
            fbcsp = FilterBankCSP(
                n_components=6, sf=sf,
                subject_mu_band=profile.mu_band if profile else None,
            )
            fbcsp.fit(epochs_clean, labels_clean)
            fbcsp_features = fbcsp.transform(epochs_clean)
            logger.info(
                "FBCSP: %d bands x %d components = %d features extracted",
                fbcsp.n_bands, 6, fbcsp_features.shape[1],
            )
        except Exception as exc:
            logger.warning("FBCSP failed (continuing without): %s", exc)

        # Phase 2b: Data augmentation
        augmenter = EEGAugmenter(strength=self._augmentation, sf=sf)
        X_aug, y_aug = augmenter.augment(epochs_clean, labels_clean, profile)

        aug_stats = {
            "original": int(epochs_clean.shape[0]),
            "augmented": int(X_aug.shape[0]),
            "multiplier": round(
                X_aug.shape[0] / max(1, epochs_clean.shape[0]), 1
            ),
            "fbcsp_features": fbcsp_features.shape[1] if fbcsp_features is not None else 0,
        }

        # ---- Phase 3: Multi-Model Training ----
        logger.info("=" * 40)
        logger.info("PHASE 3: Multi-Model Training")
        logger.info("=" * 40)
        multi_trainer = MultiModelTrainer(
            self._config, n_splits=5, model_types=self._model_types
        )
        # Cross-validate on CLEAN data (not augmented) to prevent data leakage.
        # Augmented copies of the same trial in both train/test folds inflate CV accuracy.
        # The augmenter + augmented data are stored for final model retraining.
        model_results = multi_trainer.train_all(epochs_clean, labels_clean, profile)

        # ---- Phase 4: Ensemble ----
        logger.info("=" * 40)
        logger.info("PHASE 4: Ensemble Construction")
        logger.info("=" * 40)
        # Use clean (non-augmented) data for ensemble validation
        ensemble_builder = EnsembleBuilder()
        try:
            ensemble_result = ensemble_builder.build(
                model_results, epochs_clean, labels_clean
            )
        except Exception as exc:
            logger.warning("Ensemble construction failed: %s", exc)
            ensemble_result = None

        # Determine best model
        all_results = model_results[:]
        if ensemble_result is not None:
            all_results.append(ensemble_result)
        valid_results = [r for r in all_results if r.classifier is not None]
        if valid_results:
            best = max(valid_results, key=lambda r: r.cv_accuracy)
        else:
            best = ModelResult(name="none", classifier=None)

        # ---- Phase 5: Report ----
        logger.info("=" * 40)
        logger.info("PHASE 5: Training Report")
        logger.info("=" * 40)
        report = TrainingReport(
            profile=profile,
            augmentation_stats=aug_stats,
            model_results=model_results,
            ensemble_result=ensemble_result,
            best_model_name=best.name,
            best_accuracy=best.cv_accuracy,
        )

        # Print and save
        report_text = report.format()
        logger.info("\n%s", report_text)
        report.save(self._output_dir)

        elapsed = time.time() - t_start
        logger.info(
            "Advanced training complete in %.1f seconds. "
            "Best: %s (%.1f%%)",
            elapsed,
            best.name,
            best.cv_accuracy * 100,
        )

        return report

    @staticmethod
    def from_npz(
        data_path: str,
        config: Dict,
        **kwargs: Any,
    ) -> TrainingReport:
        """Load data from .npz and run the full pipeline.

        Supports two .npz formats:

        1. Pre-epoched: keys ``epochs`` + ``labels`` (+ optional ``sf``).
        2. Raw continuous: keys ``data`` + ``events_json`` + ``sf`` +
           ``eeg_channels`` (as saved by ``collect_training_data.py``
           and ``DataRecorder.save()``).

        Args:
            data_path: Path to .npz file.
            config: Project configuration.
            **kwargs: Passed to :class:`AdvancedTrainingPipeline`.

        Returns:
            :class:`TrainingReport`.
        """
        import json as _json

        data = np.load(data_path, allow_pickle=True)
        keys = list(data.keys())

        if "epochs" in data and "labels" in data:
            # Format 1: pre-epoched (from erp_trainer.py)
            epochs = data["epochs"]
            labels = data["labels"]
            continuous = data.get("data", data.get("raw_data", None))
        elif "data" in data and "events_json" in data:
            # Format 2: raw continuous (from collect_training_data.py /
            # DataRecorder.save)
            from .trainer import ModelTrainer

            trainer = ModelTrainer(config)
            raw = data["data"]
            events = _json.loads(str(data["events_json"]))
            sf = int(data.get("sf", 125))
            eeg_ch_arr = data.get("eeg_channels", None)
            eeg_ch = (
                eeg_ch_arr.tolist()
                if eeg_ch_arr is not None
                else list(range(raw.shape[0]))
            )
            epochs, labels, _ = trainer.prepare_data(raw, events, sf, eeg_ch)
            continuous = raw[eeg_ch] if raw.ndim >= 2 else None
        else:
            raise ValueError(
                f"Unsupported .npz format. Keys: {keys}. "
                f"Expected 'epochs'+'labels' or 'data'+'events_json'."
            )

        sf = int(data.get("sf", 125))
        pipeline = AdvancedTrainingPipeline(config, **kwargs)
        return pipeline.run(
            epochs, labels, continuous_data=continuous, sf=sf
        )
