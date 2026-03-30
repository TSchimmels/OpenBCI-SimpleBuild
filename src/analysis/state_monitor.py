"""Early Warning Signal (EWS) detector for BCI state monitoring.

Detects cognitive state transitions (fatigue onset, attention lapse,
electrode degradation) BEFORE performance drops, using critical slowing
down indicators from dynamical systems theory.

EWS Indicators:
- Rising autocorrelation at lag-1 (critical slowing down)
- Rising variance (fluctuations increase before tipping)
- Spectral reddening (power shifts to lower frequencies)
- Flickering index (rapid switching between states)

References:
    Scheffer et al. (2012). Anticipating Critical Transitions. Science.
    Bury et al. (2023). Detecting and distinguishing tipping points
    using spectral early warning signals. PNAS.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch
from scipy.stats import linregress

logger = logging.getLogger(__name__)


class BCIStateMonitor:
    """Monitor subject cognitive state via early warning signals.

    Computes rolling EWS indicators on EEG data and recent classification
    accuracy to detect fatigue, attention lapses, and electrode degradation
    before they impact BCI performance.

    Args:
        sf: Sampling frequency in Hz.
        n_channels: Number of EEG channels.
        window_s: Rolling window for EWS computation (seconds).
        update_interval_s: Minimum seconds between state assessments.
        history_length: Number of past assessments to retain for trend analysis.
    """

    def __init__(
        self,
        sf: int = 125,
        n_channels: int = 16,
        window_s: float = 30.0,
        update_interval_s: float = 5.0,
        history_length: int = 60,
    ) -> None:
        self.sf = sf
        self.n_channels = n_channels
        self.window_samples = int(window_s * sf)
        self.update_interval_s = update_interval_s

        # Rolling EEG buffer
        self._eeg_buffer = np.zeros((n_channels, self.window_samples))
        self._buffer_fill = 0

        # Accuracy history
        self._accuracy_history: deque = deque(maxlen=history_length)

        # EWS history for trend detection
        self._ews_history: deque = deque(maxlen=history_length)

        # Timing
        self._last_update_time: float = 0.0

        # Thresholds
        self._fatigue_threshold = 0.6
        self._attention_threshold = 0.5
        self._electrode_threshold = 0.4

        logger.info(
            "BCIStateMonitor: sf=%d, window=%.0fs, update_interval=%.0fs",
            sf, window_s, update_interval_s,
        )

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        eeg_chunk: np.ndarray,
        classification_accuracy: Optional[float] = None,
        current_time: float = 0.0,
    ) -> Optional[Dict]:
        """Process new EEG data and assess cognitive state.

        Args:
            eeg_chunk: New EEG data, shape (n_channels, n_samples).
            classification_accuracy: Recent accuracy (0-1), if available.
            current_time: Current timestamp.

        Returns:
            State assessment dict, or None if not enough time has passed.
        """
        # Update buffer
        n_new = eeg_chunk.shape[1]
        if n_new >= self.window_samples:
            self._eeg_buffer = eeg_chunk[:, -self.window_samples:]
            self._buffer_fill = self.window_samples
        else:
            self._eeg_buffer = np.roll(self._eeg_buffer, -n_new, axis=1)
            self._eeg_buffer[:, -n_new:] = eeg_chunk
            self._buffer_fill = min(self._buffer_fill + n_new, self.window_samples)

        if classification_accuracy is not None:
            self._accuracy_history.append(classification_accuracy)

        # Check timing
        if current_time - self._last_update_time < self.update_interval_s:
            return None
        if self._buffer_fill < self.window_samples // 2:
            return None

        self._last_update_time = current_time

        # Compute EWS
        data = self._eeg_buffer[:, -self._buffer_fill:]
        ews = self._compute_ews(data)
        self._ews_history.append(ews)

        # Assess state
        fatigue = self._detect_fatigue(ews)
        attention = self._detect_attention(ews)
        electrode_quality = self._estimate_electrode_quality(data)
        time_to_deg = self._predict_time_to_transition()

        # Overall state
        worst_score = max(fatigue, 1.0 - attention, 1.0 - electrode_quality)
        if worst_score > 0.7:
            state = "degraded"
        elif worst_score > 0.4:
            state = "warning"
        else:
            state = "stable"

        # Recommendation
        recommendation = self._generate_recommendation(
            state, fatigue, attention, electrode_quality
        )

        result = {
            "state": state,
            "fatigue_score": fatigue,
            "attention_score": attention,
            "electrode_quality": electrode_quality,
            "ews_indicators": ews,
            "recommendation": recommendation,
            "time_to_degradation_s": time_to_deg,
        }

        return result

    # ------------------------------------------------------------------
    # EWS computation
    # ------------------------------------------------------------------

    def _compute_ews(self, data: np.ndarray) -> Dict:
        """Compute all early warning signal indicators.

        Args:
            data: EEG data, shape (n_channels, n_samples).

        Returns:
            Dict with EWS indicator values.
        """
        # Use mean across motor cortex channels for EWS
        signal = data.mean(axis=0)

        # 1. Autocorrelation at lag-1 (critical slowing down)
        n = len(signal)
        if n > 2:
            mean_s = signal.mean()
            var_s = signal.var()
            if var_s > 0:
                ac1 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
                if not np.isfinite(ac1):
                    ac1 = 0.0
            else:
                ac1 = 0.0
        else:
            ac1 = 0.0

        # 2. Rolling variance trend
        chunk_size = max(self.sf * 2, 1)  # 2-second chunks
        n_chunks = n // chunk_size
        if n_chunks >= 3:
            variances = []
            for i in range(n_chunks):
                chunk = signal[i * chunk_size:(i + 1) * chunk_size]
                variances.append(chunk.var())
            var_array = np.array(variances)
            if len(var_array) >= 3:
                slope, _, _, _, _ = linregress(
                    np.arange(len(var_array)), var_array
                )
                variance_trend = float(slope)
            else:
                variance_trend = 0.0
        else:
            variance_trend = 0.0

        # 3. Spectral reddening (ratio of low-freq to high-freq power)
        try:
            freqs, psd = welch(signal, fs=self.sf, nperseg=min(256, n))
            low_mask = freqs <= 4.0
            high_mask = (freqs > 4.0) & (freqs <= 30.0)
            low_power = psd[low_mask].sum() if low_mask.any() else 1.0
            high_power = psd[high_mask].sum() if high_mask.any() else 1.0
            spectral_ratio = low_power / max(high_power, 1e-10)
        except Exception:
            spectral_ratio = 1.0

        # 4. Flickering index (bimodality via Hartigan's dip-like measure)
        if n > 10:
            sorted_s = np.sort(signal)
            mid = n // 2
            lower_var = sorted_s[:mid].var()
            upper_var = sorted_s[mid:].var()
            total_var = signal.var()
            flickering = 1.0 - (lower_var + upper_var) / max(2.0 * total_var, 1e-10)
            flickering = max(0.0, min(1.0, flickering))
        else:
            flickering = 0.0

        return {
            "autocorrelation_lag1": ac1,
            "variance_trend": variance_trend,
            "spectral_reddening": spectral_ratio,
            "flickering_index": flickering,
        }

    # ------------------------------------------------------------------
    # State detection
    # ------------------------------------------------------------------

    def _detect_fatigue(self, ews: Dict) -> float:
        """Detect fatigue from EWS + accuracy trend. Returns 0-1."""
        score = 0.0

        # Rising autocorrelation → critical slowing → fatigue
        if ews["autocorrelation_lag1"] > 0.7:
            score += 0.3
        elif ews["autocorrelation_lag1"] > 0.5:
            score += 0.15

        # Rising variance → approaching state transition
        if ews["variance_trend"] > 0:
            score += min(0.3, ews["variance_trend"] * 100)

        # Declining accuracy trend
        if len(self._accuracy_history) >= 5:
            recent = list(self._accuracy_history)[-10:]
            if len(recent) >= 5:
                slope, _, _, _, _ = linregress(np.arange(len(recent)), recent)
                if slope < -0.01:
                    score += min(0.4, abs(slope) * 20)

        return min(1.0, score)

    def _detect_attention(self, ews: Dict) -> float:
        """Detect attention level. Returns 0-1 (1 = fully attentive)."""
        score = 1.0

        # Spectral reddening → mind wandering (alpha increase)
        if ews["spectral_reddening"] > 3.0:
            score -= 0.3
        elif ews["spectral_reddening"] > 2.0:
            score -= 0.15

        # High flickering → unstable attention
        score -= ews["flickering_index"] * 0.3

        # Low autocorrelation can indicate random/unfocused state
        if ews["autocorrelation_lag1"] < 0.1:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _estimate_electrode_quality(self, data: np.ndarray) -> float:
        """Estimate electrode contact quality. Returns 0-1 (1 = good)."""
        quality = 1.0

        # Check for flatline channels (dead electrodes)
        channel_stds = data.std(axis=1)
        median_std = np.median(channel_stds)
        if median_std > 0:
            flatline_count = np.sum(channel_stds < 0.1 * median_std)
            quality -= flatline_count * 0.1

        # Check for extremely noisy channels
        if median_std > 0:
            noisy_count = np.sum(channel_stds > 5.0 * median_std)
            quality -= noisy_count * 0.1

        # Check for 50/60 Hz line noise dominance
        try:
            for ch in range(min(data.shape[0], 4)):
                freqs, psd = welch(data[ch], fs=self.sf, nperseg=min(256, data.shape[1]))
                total = psd.sum()
                if total > 0:
                    line_mask = ((freqs > 49) & (freqs < 51)) | ((freqs > 59) & (freqs < 61))
                    line_power = psd[line_mask].sum()
                    if line_power / total > 0.3:
                        quality -= 0.15
        except Exception:
            pass

        return max(0.0, min(1.0, quality))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_time_to_transition(self) -> Optional[float]:
        """Estimate seconds until state degrades, based on EWS trends."""
        if len(self._ews_history) < 5:
            return None

        # Use autocorrelation trend to predict when it crosses threshold
        ac_values = [h["autocorrelation_lag1"] for h in self._ews_history]
        if len(ac_values) < 5:
            return None

        x = np.arange(len(ac_values))
        slope, intercept, _, _, _ = linregress(x, ac_values)

        if slope <= 0:
            return None  # Not trending toward degradation

        # When will AC cross 0.8 (degradation threshold)?
        target = 0.8
        current = ac_values[-1]
        if current >= target:
            return 0.0

        steps_to_target = (target - current) / slope
        return steps_to_target * self.update_interval_s

    def _generate_recommendation(
        self,
        state: str,
        fatigue: float,
        attention: float,
        electrode_quality: float,
    ) -> str:
        """Generate human-readable recommendation."""
        if state == "stable":
            return "Signal quality good. Continue."

        issues = []
        if fatigue > 0.6:
            issues.append("Take a 2-minute break (fatigue detected)")
        elif fatigue > 0.4:
            issues.append("Mild fatigue — consider a short break soon")

        if attention < 0.4:
            issues.append("Refocus attention on the imagery task")
        elif attention < 0.6:
            issues.append("Attention drifting — try to concentrate")

        if electrode_quality < 0.5:
            issues.append("Check electrode contact (signal degradation)")
        elif electrode_quality < 0.7:
            issues.append("Some electrodes may need gel refresh")

        return "; ".join(issues) if issues else "Monitoring."

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state."""
        self._eeg_buffer = np.zeros((self.n_channels, self.window_samples))
        self._buffer_fill = 0
        self._accuracy_history.clear()
        self._ews_history.clear()
        self._last_update_time = 0.0

    def get_history(self) -> List[Dict]:
        """Return EWS assessment history."""
        return list(self._ews_history)

    def __repr__(self) -> str:
        return (
            f"BCIStateMonitor(sf={self.sf}, window={self.window_samples/self.sf:.0f}s, "
            f"history={len(self._ews_history)})"
        )
