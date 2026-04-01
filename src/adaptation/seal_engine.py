"""SEAL Adaptation Engine — Self-Evolving Adaptive Learning for BCI.

Uses involuntary ErrP/P300 brain signals as reward signals to
continuously update the MI classifier without explicit user feedback.

The brain's error monitoring system (anterior cingulate cortex) provides
the reward signal automatically after every action:
- P300 at parietal sites → "correct" → reinforce this pattern
- ErrP at frontocentral sites → "error" → correct and undo

Architecture:
- Positive buffer: EEG patterns confirmed correct by P300
- Negative buffer: EEG patterns marked as errors by ErrP
- Replay buffer: Original calibration data (prevents forgetting)
- Periodic update: Every N seconds, fine-tune the model

References:
    Schmidt & Blankertz (2010). ErrP detection → 49% faster typing.
    MIT SEAL framework adapted for neural interfaces.
    Wirth et al. (2025). RL-driven adaptive BCI with error-related potentials.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SEALAdaptationEngine:
    """Self-Evolving Adaptive Learning engine for BCI classifiers.

    Accumulates reward signals from the ErrP/P300 detector and
    periodically updates the classifier. Maintains a replay buffer
    from original calibration data to prevent catastrophic forgetting.

    Args:
        config: Full application configuration dictionary.
    """

    def __init__(self, config: Dict) -> None:
        adapt_cfg = config.get("adaptation", {})

        # Timing
        self.update_interval_s: float = adapt_cfg.get("update_interval_s", 30.0)
        self.min_samples_for_update: int = adapt_cfg.get("min_samples_for_update", 5)

        # Buffer sizes
        self.replay_buffer_size: int = adapt_cfg.get("replay_buffer_size", 500)
        self.adaptation_buffer_size: int = adapt_cfg.get("adaptation_buffer_size", 100)

        # Learning parameters
        self.blend_ratio: float = adapt_cfg.get("blend_ratio", 0.3)
        self.covariance_alpha: float = adapt_cfg.get("covariance_alpha", 0.05)

        # Buffers
        self._positive_buffer: deque = deque(maxlen=self.adaptation_buffer_size)
        self._negative_buffer: deque = deque(maxlen=self.adaptation_buffer_size)
        self._replay_buffer: deque = deque(maxlen=self.replay_buffer_size)
        self._pending: deque = deque(maxlen=100)

        # State
        self._last_update_time: float = 0.0
        self._classifier = None
        self._class_names: Optional[List[str]] = None

        # Undo history (last N actions for auto-correction)
        self._action_history: deque = deque(maxlen=20)

        # Statistics
        self.n_adaptations: int = 0
        self.n_confirmed: int = 0
        self.n_corrected: int = 0
        self.n_updates: int = 0

        # Enabled flag
        self.enabled: bool = adapt_cfg.get("enabled", True)

        logger.info(
            "SEALAdaptationEngine: update_interval=%.0fs, "
            "min_samples=%d, blend_ratio=%.2f, replay_size=%d, enabled=%s",
            self.update_interval_s, self.min_samples_for_update,
            self.blend_ratio, self.replay_buffer_size, self.enabled,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_classifier(self, classifier, class_names: List[str]) -> None:
        """Register the active classifier for adaptation.

        Args:
            classifier: A fitted BaseClassifier instance.
            class_names: Class name list in label order.
        """
        self._classifier = classifier
        self._class_names = class_names
        logger.info("SEAL: Classifier registered (%r)", classifier)

    def load_replay_buffer(
        self, X: np.ndarray, y: np.ndarray
    ) -> None:
        """Load original calibration data into the replay buffer.

        This data is mixed with new reward-labeled data during updates
        to prevent catastrophic forgetting.

        Args:
            X: Training epochs, shape (n_trials, n_channels, n_samples).
            y: Integer labels, shape (n_trials,).
        """
        for i in range(X.shape[0]):
            self._replay_buffer.append({
                "eeg": X[i],
                "label": int(y[i]),
            })
        logger.info(
            "SEAL: Replay buffer loaded with %d calibration epochs",
            len(self._replay_buffer),
        )

    # ------------------------------------------------------------------
    # Prediction tracking
    # ------------------------------------------------------------------

    def on_prediction(
        self,
        eeg_epoch: np.ndarray,
        predicted_class: int,
        action_time: float,
        action_type: str = "move",
    ) -> None:
        """Called after every classifier prediction + action execution.

        Args:
            eeg_epoch: The EEG epoch that was classified.
            predicted_class: Integer class label predicted.
            action_time: Timestamp of the action.
            action_type: "move", "click", or "none".
        """
        if not self.enabled:
            return

        entry = {
            "eeg": eeg_epoch.copy() if eeg_epoch is not None else None,
            "predicted_class": predicted_class,
            "action_time": action_time,
            "action_type": action_type,
        }
        self._pending.append(entry)
        self._action_history.append(entry)

    # ------------------------------------------------------------------
    # Reward signal processing
    # ------------------------------------------------------------------

    def on_errp_result(
        self, action_time: float, result: str, confidence: float = 0.0
    ) -> Optional[Dict]:
        """Process an ErrP/P300 detection result.

        Called by the control loop when the ErrP detector classifies
        the brain's response to a past action.

        Args:
            action_time: Which action this refers to.
            result: "correct", "error", or "neutral".
            confidence: Detection confidence (0-1).

        Returns:
            Dict with adaptation info, or None.
        """
        if not self.enabled:
            return None

        # Find the corresponding prediction
        pending = self._find_pending(action_time)
        if pending is None:
            return None

        info = {
            "result": result,
            "predicted_class": pending["predicted_class"],
            "action_type": pending["action_type"],
            "confidence": confidence,
            "should_undo": False,
        }

        if result == "correct" and confidence > 0.3:
            # Brain confirms: this EEG→class mapping is RIGHT
            if pending["eeg"] is not None:
                self._positive_buffer.append({
                    "eeg": pending["eeg"],
                    "label": pending["predicted_class"],
                })
            self.n_confirmed += 1
            logger.debug(
                "SEAL: Confirmed correct (class=%d, conf=%.2f)",
                pending["predicted_class"], confidence,
            )

        elif result == "error" and confidence > 0.3:
            # Brain says ERROR: the predicted class is WRONG
            if pending["eeg"] is not None:
                self._negative_buffer.append({
                    "eeg": pending["eeg"],
                    "wrong_label": pending["predicted_class"],
                })
            self.n_corrected += 1
            info["should_undo"] = True
            logger.info(
                "SEAL: Error detected (class=%d was WRONG, conf=%.2f) — undo recommended",
                pending["predicted_class"], confidence,
            )

        return info

    def _find_pending(self, action_time: float) -> Optional[Dict]:
        """Find and remove a pending action by timestamp."""
        for i, entry in enumerate(self._pending):
            if abs(entry["action_time"] - action_time) < 0.05:
                self._pending.remove(entry)
                return entry
        return None

    # ------------------------------------------------------------------
    # Model update
    # ------------------------------------------------------------------

    def maybe_update(self, current_time: float) -> bool:
        """Check if it's time to update the model.

        Call this every control loop iteration. Only actually updates
        when enough time has passed AND enough new data has accumulated.

        Args:
            current_time: Current timestamp.

        Returns:
            True if the model was updated.
        """
        if not self.enabled or self._classifier is None:
            return False

        if current_time - self._last_update_time < self.update_interval_s:
            return False

        n_new = len(self._positive_buffer) + len(self._negative_buffer)
        if n_new < self.min_samples_for_update:
            return False

        success = self._update_model()
        self._last_update_time = current_time
        return success

    def _update_model(self) -> bool:
        """Apply accumulated reward signals to update the classifier.

        Blends new reward-labeled data with replay buffer to prevent
        catastrophic forgetting.
        """
        if self._classifier is None:
            return False

        # Build training batch
        X_new = []
        y_new = []

        # 1. Positive buffer (confirmed correct)
        for entry in self._positive_buffer:
            X_new.append(entry["eeg"])
            y_new.append(entry["label"])

        # 2. Negative buffer (known errors — use classifier's second-best
        #    prediction as the correction label to avoid class imbalance)
        for entry in self._negative_buffer:
            wrong_label = entry["wrong_label"]
            if self._classifier is not None and entry["eeg"] is not None:
                try:
                    proba = self._classifier.predict_proba(
                        entry["eeg"][np.newaxis]
                    )[0]
                    # Zero out the wrong class and pick the next best
                    proba[wrong_label] = 0.0
                    correction_label = int(np.argmax(proba))
                    X_new.append(entry["eeg"])
                    y_new.append(correction_label)
                except Exception:
                    # Fallback: exclude error epochs if classifier fails
                    pass

        if len(X_new) == 0:
            return False

        X_new = np.stack(X_new, axis=0)
        y_new = np.array(y_new, dtype=np.int64)

        # 3. Mix with replay buffer
        if self.blend_ratio > 0 and self.blend_ratio < 1.0:
            n_replay = int(len(X_new) / self.blend_ratio * (1 - self.blend_ratio))
        else:
            n_replay = 0
        n_replay = min(n_replay, len(self._replay_buffer))

        if n_replay > 0:
            replay_indices = np.random.choice(
                len(self._replay_buffer), size=n_replay, replace=False
            )
            X_replay = np.stack(
                [self._replay_buffer[i]["eeg"] for i in replay_indices]
            )
            y_replay = np.array(
                [self._replay_buffer[i]["label"] for i in replay_indices],
                dtype=np.int64,
            )
            X_combined = np.concatenate([X_new, X_replay], axis=0)
            y_combined = np.concatenate([y_new, y_replay], axis=0)
        else:
            X_combined = X_new
            y_combined = y_new

        # 4. Update the classifier
        try:
            self._apply_update(X_combined, y_combined)
            self.n_updates += 1
            logger.info(
                "SEAL: Model updated (#%d) — %d new + %d replay = %d total samples. "
                "Confirmed=%d, Corrected=%d",
                self.n_updates, len(X_new), n_replay, len(X_combined),
                self.n_confirmed, self.n_corrected,
            )
        except Exception as exc:
            logger.warning("SEAL: Model update failed: %s", exc)
            return False

        # Add confirmed-correct data to replay buffer BEFORE clearing
        for entry in list(self._positive_buffer):
            self._replay_buffer.append(entry)

        # Clear adaptation buffers (keep replay buffer)
        self._positive_buffer.clear()
        self._negative_buffer.clear()

        self.n_adaptations += 1
        return True

    def _apply_update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Apply the actual model update.

        Strategy depends on classifier type:
        - CSP+LDA: Refit on combined data (replay + new)
        - EEGNet: Fine-tune last layers (1-3 gradient steps)
        - Riemannian: Refit on combined data
        """
        from src.classification.csp_lda import CSPLDAClassifier
        from src.classification.eegnet import EEGNetClassifier
        from src.classification.pipeline import RiemannianClassifier

        if isinstance(self._classifier, CSPLDAClassifier):
            # Refit CSP+LDA on combined data
            # The shrinkage LDA handles small sample sizes well
            self._classifier.fit(X, y)

        elif isinstance(self._classifier, EEGNetClassifier):
            # EEGNet — fine-tune last 2 layers only
            self._finetune_eegnet(X, y)

        elif isinstance(self._classifier, RiemannianClassifier):
            # Riemannian — refit on combined data
            self._classifier.fit(X, y)

        else:
            # Generic fallback — full refit
            self._classifier.fit(X, y)

    def _finetune_eegnet(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fine-tune EEGNet with tiny learning rate, 1-3 gradient steps."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.warning("PyTorch not available for EEGNet fine-tuning")
            return

        model = self._classifier._model
        if model is None:
            return

        # Freeze all layers except classifier head
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        # Also unfreeze last batch norm
        for param in model.bn3.parameters():
            param.requires_grad = True

        # Convert data
        X_t = self._classifier._numpy_to_tensor(X)
        y_t = torch.from_numpy(y.astype(np.int64)).to(self._classifier.device)

        # Tiny learning rate, 1-3 steps
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4,
        )
        criterion = nn.CrossEntropyLoss()

        model.train()
        for step in range(min(3, max(1, len(X) // 8))):
            optimizer.zero_grad()
            logits = model(X_t)
            loss = criterion(logits, y_t)
            loss.backward()
            optimizer.step()

        model.eval()

        # Unfreeze all for future full retraining
        for param in model.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Undo support
    # ------------------------------------------------------------------

    def get_last_action(self) -> Optional[Dict]:
        """Get the most recent action for undo purposes."""
        if self._action_history:
            return self._action_history[-1]
        return None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict:
        """Return adaptation statistics."""
        return {
            "enabled": self.enabled,
            "n_adaptations": self.n_adaptations,
            "n_confirmed": self.n_confirmed,
            "n_corrected": self.n_corrected,
            "n_updates": self.n_updates,
            "positive_buffer": len(self._positive_buffer),
            "negative_buffer": len(self._negative_buffer),
            "replay_buffer": len(self._replay_buffer),
            "pending": len(self._pending),
        }

    def reset(self) -> None:
        """Reset all state."""
        self._positive_buffer.clear()
        self._negative_buffer.clear()
        self._pending.clear()
        self._action_history.clear()
        self.n_adaptations = 0
        self.n_confirmed = 0
        self.n_corrected = 0
        self.n_updates = 0
        self._last_update_time = 0.0

    def __repr__(self) -> str:
        return (
            f"SEALAdaptationEngine(updates={self.n_updates}, "
            f"confirmed={self.n_confirmed}, corrected={self.n_corrected}, "
            f"enabled={self.enabled})"
        )
