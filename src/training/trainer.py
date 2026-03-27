"""Offline model trainer for motor imagery BCI.

Orchestrates the full offline training workflow: epoch extraction,
bandpass filtering, artifact rejection, classifier fitting, and
cross-validated evaluation.  Designed to work with any classifier
that implements :class:`~src.classification.base.BaseClassifier`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ..preprocessing.artifacts import reject_epochs
from ..preprocessing.filters import bandpass_filter, common_average_reference
from .recorder import DataRecorder

if TYPE_CHECKING:
    from ..classification.base import BaseClassifier

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Orchestrate offline classifier training and evaluation.

    Handles the pipeline from raw recorded data to a trained, evaluated
    classifier:

    1. **Epoch extraction** from continuous data using event markers.
    2. **Bandpass filtering** (default 8--30 Hz for mu + beta rhythms).
    3. **Artifact rejection** (peak-to-peak threshold).
    4. **Classifier fitting** and cross-validated evaluation.

    Args:
        config: Full application configuration dictionary (as loaded from
            ``config/settings.yaml``).  Reads from the ``training``,
            ``preprocessing``, and ``features`` sections.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Dict) -> None:
        train_cfg = config.get("training", {})
        preproc_cfg = config.get("preprocessing", {})

        # Bandpass for MI feature extraction (mu + beta band)
        self.bp_low: float = preproc_cfg.get("mi_bandpass_low", 8.0)
        self.bp_high: float = preproc_cfg.get("mi_bandpass_high", 30.0)
        self.bp_order: int = preproc_cfg.get("bandpass_order", 4)

        # Artifact rejection threshold
        self.artifact_threshold_uv: float = preproc_cfg.get(
            "artifact_threshold_uv", 100.0
        )

        # Epoch timing relative to cue onset
        self.tmin: float = train_cfg.get("classification_window_start", 1.5)
        self.tmax: float = train_cfg.get("classification_window_end", 4.0)

        # Number of classes (for chance-level calculation)
        self.n_classes: int = train_cfg.get("n_classes", 3)

        logger.info(
            "ModelTrainer: bp=[%.1f, %.1f] Hz, artifact_thresh=%.0f uV, "
            "epoch=[%.1f, %.1f] s, n_classes=%d.",
            self.bp_low,
            self.bp_high,
            self.artifact_threshold_uv,
            self.tmin,
            self.tmax,
            self.n_classes,
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        raw_data: np.ndarray,
        events: List[Dict],
        sf: int,
        eeg_channels: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """Extract, filter, and clean epochs from a raw recording.

        Pipeline:

        1. Extract epochs from continuous data using event markers.
        2. Bandpass filter each epoch (8--30 Hz, zero-phase).
        3. Reject epochs with excessive peak-to-peak amplitude.

        Args:
            raw_data: Continuous recording, shape
                ``(n_total_channels, n_samples)``.
            events: Event marker list (as returned by
                :meth:`DataRecorder.stop` or :meth:`DataRecorder.load`).
            sf: Sampling frequency in Hz.
            eeg_channels: List of EEG channel indices to extract from
                *raw_data*.

        Returns:
            A 3-tuple ``(clean_epochs, labels, label_map)`` where:

            - **clean_epochs** -- 3-D array of shape
              ``(n_clean, n_channels, n_samples)``, bandpass-filtered
              and artifact-free.
            - **labels** -- 1-D integer array of shape ``(n_clean,)``.
            - **label_map** -- dictionary mapping label strings to
              integer indices (e.g. ``{"left_hand": 0, "rest": 1}``).
        """
        # Step 1: epoch extraction
        epochs, labels, self.label_map = DataRecorder.extract_epochs(
            raw_data,
            events,
            sf,
            tmin=self.tmin,
            tmax=self.tmax,
            eeg_channels=eeg_channels,
        )

        if epochs.shape[0] == 0:
            logger.warning("No epochs extracted. Returning empty arrays.")
            return epochs, labels, self.label_map

        logger.info(
            "Extracted %d epochs (%d channels, %d samples).",
            epochs.shape[0],
            epochs.shape[1],
            epochs.shape[2],
        )

        # Step 2: bandpass filter each epoch (zero-phase, offline mode)
        for i in range(epochs.shape[0]):
            epochs[i] = bandpass_filter(
                epochs[i],
                sf=sf,
                low=self.bp_low,
                high=self.bp_high,
                order=self.bp_order,
                causal=False,
            )

        logger.info(
            "Bandpass filtered epochs: [%.1f, %.1f] Hz, order %d.",
            self.bp_low,
            self.bp_high,
            self.bp_order,
        )

        # Step 2b: Common average reference (matches real-time pipeline)
        for i in range(epochs.shape[0]):
            epochs[i] = common_average_reference(epochs[i])

        logger.info("Applied common average reference to %d epochs.", epochs.shape[0])

        # Step 3: artifact rejection
        clean_epochs, clean_labels, rejected = reject_epochs(
            epochs,
            labels,
            threshold_uv=self.artifact_threshold_uv,
        )

        n_rejected = len(rejected)
        n_remaining = clean_epochs.shape[0]
        logger.info(
            "Artifact rejection: %d / %d epochs rejected (threshold=%.0f uV). "
            "%d clean epochs remain.",
            n_rejected,
            epochs.shape[0],
            self.artifact_threshold_uv,
            n_remaining,
        )

        if n_remaining == 0:
            logger.warning(
                "All epochs were rejected. Consider increasing the "
                "artifact threshold or checking electrode impedances."
            )

        return clean_epochs, clean_labels, self.label_map

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        classifier: "BaseClassifier",
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple["BaseClassifier", Dict]:
        """Fit a classifier on the provided data.

        Args:
            classifier: An unfitted classifier implementing
                :class:`BaseClassifier`.
            X: Training epochs, shape ``(n_trials, n_channels, n_samples)``.
            y: Integer class labels, shape ``(n_trials,)``.

        Returns:
            A 2-tuple ``(trained_classifier, metrics)`` where:

            - **trained_classifier** -- the same classifier instance,
              now fitted.
            - **metrics** -- dictionary with at least ``'accuracy'``
              (training accuracy on the full dataset).
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot train on zero epochs.")

        logger.info(
            "Training %r on %d epochs (%d channels, %d samples).",
            classifier,
            X.shape[0],
            X.shape[1],
            X.shape[2],
        )

        classifier.fit(X, y)

        # Compute training accuracy
        y_pred = classifier.predict(X)
        acc = float(accuracy_score(y, y_pred))

        metrics = {"accuracy": acc}

        logger.info("Training accuracy: %.2f%%.", acc * 100)
        return classifier, metrics

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        classifier: "BaseClassifier",
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 10,
    ) -> Dict:
        """Evaluate the classifier using stratified K-fold cross-validation.

        For CSP+LDA classifiers this uses :func:`sklearn.model_selection.cross_val_score`
        which re-fits the full sklearn pipeline (CSP + LDA) on each fold,
        ensuring that spatial filters are learned only from training data.

        Args:
            classifier: A classifier implementing :class:`BaseClassifier`.
                Must expose a ``_pipeline`` attribute (sklearn Pipeline)
                for use with ``cross_val_score``, **or** implement the
                sklearn estimator interface (``fit`` / ``predict``).
            X: Epochs, shape ``(n_trials, n_channels, n_samples)``.
            y: Integer class labels, shape ``(n_trials,)``.
            n_splits: Number of folds (default 10).

        Returns:
            Dictionary with keys:

            - ``'mean_accuracy'`` -- mean accuracy across folds.
            - ``'std_accuracy'`` -- standard deviation of fold accuracies.
            - ``'scores'`` -- list of per-fold accuracy values.
            - ``'chance_level'`` -- theoretical chance level (1 / n_classes).
        """
        if X.shape[0] == 0:
            raise ValueError("Cannot cross-validate on zero epochs.")

        n_classes = len(np.unique(y))
        chance_level = 1.0 / n_classes

        # Guard: too few total trials for any meaningful CV
        if X.shape[0] < 4:
            logger.warning(
                "Only %d total epochs — too few for meaningful "
                "cross-validation. Results will be unreliable.",
                X.shape[0],
            )

        # Compute per-class counts (use unique+counts for safety with
        # non-contiguous label integers)
        _, class_counts = np.unique(y, return_counts=True)
        min_per_class = int(class_counts.min())
        max_per_class = int(class_counts.max())

        # Ensure we don't request more folds than the smallest class can support
        if n_splits > min_per_class:
            n_splits = max(2, min_per_class)
            logger.warning(
                "Reduced n_splits to %d (limited by smallest class size %d).",
                n_splits,
                min_per_class,
            )

        # Guard: severely imbalanced classes
        if min_per_class > 0 and max_per_class > 5 * min_per_class:
            logger.warning(
                "Severe class imbalance: per-class counts = %s. "
                "CV accuracy may be misleading. Consider collecting "
                "more data for under-represented classes.",
                dict(zip(np.unique(y).tolist(), class_counts.tolist())),
            )

        # Use the sklearn pipeline if available (handles CSP refitting)
        estimator = getattr(classifier, "_pipeline", classifier)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        logger.info(
            "Running %d-fold stratified cross-validation on %d epochs...",
            n_splits,
            X.shape[0],
        )

        scores = cross_val_score(
            estimator, X, y, cv=cv, scoring="accuracy"
        )

        mean_acc = float(np.mean(scores))
        std_acc = float(np.std(scores))

        result = {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "scores": scores.tolist(),
            "chance_level": chance_level,
        }

        logger.info(
            "Cross-validation: %.2f%% +/- %.2f%% (chance=%.2f%%).",
            mean_acc * 100,
            std_acc * 100,
            chance_level * 100,
        )

        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        classifier: "BaseClassifier",
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """Evaluate a fitted classifier on held-out test data.

        Args:
            classifier: A **fitted** classifier implementing
                :class:`BaseClassifier`.
            X_test: Test epochs, shape
                ``(n_trials, n_channels, n_samples)``.
            y_test: True integer class labels, shape ``(n_trials,)``.

        Returns:
            Dictionary with keys:

            - ``'accuracy'`` -- overall classification accuracy.
            - ``'confusion_matrix'`` -- 2-D NumPy array, shape
              ``(n_classes, n_classes)``.
            - ``'classification_report'`` -- formatted string report
              from :func:`sklearn.metrics.classification_report`.
            - ``'cohen_kappa'`` -- Cohen's kappa coefficient, which
              accounts for chance agreement.
        """
        if X_test.shape[0] == 0:
            raise ValueError("Cannot evaluate on zero epochs.")

        y_pred = classifier.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        kappa = float(cohen_kappa_score(y_test, y_pred))

        result = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "classification_report": report,
            "cohen_kappa": kappa,
        }

        logger.info(
            "Evaluation: accuracy=%.2f%%, kappa=%.3f.",
            acc * 100,
            kappa,
        )
        logger.debug("Confusion matrix:\n%s", cm)
        logger.debug("Classification report:\n%s", report)

        return result

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ModelTrainer(bp=[{self.bp_low}, {self.bp_high}], "
            f"artifact_thresh={self.artifact_threshold_uv}, "
            f"epoch=[{self.tmin}, {self.tmax}])"
        )
