#!/usr/bin/env python3
"""Train a motor imagery classifier on previously recorded EEG data.

Loads a session .npz file (produced by collect_training_data.py),
extracts and filters epochs, rejects artifacts, performs stratified
cross-validation, trains on the full dataset, and saves the fitted
model.

Usage:
    python scripts/train_model.py --data-path data/raw/session_20260325_140000.npz
    python scripts/train_model.py --data-path data/raw/session_*.npz --model-type csp_lda
    python scripts/train_model.py --data-path data/raw/session.npz --n-folds 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path hack: allow running from the repo root or the scripts/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import load_config
from src.training.recorder import DataRecorder
from src.training.trainer import ModelTrainer
from src.classification.pipeline import ClassifierFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a motor imagery classifier on recorded EEG data."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to the .npz recording file (from collect_training_data.py).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to settings.yaml (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["csp_lda", "riemannian", "eegnet", "neural_sde", "adaptive_router"],
        help="Override the model type from config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the trained model (default: from config or models/).",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=10,
        help="Number of cross-validation folds (default: 10).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("train_model")

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Override model type if specified on the command line
    if args.model_type is not None:
        config.setdefault("classification", {})
        config["classification"]["model_type"] = args.model_type
        logger.info("Model type overridden to: %s", args.model_type)

    model_type = config.get("classification", {}).get("model_type", "csp_lda")

    # Resolve output directory
    paths_cfg = config.get("paths", {})
    output_dir = Path(args.output_dir or paths_cfg.get("models_dir", "models"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Load recorded data
    # ------------------------------------------------------------------
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error("Data file not found: %s", data_path)
        sys.exit(1)

    # Check for synthetic data flag
    if "SYNTHETIC" in data_path.name:
        logger.warning("")
        logger.warning("=" * 60)
        logger.warning("  WARNING: Training on SYNTHETIC data!")
        logger.warning("  Models trained on synthetic data will NOT work")
        logger.warning("  for real BCI control. This is for pipeline testing only.")
        logger.warning("=" * 60)
        logger.warning("")

    logger.info("Loading recorded data from %s...", data_path)
    raw_data, events, recording_metadata = DataRecorder.load(str(data_path))

    n_samples = raw_data.shape[1] if raw_data.ndim == 2 else 0
    n_events = len(events)
    logger.info("Loaded: %d channels, %d samples, %d events.", raw_data.shape[0], n_samples, n_events)

    if n_events == 0:
        logger.error("No events found in the recording. Cannot train.")
        sys.exit(1)

    # Determine sampling rate and EEG channels from .npz metadata or config.
    # The .npz file saved by collect_training_data.py / DataRecorder.save()
    # includes sf and eeg_channels, so we don't need the board connected.
    board_cfg = config.get("board", {})
    if "sf" in recording_metadata:
        sf = int(recording_metadata["sf"])
    else:
        sf = int(board_cfg.get("sampling_rate_override", 125))
        logger.warning("No sf in recording metadata; using config value %d Hz.", sf)

    if "eeg_channels" in recording_metadata:
        eeg_channels = recording_metadata["eeg_channels"]
    else:
        # Fall back to 0..channel_count-1
        n_ch = board_cfg.get("channel_count", 16)
        eeg_channels = list(range(min(n_ch, raw_data.shape[0])))
        logger.warning("No eeg_channels in metadata; using first %d channels.", len(eeg_channels))

    # Clamp EEG channels to the data dimensions
    max_ch = raw_data.shape[0]
    eeg_channels = [ch for ch in eeg_channels if ch < max_ch]
    if not eeg_channels:
        logger.error("No valid EEG channels for the recorded data (max index=%d).", max_ch - 1)
        sys.exit(1)
    logger.info("Using sf=%d Hz, %d EEG channels %s.", sf, len(eeg_channels), eeg_channels)

    # ------------------------------------------------------------------
    # 3. Create ModelTrainer, prepare data
    # ------------------------------------------------------------------
    logger.info("Preparing data (epoch extraction, filtering, artifact rejection)...")
    trainer = ModelTrainer(config)
    epochs, labels, label_map = trainer.prepare_data(raw_data, events, sf, eeg_channels)

    if epochs.shape[0] == 0:
        logger.error("No clean epochs after preparation. Cannot train.")
        sys.exit(1)

    n_clean = epochs.shape[0]
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    logger.info(
        "Prepared %d clean epochs: %s",
        n_clean,
        dict(zip(unique_labels.tolist(), label_counts.tolist())),
    )

    # ------------------------------------------------------------------
    # 4. Create classifier
    # ------------------------------------------------------------------
    logger.info("Creating classifier: %s...", model_type)
    classifier = ClassifierFactory.create(config)
    logger.info("Classifier: %r", classifier)

    # ------------------------------------------------------------------
    # 5. Cross-validation
    # ------------------------------------------------------------------
    n_folds = min(args.n_folds, n_clean)
    if n_folds < 2:
        logger.warning("Too few epochs (%d) for cross-validation. Skipping.", n_clean)
        cv_results = None
    else:
        # Ensure at least 2 samples per class per fold
        min_class_count = int(label_counts.min())
        if n_folds > min_class_count:
            n_folds = max(2, min_class_count)
            logger.warning(
                "Reducing folds to %d (smallest class has %d samples).",
                n_folds, min_class_count,
            )

        logger.info("Running %d-fold stratified cross-validation...", n_folds)
        cv_results = trainer.cross_validate(classifier, epochs, labels, n_splits=n_folds)

        print("\n" + "-" * 50)
        print("CROSS-VALIDATION RESULTS")
        print("-" * 50)
        print(f"  Folds:            {n_folds}")
        print(f"  Mean accuracy:    {cv_results['mean_accuracy'] * 100:.2f}%")
        print(f"  Std accuracy:     {cv_results['std_accuracy'] * 100:.2f}%")
        print(f"  Chance level:     {cv_results['chance_level'] * 100:.2f}%")
        print(f"  Per-fold scores:  {[f'{s:.3f}' for s in cv_results['scores']]}")
        print("-" * 50)

    # ------------------------------------------------------------------
    # 6. Train on full dataset
    # ------------------------------------------------------------------
    logger.info("Training on full dataset (%d epochs)...", n_clean)
    trained_clf, train_metrics = trainer.train(classifier, epochs, labels)

    logger.info("Training accuracy: %.2f%%", train_metrics["accuracy"] * 100)

    # ------------------------------------------------------------------
    # 7. Save model
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_type}_{timestamp}.pkl"
    model_path = output_dir / model_filename

    trained_clf.save(str(model_path))
    logger.info("Model saved to %s", model_path)

    # Save label mapping alongside the model so runtime can use the same encoding
    import json
    label_map_path = model_path.with_suffix('.labels.json')
    with open(str(label_map_path), 'w') as f:
        json.dump(label_map, f)
    logger.info("Label mapping saved to %s", label_map_path)

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Data file:         {data_path}")
    print(f"  Model type:        {model_type}")
    print(f"  Clean epochs:      {n_clean}")
    print(f"  Label distribution:")
    for lbl, cnt in zip(unique_labels.tolist(), label_counts.tolist()):
        print(f"    Class {lbl}: {cnt} epochs")
    print(f"  Training accuracy: {train_metrics['accuracy'] * 100:.2f}%")
    if cv_results is not None:
        print(f"  CV accuracy:       {cv_results['mean_accuracy'] * 100:.2f}% "
              f"+/- {cv_results['std_accuracy'] * 100:.2f}%")
        print(f"  Chance level:      {cv_results['chance_level'] * 100:.2f}%")
    print(f"  Model saved to:    {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
