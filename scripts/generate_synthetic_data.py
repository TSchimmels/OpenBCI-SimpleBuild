#!/usr/bin/env python3
"""Generate synthetic labeled MI data for headless pipeline testing.

Creates a .npz file with simulated 5-class motor imagery EEG data that
can be used to test the full train -> classify -> cursor pipeline without
requiring a display (pygame) or real EEG hardware.

The synthetic data has class-specific patterns injected into motor cortex
channels (C3, C4, Cz) so classifiers can achieve above-chance accuracy
(~40-60% for 5 classes). This is NOT real EEG — it's for pipeline testing
only and must never be confused with real recordings.

Output files are prefixed with 'SYNTHETIC_' and contain a metadata flag
`synthetic=True` to prevent accidental use as real data.

Usage:
    python scripts/generate_synthetic_data.py
    python scripts/generate_synthetic_data.py --output-dir data/demo --n-trials 60
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic labeled MI data for pipeline testing."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/demo",
        help="Directory to save (default: data/demo — NOT data/raw).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=40,
        help="Trials per class (default: 40).",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to settings.yaml (default: config/settings.yaml).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def generate_mi_epoch(
    class_name: str,
    n_channels: int,
    n_samples: int,
    sf: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Generate one synthetic MI epoch with class-specific patterns.

    Injects ERD (event-related desynchronization) patterns:
    - left_hand: mu suppression at C4 (channel 1), beta increase at C3 (channel 0)
    - right_hand: mu suppression at C3 (channel 0), beta increase at C4 (channel 1)
    - feet: mu suppression at Cz (channel 2), bilateral beta increase
    - tongue: bilateral mu increase (no suppression)
    - rest: baseline noise only
    """
    t = np.arange(n_samples) / sf

    # Base: pink noise + alpha oscillation
    epoch = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        # Pink noise (1/f spectrum)
        freqs = np.fft.rfftfreq(n_samples, 1.0 / sf)
        freqs[0] = 1.0  # avoid div by zero
        spectrum = rng.randn(len(freqs)) + 1j * rng.randn(len(freqs))
        spectrum /= np.sqrt(freqs)
        epoch[ch] = np.fft.irfft(spectrum, n=n_samples) * 20.0

        # Background alpha (10 Hz) on all channels
        epoch[ch] += 15.0 * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))

    # Class-specific patterns (motor cortex channels: 0=C3, 1=C4, 2=Cz)
    mu_freq = 10.0 + rng.uniform(-1, 1)  # subject-specific mu
    beta_freq = 20.0 + rng.uniform(-2, 2)

    if class_name == "left_hand":
        # ERD at C4 (contralateral), ERS at C3
        epoch[1] *= 0.5  # mu suppression at C4
        epoch[0] += 10.0 * np.sin(2 * np.pi * beta_freq * t)  # beta ERS at C3
    elif class_name == "right_hand":
        # ERD at C3 (contralateral), ERS at C4
        epoch[0] *= 0.5  # mu suppression at C3
        epoch[1] += 10.0 * np.sin(2 * np.pi * beta_freq * t)  # beta ERS at C4
    elif class_name == "feet":
        # ERD at Cz (midline), bilateral beta
        epoch[2] *= 0.4
        epoch[0] += 8.0 * np.sin(2 * np.pi * beta_freq * t)
        epoch[1] += 8.0 * np.sin(2 * np.pi * beta_freq * t)
    elif class_name == "tongue":
        # Bilateral mu ERS (increase, not suppression)
        epoch[0] += 20.0 * np.sin(2 * np.pi * mu_freq * t)
        epoch[1] += 20.0 * np.sin(2 * np.pi * mu_freq * t)
    # rest: no modification (baseline noise only)

    return epoch


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("generate_synthetic")

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    board_cfg = config.get("board", {})

    classes = train_cfg.get("classes", ["rest", "left_hand", "right_hand", "feet", "tongue"])
    sf = board_cfg.get("sampling_rate_override", None) or 250
    n_channels = board_cfg.get("channel_count", 16)
    window_start = train_cfg.get("classification_window_start", 1.5)
    window_end = train_cfg.get("classification_window_end", 4.0)
    n_samples = int(sf * (window_end - window_start))
    n_trials = args.n_trials

    rng = np.random.RandomState(42)

    logger.info("Generating synthetic MI data:")
    logger.info("  Classes: %s", classes)
    logger.info("  Trials/class: %d", n_trials)
    logger.info("  Channels: %d, Samples: %d, SF: %d Hz", n_channels, n_samples, sf)

    # Build continuous raw data + events (mimics DataRecorder.save format)
    # Each trial: fixation (2s) + cue (1.25s) + imagery (4s) + rest (2s)
    trial_samples = int(sf * (2.0 + 1.25 + 4.0 + 2.0))  # ~9.25s per trial
    total_trials = n_trials * len(classes)

    # Create shuffled trial order
    trial_order = []
    for cls in classes:
        trial_order.extend([cls] * n_trials)
    rng.shuffle(trial_order)

    # Build continuous data stream
    all_data = []
    events = []
    current_sample = 0

    for trial_idx, cls_name in enumerate(trial_order):
        # Fixation period (2s of noise)
        fixation = rng.randn(n_channels, int(sf * 2.0)) * 15.0

        # Cue onset -> imagery window
        cue_offset = int(sf * 2.0)
        imagery_start = int(sf * (2.0 + window_start))

        # Full trial
        trial_data = rng.randn(n_channels, trial_samples) * 15.0

        # Inject MI pattern in the classification window
        epoch = generate_mi_epoch(cls_name, n_channels, n_samples, sf, rng)
        trial_data[:, imagery_start:imagery_start + n_samples] = epoch

        # Record event at cue onset
        events.append({
            "type": "cue",
            "class": cls_name,
            "label": classes.index(cls_name),
            "sample": current_sample + cue_offset,
            "timestamp": (current_sample + cue_offset) / sf,
        })

        all_data.append(trial_data)
        current_sample += trial_samples

    raw_data = np.hstack(all_data)
    logger.info("Raw data shape: %s (%.1f seconds)", raw_data.shape, raw_data.shape[1] / sf)

    # Save in DataRecorder-compatible format
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"SYNTHETIC_demo_{timestamp}.npz"
    save_path = output_dir / filename

    events_json = json.dumps(events)
    eeg_channels = list(range(n_channels))

    np.savez(
        str(save_path),
        data=raw_data,
        events_json=events_json,
        sf=np.array(sf),
        eeg_channels=np.array(eeg_channels),
        synthetic=np.array(True),  # FLAG: this is synthetic data
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("SYNTHETIC DATA GENERATED")
    logger.info("=" * 60)
    logger.info("  File: %s", save_path)
    logger.info("  Trials: %d (%d per class x %d classes)", total_trials, n_trials, len(classes))
    logger.info("  Duration: %.0f seconds", raw_data.shape[1] / sf)
    logger.info("")
    logger.info("  WARNING: This is SYNTHETIC data for pipeline testing only.")
    logger.info("  It contains artificial class-specific patterns, NOT real EEG.")
    logger.info("  Models trained on this data will NOT work for real BCI control.")
    logger.info("")
    logger.info("  To train on this data:")
    logger.info("    python scripts/train_model.py --data-path %s", save_path)
    logger.info("")


if __name__ == "__main__":
    main()
