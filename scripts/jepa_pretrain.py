#!/usr/bin/env python3
"""JEPA Self-Supervised Pre-Training for EEG.

Connects to the board, collects ~2 minutes of unlabeled EEG, then runs
JEPAPretrainer to learn EEG structure without task labels. Saves the
encoder to models/jepa_encoder.pkl.

Usage:
    python scripts/jepa_pretrain.py [--duration 120] [--verbose]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JEPA self-supervised pre-training on unlabeled EEG"
    )
    parser.add_argument(
        "--duration", type=int, default=120,
        help="Duration of unlabeled EEG collection in seconds (default: 120)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of pre-training epochs (default: 50)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for pre-training (default: 16)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config()
    project_dir = Path(__file__).parent.parent

    # --- Connect to board and collect unlabeled EEG ---
    logger.info("JEPA Self-Supervised Pre-Training")
    logger.info("Connecting to board...")

    from src.acquisition.board import BoardManager

    board = BoardManager(config)
    board.connect()
    sf = board.get_sampling_rate()
    eeg_ch = board.get_eeg_channels()
    logger.info("Board: %dHz, %d channels", sf, len(eeg_ch))
    logger.info(
        "Collecting %d seconds of unlabeled EEG for pre-training...",
        args.duration,
    )

    # Flush any stale data
    board.get_board_data()

    # Collect unlabeled data
    time.sleep(args.duration)
    raw = board.get_board_data()
    eeg = raw[eeg_ch, :]
    board.disconnect()

    logger.info(
        "Collected %d samples (%.0f seconds)",
        eeg.shape[1],
        eeg.shape[1] / sf,
    )

    # --- Pre-train ---
    from src.training.pretrain import JEPAPretrainer

    win_len = int(2.5 * sf)
    pt = JEPAPretrainer(n_channels=len(eeg_ch), n_samples=win_len, sf=sf)

    # Cut into overlapping windows (50% overlap)
    step = win_len // 2
    windows = []
    for i in range(0, eeg.shape[1] - win_len, step):
        windows.append(eeg[:, i : i + win_len])
    X = np.stack(windows)
    logger.info("Training on %d windows...", X.shape[0])

    result = pt.pretrain(X, n_epochs=args.epochs, batch_size=args.batch_size)
    final_loss = result["loss_history"][-1]
    logger.info("Pre-training complete. Final loss: %.4f", final_loss)

    # --- Save encoder ---
    import joblib

    out_path = project_dir / "models" / "jepa_encoder.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pt, str(out_path))
    logger.info("Encoder saved to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
