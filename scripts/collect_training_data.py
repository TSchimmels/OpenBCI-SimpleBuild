#!/usr/bin/env python3
"""Run the Graz calibration paradigm and save training data.

Connects to the configured EEG board, presents the visual cueing
protocol (arrows + fixation cross + beep), records time-locked event
markers alongside the continuous EEG stream, and saves everything
to a timestamped .npz file under data/raw/.

Usage:
    python scripts/collect_training_data.py
    python scripts/collect_training_data.py --config config/settings.yaml
    python scripts/collect_training_data.py --output-dir data/raw --debug
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path hack: allow running from the repo root or the scripts/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.acquisition.board import BoardManager
from src.training.recorder import DataRecorder
from src.training.paradigm import GrazParadigm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Graz motor imagery paradigm and record training data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to settings.yaml (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save the recording. "
            "Default: value from config paths.raw_data_dir (data/raw)."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run the paradigm in a windowed display (not fullscreen).",
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
    logger = logging.getLogger("collect_training_data")

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Resolve output directory
    paths_cfg = config.get("paths", {})
    output_dir = args.output_dir or paths_cfg.get("raw_data_dir", "data/raw")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Create BoardManager, connect
    # ------------------------------------------------------------------
    logger.info("Initializing board...")
    board = BoardManager(config)
    board.connect()

    sf = board.get_sampling_rate()
    n_eeg = len(board.get_eeg_channels())
    logger.info("Board ready: %d Hz, %d EEG channels, synthetic=%s",
                sf, n_eeg, board.is_synthetic())

    try:
        # ------------------------------------------------------------------
        # 3. Create DataRecorder
        # ------------------------------------------------------------------
        recorder = DataRecorder(board)

        # ------------------------------------------------------------------
        # 4. Create GrazParadigm
        # ------------------------------------------------------------------
        paradigm = GrazParadigm(config)
        if args.debug:
            paradigm.debug = True

        train_cfg = config.get("training", {})
        n_classes = train_cfg.get("n_classes", 2)
        classes = train_cfg.get("classes", ["left_hand", "right_hand"])
        trials_per_class = train_cfg.get("n_trials_per_class", 40)
        total_trials = trials_per_class * n_classes

        logger.info(
            "Paradigm configured: %d classes %s, %d trials/class, %d total trials.",
            n_classes, classes, trials_per_class, total_trials,
        )

        # ------------------------------------------------------------------
        # 5. Run paradigm
        # ------------------------------------------------------------------
        logger.info("Starting calibration paradigm...")
        print("\n" + "=" * 60)
        print("EEG CURSOR CALIBRATION")
        print("=" * 60)
        print(f"  Classes:        {classes}")
        print(f"  Trials/class:   {trials_per_class}")
        print(f"  Total trials:   {total_trials}")
        print(f"  Press ESC to abort at any time.")
        print("=" * 60 + "\n")

        recorder.start()
        start_time = time.time()

        paradigm.run(recorder)

        elapsed = time.time() - start_time

        # ------------------------------------------------------------------
        # 6. Stop recording and save data
        # ------------------------------------------------------------------
        raw_data, events = recorder.stop()
        n_events = len(events)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.npz"
        save_path = output_dir / filename

        recorder._events = events  # Restore events for save()
        # Save manually since stop() already pulled the data
        import json
        import numpy as np
        save_path.parent.mkdir(parents=True, exist_ok=True)
        events_json = json.dumps(events)
        np.savez(str(save_path), data=raw_data, events_json=events_json)

        logger.info("Session saved to %s", save_path)

        # ------------------------------------------------------------------
        # 7. Print summary
        # ------------------------------------------------------------------
        n_samples = raw_data.shape[1] if raw_data.ndim == 2 else 0
        duration_data = n_samples / sf if sf > 0 else 0

        # Count events per class
        from collections import Counter
        event_counts = Counter(ev["label"] for ev in events)

        print("\n" + "=" * 60)
        print("RECORDING COMPLETE")
        print("=" * 60)
        print(f"  Session duration:  {elapsed:.1f} s")
        print(f"  Data duration:     {duration_data:.1f} s ({n_samples} samples)")
        print(f"  Total events:      {n_events}")
        for cls_name, count in sorted(event_counts.items()):
            print(f"    {cls_name:20s}: {count} trials")
        print(f"  Saved to:          {save_path}")
        print("=" * 60)

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    except Exception:
        logger.exception("An error occurred during data collection.")
        raise
    finally:
        # ------------------------------------------------------------------
        # 8. Disconnect
        # ------------------------------------------------------------------
        logger.info("Disconnecting board...")
        board.disconnect()
        logger.info("Done.")


if __name__ == "__main__":
    main()
