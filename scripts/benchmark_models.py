#!/usr/bin/env python3
"""Benchmark all classifier types on recorded data.

Trains and cross-validates each available classifier on the same dataset,
producing a comparison table of accuracy, latency, and robustness metrics.

Usage:
    python scripts/benchmark_models.py --data-path data/raw/session_20260330.npz
    python scripts/benchmark_models.py --data-path data/raw/session.npz --folds 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import load_config
from src.acquisition.board import BoardManager
from src.training.recorder import DataRecorder
from src.training.trainer import ModelTrainer
from src.classification.pipeline import ClassifierFactory


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark all classifier types.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config = load_config(args.config)
    data_path = Path(args.data_path)

    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    # Load and prepare data
    raw_data, events, metadata = DataRecorder.load(str(data_path))
    board = BoardManager(config)
    sf = board.get_sampling_rate()
    eeg_channels = board.get_eeg_channels()
    eeg_channels = [ch for ch in eeg_channels if ch < raw_data.shape[0]]

    trainer = ModelTrainer(config)
    epochs, labels, label_map = trainer.prepare_data(raw_data, events, sf, eeg_channels)

    if epochs.shape[0] == 0:
        print("No clean epochs. Cannot benchmark.")
        sys.exit(1)

    n_classes = len(np.unique(labels))
    chance = 1.0 / n_classes

    print(f"\n{'=' * 70}")
    print(f"MODEL BENCHMARK")
    print(f"{'=' * 70}")
    print(f"  Data:    {data_path.name}")
    print(f"  Epochs:  {epochs.shape[0]} ({epochs.shape[1]}ch x {epochs.shape[2]}samp)")
    print(f"  Classes: {n_classes} (chance = {chance * 100:.1f}%)")
    print(f"  Folds:   {args.folds}")
    print(f"{'=' * 70}\n")

    model_types = ["csp_lda", "riemannian"]

    # Check for PyTorch
    try:
        import torch
        model_types.append("eegnet")
    except ImportError:
        pass

    results = []

    for model_type in model_types:
        print(f"  Testing {model_type}...", end="", flush=True)

        config["classification"]["model_type"] = model_type
        clf = ClassifierFactory.create(config)

        # Time the fitting
        t0 = time.monotonic()
        try:
            cv = trainer.cross_validate(clf, epochs, labels, n_splits=args.folds)
            fit_time = time.monotonic() - t0

            # Time single prediction
            t1 = time.monotonic()
            for _ in range(10):
                clf.predict(epochs[:1])
            pred_time = (time.monotonic() - t1) / 10 * 1000  # ms

            results.append({
                "model": model_type,
                "accuracy": cv["mean_accuracy"],
                "std": cv["std_accuracy"],
                "fit_time": fit_time,
                "pred_ms": pred_time,
                "status": "OK",
            })
            print(f" {cv['mean_accuracy'] * 100:.1f}% +/- {cv['std_accuracy'] * 100:.1f}%")

        except Exception as e:
            results.append({
                "model": model_type,
                "accuracy": 0,
                "std": 0,
                "fit_time": 0,
                "pred_ms": 0,
                "status": str(e)[:50],
            })
            print(f" FAILED: {e}")

    # Print results table
    print(f"\n{'=' * 70}")
    print(f"{'Model':<20} {'Accuracy':>10} {'Std':>8} {'CV Time':>10} {'Pred':>8} {'Status':>10}")
    print(f"{'-' * 70}")
    for r in results:
        if r["status"] == "OK":
            print(f"{r['model']:<20} {r['accuracy'] * 100:>9.1f}% {r['std'] * 100:>7.1f}% "
                  f"{r['fit_time']:>9.1f}s {r['pred_ms']:>7.1f}ms {'OK':>10}")
        else:
            print(f"{r['model']:<20} {'—':>10} {'—':>8} {'—':>10} {'—':>8} {'FAIL':>10}")
    print(f"{'-' * 70}")
    print(f"{'Chance level':<20} {chance * 100:>9.1f}%")
    print(f"{'=' * 70}")

    # Best model
    ok_results = [r for r in results if r["status"] == "OK"]
    if ok_results:
        best = max(ok_results, key=lambda r: r["accuracy"])
        print(f"\n  Best: {best['model']} ({best['accuracy'] * 100:.1f}%)")


if __name__ == "__main__":
    main()
