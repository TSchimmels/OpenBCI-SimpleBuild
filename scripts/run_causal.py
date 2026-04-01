#!/usr/bin/env python3
"""Run causal channel discovery on recorded EEG data.

Discovers per-class causal structure between EEG channels to identify
which channels are most important for each motor imagery class.

Usage:
    python scripts/run_causal.py data/raw/session_20260330.npz
    python scripts/run_causal.py data/raw/session.npz --top-k 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import load_config
from src.training.recorder import DataRecorder
from src.training.trainer import ModelTrainer
from src.analysis.causal_channels import CausalChannelDiscovery
from src.analysis.topography import CHANNEL_NAMES_16


def main():
    parser = argparse.ArgumentParser(description="Causal channel discovery.")
    parser.add_argument("path", type=str, help="Path to .npz recording.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    path = Path(args.path)

    # Load data
    raw_data, events, metadata = DataRecorder.load(str(path))
    from src.acquisition.board import BoardManager
    board = BoardManager(config)
    sf = board.get_sampling_rate()
    eeg_channels = board.get_eeg_channels()
    eeg_channels = [ch for ch in eeg_channels if ch < raw_data.shape[0]]
    n_ch = len(eeg_channels)

    # Prepare epochs
    trainer = ModelTrainer(config)
    epochs, labels, label_map = trainer.prepare_data(raw_data, events, sf, eeg_channels)

    if epochs.shape[0] == 0:
        print("No clean epochs. Cannot run causal discovery.")
        sys.exit(1)

    # Invert label map
    inv_map = {v: k for k, v in label_map.items()}
    class_names = [inv_map[i] for i in range(len(label_map))]
    ch_names = CHANNEL_NAMES_16[:n_ch]

    # Run causal discovery
    discovery = CausalChannelDiscovery(
        n_channels=n_ch, sf=sf, channel_names=ch_names, class_names=class_names,
    )
    results = discovery.discover(epochs, labels)

    print(f"\n{'=' * 60}")
    print(f"CAUSAL CHANNEL DISCOVERY")
    print(f"{'=' * 60}")
    print(f"  File: {path.name}")
    print(f"  Epochs: {epochs.shape[0]}, Channels: {n_ch}")
    print(f"  Classes: {class_names}")

    for cls_name in class_names:
        print(f"\n  --- {cls_name} ---")
        top_ch = discovery.get_important_channels(cls_name, top_k=args.top_k)
        hubs = discovery.get_hub_channels(cls_name, top_k=3)
        pairs = discovery.get_channel_pairs(cls_name, top_k=5)

        print(f"  Top channels: {[ch_names[i] for i in top_ch]}")
        print(f"  Hub channels (most outgoing): {[ch_names[i] for i in hubs]}")
        print(f"  Strongest causal pairs:")
        for src, tgt in pairs:
            print(f"    {ch_names[src]} -> {ch_names[tgt]}")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
