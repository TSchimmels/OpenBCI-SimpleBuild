#!/usr/bin/env python3
"""Quick analysis of a recorded EEG session.

Loads a .npz recording and prints summary statistics, class distribution,
signal quality metrics, and optionally plots key visualizations.

Usage:
    python scripts/analyze_session.py data/raw/session_20260330.npz
    python scripts/analyze_session.py data/raw/session_20260330.npz --plot
    python scripts/analyze_session.py data/raw/erp_session_20260330.npz --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import load_config
from src.preprocessing.filters import bandpass_filter, common_average_reference


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a recorded EEG session.")
    parser.add_argument("path", type=str, help="Path to .npz recording file.")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plots.")
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    config = load_config(args.config)
    archive = np.load(str(path), allow_pickle=False)

    print(f"\n{'=' * 60}")
    print(f"SESSION ANALYSIS: {path.name}")
    print(f"{'=' * 60}")
    print(f"  File: {path}")
    print(f"  Keys: {list(archive.keys())}")

    # Handle both formats
    if "epochs" in archive:
        epochs = archive["epochs"]
        labels = archive["labels"]
        sf = int(archive.get("sf", 125))
        print(f"\n  Format: Pre-epoched")
        print(f"  Epochs: {epochs.shape[0]}")
        print(f"  Channels: {epochs.shape[1]}")
        print(f"  Samples/epoch: {epochs.shape[2]}")
        print(f"  Sampling rate: {sf} Hz")
        print(f"  Epoch duration: {epochs.shape[2] / sf:.1f} s")

        class_counts = Counter(labels)
        print(f"\n  Class distribution:")
        for cls, cnt in sorted(class_counts.items()):
            print(f"    {cls}: {cnt} epochs")

        # Signal quality per channel
        print(f"\n  Signal quality (mean amplitude per channel):")
        for ch in range(min(epochs.shape[1], 8)):
            rms = np.sqrt(np.mean(epochs[:, ch, :] ** 2))
            ptp = np.mean(np.ptp(epochs[:, ch, :], axis=1))
            print(f"    Ch {ch:2d}: RMS={rms:7.1f} uV, PtP={ptp:7.1f} uV")

    elif "data" in archive:
        data = archive["data"]
        events = json.loads(str(archive["events_json"]))
        sf = int(archive.get("sf", 125))
        print(f"\n  Format: Continuous recording")
        print(f"  Channels: {data.shape[0]}")
        print(f"  Samples: {data.shape[1]}")
        print(f"  Duration: {data.shape[1] / sf:.1f} s")
        print(f"  Sampling rate: {sf} Hz")
        print(f"  Events: {len(events)}")

        if events:
            event_counts = Counter(ev["label"] for ev in events)
            print(f"\n  Event distribution:")
            for cls, cnt in sorted(event_counts.items()):
                print(f"    {cls}: {cnt}")

        # Signal quality
        print(f"\n  Signal quality (first 16 channels):")
        for ch in range(min(data.shape[0], 16)):
            rms = np.sqrt(np.mean(data[ch] ** 2))
            std = data[ch].std()
            print(f"    Ch {ch:2d}: RMS={rms:8.1f}, STD={std:8.1f}")
    else:
        print(f"  Unknown format. Keys: {list(archive.keys())}")
        sys.exit(1)

    # Plotting
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            if "epochs" in archive:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                fig.suptitle(f"Session: {path.name}", fontsize=14)

                # 1. Mean ERP per class
                ax = axes[0, 0]
                unique_labels = sorted(set(labels))
                colors = ["#888", "#2196F3", "#F44336", "#4CAF50", "#FF9800"]
                for i, lbl in enumerate(unique_labels):
                    mask = labels == lbl
                    mean_erp = epochs[mask].mean(axis=0)
                    ax.plot(mean_erp[0], label=str(lbl),
                            color=colors[i % len(colors)])
                ax.set_title("Mean ERP (Ch 0)")
                ax.set_xlabel("Samples")
                ax.legend(fontsize=8)

                # 2. RMS per channel
                ax = axes[0, 1]
                rms_vals = [np.sqrt(np.mean(epochs[:, ch, :] ** 2))
                            for ch in range(epochs.shape[1])]
                ax.bar(range(len(rms_vals)), rms_vals)
                ax.set_title("RMS Amplitude per Channel")
                ax.set_xlabel("Channel")

                # 3. Class balance
                ax = axes[1, 0]
                counts = [class_counts.get(lbl, 0) for lbl in unique_labels]
                ax.bar([str(l) for l in unique_labels], counts,
                       color=colors[:len(unique_labels)])
                ax.set_title("Class Distribution")

                # 4. Single trial overlay
                ax = axes[1, 1]
                for i in range(min(20, epochs.shape[0])):
                    ax.plot(epochs[i, 0, :], alpha=0.2, color="gray")
                ax.plot(epochs[:, 0, :].mean(axis=0), color="red", linewidth=2,
                        label="Mean")
                ax.set_title("Single Trial Overlay (Ch 0)")
                ax.legend()

                plt.tight_layout()
                plt.show()

            elif "data" in archive:
                fig, axes = plt.subplots(2, 1, figsize=(14, 8))
                fig.suptitle(f"Session: {path.name}", fontsize=14)

                # 1. Raw signals (first 8 channels, last 10s)
                ax = axes[0]
                n_show = min(sf * 10, data.shape[1])
                t = np.arange(n_show) / sf
                for ch in range(min(8, data.shape[0])):
                    offset = ch * 200
                    ax.plot(t, data[ch, -n_show:] + offset, linewidth=0.5)
                ax.set_title("Raw EEG (last 10s, 8 channels)")
                ax.set_xlabel("Time (s)")

                # 2. Power spectrum
                ax = axes[1]
                from scipy.signal import welch
                for ch in range(min(4, data.shape[0])):
                    f, psd = welch(data[ch], fs=sf, nperseg=min(512, data.shape[1]))
                    ax.semilogy(f, psd, label=f"Ch {ch}")
                ax.set_title("Power Spectral Density")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_xlim(0, 50)
                ax.legend()

                plt.tight_layout()
                plt.show()

        except ImportError:
            print("\n  matplotlib not available. Install with: pip install matplotlib")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    main()
