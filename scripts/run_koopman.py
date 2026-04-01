#!/usr/bin/env python3
"""Run Koopman spectral decomposition on recorded EEG data.

Discovers the subject's actual oscillatory modes (including their
true mu rhythm frequency) using Dynamic Mode Decomposition.

Usage:
    python scripts/run_koopman.py data/raw/session_20260330.npz
    python scripts/run_koopman.py data/raw/session.npz --n-modes 10 --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import load_config
from src.preprocessing.filters import bandpass_filter, common_average_reference
from src.analysis.koopman_decomposition import KoopmanEEGDecomposition


def main():
    parser = argparse.ArgumentParser(description="Koopman spectral analysis.")
    parser.add_argument("path", type=str, help="Path to .npz recording.")
    parser.add_argument("--n-modes", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    path = Path(args.path)
    archive = np.load(str(path), allow_pickle=False)

    sf = int(archive.get("sf", 125))  # Cyton+Daisy default

    if "epochs" in archive:
        data = archive["epochs"].mean(axis=0)  # Average across trials
    elif "data" in archive:
        eeg_ch = archive.get("eeg_channels", np.arange(16)).tolist()
        data = archive["data"][eeg_ch, :]
    else:
        print("Unknown format")
        sys.exit(1)

    # Preprocess
    preproc = config.get("preprocessing", {})
    data = bandpass_filter(data, sf=sf, low=1.0, high=min(40.0, sf / 2 - 1), causal=False)
    data = common_average_reference(data)

    # Run Koopman
    n_ch = data.shape[0]
    koopman = KoopmanEEGDecomposition(n_channels=n_ch, sf=sf, n_modes=args.n_modes)
    koopman.fit(data)

    modes = koopman.get_modes()
    mu_band = koopman.get_subject_mu_band()

    print(f"\n{'=' * 60}")
    print(f"KOOPMAN SPECTRAL DECOMPOSITION")
    print(f"{'=' * 60}")
    print(f"  File: {path.name}")
    print(f"  Channels: {n_ch}, Samples: {data.shape[1]}, SF: {sf} Hz")
    print(f"\n  Discovered modes (sorted by amplitude):")
    print(f"  {'#':>3} {'Freq (Hz)':>10} {'Growth':>10} {'Amplitude':>10}")
    print(f"  {'-' * 40}")
    for i, m in enumerate(modes):
        print(f"  {i + 1:>3} {m['frequency_hz']:>10.2f} {m['growth_rate']:>10.4f} {m['amplitude']:>10.4f}")

    if mu_band:
        # mu_band is (center_freq, bandwidth), not (low, high)
        center, bw = mu_band
        print(f"\n  Subject mu band: {center - bw:.1f} - {center + bw:.1f} Hz (center={center:.1f}, bw={bw:.1f})")
        print(f"  (Config default: 8.0 - 12.0 Hz)")
        if center - bw < 7.0 or center + bw > 13.0:
            print(f"  ** Consider updating mi_bandpass_low/high in settings.yaml **")
    else:
        print(f"\n  No clear mu-band mode found.")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            freqs = [m["frequency_hz"] for m in modes]
            amps = [m["amplitude"] for m in modes]
            ax.stem(freqs, amps)
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Mode Amplitude")
            ax.set_title("Koopman Spectral Modes")
            if mu_band:
                ax.axvspan(mu_band[0], mu_band[1], alpha=0.2, color="cyan", label="Mu band")
                ax.legend()
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("\n  matplotlib not available for plotting.")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
