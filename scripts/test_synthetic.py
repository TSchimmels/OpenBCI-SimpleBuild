#!/usr/bin/env python3
"""End-to-end pipeline test with the BrainFlow SYNTHETIC_BOARD.

Verifies every stage of the Mental Mouse pipeline without requiring
real EEG hardware:

    1. Board connection (synthetic, board_id=-1)
    2. Data acquisition (5 seconds of streaming)
    3. Preprocessing (bandpass, notch, CAR)
    4. Chaos / nonlinear feature extraction
    5. Band power feature extraction

Usage:
    python scripts/test_synthetic.py
    python scripts/test_synthetic.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path hack: allow running from the repo root or the scripts/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import load_config
from src.acquisition.board import BoardManager
from src.preprocessing.filters import bandpass_filter, notch_filter, common_average_reference
from src.features.chaos import ChaosFeatureExtractor
from src.features.bandpower import BandPowerExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the full BCI pipeline on synthetic EEG data."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to settings.yaml (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Seconds of data to stream before processing (default: 5).",
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
    logger = logging.getLogger("test_synthetic")

    # ------------------------------------------------------------------
    # 1. Load config, force synthetic board
    # ------------------------------------------------------------------
    logger.info("Loading configuration...")
    config = load_config(args.config)

    # Override board settings to guarantee synthetic mode
    config.setdefault("board", {})
    config["board"]["board_id"] = -1
    config["board"]["serial_port"] = ""
    logger.info("Board forced to SYNTHETIC_BOARD (board_id=-1).")

    # ------------------------------------------------------------------
    # 2. Create BoardManager, connect
    # ------------------------------------------------------------------
    board = BoardManager(config)
    logger.info("Connecting to synthetic board...")
    board.connect()

    sf = board.get_sampling_rate()
    eeg_channels = board.get_eeg_channels()
    logger.info("Sampling rate: %d Hz", sf)
    logger.info("EEG channels: %s (%d total)", eeg_channels, len(eeg_channels))

    # ------------------------------------------------------------------
    # 3. Stream data for the requested duration
    # ------------------------------------------------------------------
    stream_seconds = args.duration
    logger.info("Streaming for %.1f seconds...", stream_seconds)
    time.sleep(stream_seconds)

    n_samples_expected = int(sf * stream_seconds)
    raw = board.get_data(n_samples_expected)
    logger.info("Raw data shape: %s (channels x samples)", raw.shape)

    # Extract only EEG rows
    eeg_data = raw[eeg_channels, :]
    logger.info(
        "EEG data shape: %s (%d channels x %d samples)",
        eeg_data.shape,
        eeg_data.shape[0],
        eeg_data.shape[1],
    )

    # ------------------------------------------------------------------
    # 4. Preprocessing
    # ------------------------------------------------------------------
    preproc_cfg = config.get("preprocessing", {})
    bp_low = preproc_cfg.get("bandpass_low", 1.0)
    bp_high = preproc_cfg.get("bandpass_high", 40.0)
    bp_order = preproc_cfg.get("bandpass_order", 4)
    notch_freq = preproc_cfg.get("notch_freq", 60.0)
    notch_quality = preproc_cfg.get("notch_quality", 30.0)

    # Clamp high-frequency cutoff to stay below Nyquist
    nyquist = sf / 2.0
    if bp_high >= nyquist:
        bp_high = nyquist - 1.0
        logger.warning("Clamped bandpass_high to %.1f Hz (Nyquist=%.1f Hz).", bp_high, nyquist)

    logger.info("Applying bandpass filter: [%.1f, %.1f] Hz, order %d...", bp_low, bp_high, bp_order)
    filtered = bandpass_filter(eeg_data, sf=sf, low=bp_low, high=bp_high, order=bp_order, causal=False)

    if notch_freq < nyquist:
        logger.info("Applying notch filter at %.1f Hz (Q=%.1f)...", notch_freq, notch_quality)
        filtered = notch_filter(filtered, sf=sf, freq=notch_freq, quality=notch_quality, causal=False)
    else:
        logger.warning("Skipping notch filter: %.1f Hz >= Nyquist %.1f Hz.", notch_freq, nyquist)

    logger.info("Applying common average reference (CAR)...")
    filtered = common_average_reference(filtered)

    logger.info("Preprocessed data shape: %s", filtered.shape)
    logger.info(
        "Preprocessed stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        float(np.mean(filtered)),
        float(np.std(filtered)),
        float(np.min(filtered)),
        float(np.max(filtered)),
    )

    # ------------------------------------------------------------------
    # 5. Chaos / nonlinear features
    # ------------------------------------------------------------------
    feat_cfg = config.get("features", {})
    chaos_features = feat_cfg.get("chaos_features", ["hjorth", "perm_entropy", "spectral_entropy"])
    chaos_channels = feat_cfg.get("chaos_channels", [0, 1, 2])

    # Clamp channel indices to available channels
    max_ch = filtered.shape[0]
    chaos_channels = [c for c in chaos_channels if c < max_ch]
    if not chaos_channels:
        chaos_channels = list(range(min(3, max_ch)))
    logger.info("Chaos feature channels (clamped): %s", chaos_channels)

    logger.info("Extracting chaos features: %s...", chaos_features)
    chaos_extractor = ChaosFeatureExtractor(features=chaos_features, sf=sf)

    chaos_feats = chaos_extractor.extract_multi_channel(filtered, channel_indices=chaos_channels)
    chaos_names = chaos_extractor.get_feature_names(channel_indices=chaos_channels)

    logger.info("Chaos features (%d values):", len(chaos_feats))
    for name, val in zip(chaos_names, chaos_feats):
        logger.info("  %-35s = %.6f", name, val)

    # ------------------------------------------------------------------
    # 6. Band power features
    # ------------------------------------------------------------------
    bp_bands = feat_cfg.get("bandpower_bands", {"mu": [8, 12], "beta": [13, 30]})
    bp_channels = feat_cfg.get("bandpower_channels", [0, 1, 2])
    bp_channels = [c for c in bp_channels if c < max_ch]
    if not bp_channels:
        bp_channels = list(range(min(3, max_ch)))
    logger.info("Band power channels (clamped): %s", bp_channels)

    logger.info("Extracting band power features: bands=%s...", bp_bands)
    bp_extractor = BandPowerExtractor(bands=bp_bands, sf=sf)

    bp_feats = bp_extractor.extract(filtered, channel_indices=bp_channels)
    bp_names = bp_extractor.get_feature_names(channel_indices=bp_channels)

    logger.info("Band power features (%d values):", len(bp_feats))
    for name, val in zip(bp_names, bp_feats):
        logger.info("  %-35s = %.6f", name, val)

    # ------------------------------------------------------------------
    # 7. Disconnect
    # ------------------------------------------------------------------
    logger.info("Disconnecting board...")
    board.disconnect()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SYNTHETIC PIPELINE TEST COMPLETE")
    print("=" * 60)
    print(f"  Board:              SYNTHETIC_BOARD (id=-1)")
    print(f"  Sampling rate:      {sf} Hz")
    print(f"  EEG channels:       {len(eeg_channels)}")
    print(f"  Stream duration:    {stream_seconds:.1f} s")
    print(f"  Samples acquired:   {eeg_data.shape[1]}")
    print(f"  Chaos features:     {len(chaos_feats)} values")
    print(f"  Band power features:{len(bp_feats)} values")
    print(f"  All stages:         PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
