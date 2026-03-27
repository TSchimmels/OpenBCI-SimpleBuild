#!/usr/bin/env python3
"""ERP Signal Training & Data Collection Tool.

A standalone tool for collecting motor imagery EEG data while providing
real-time visual feedback of the subject's ERPs, ERDS% maps, and signal
quality — WITHOUT requiring model training. Designed to help subjects
learn to produce consistent, detectable motor imagery signals.

Can be used in two modes:
    1. COLLECTION MODE (default): Runs the Graz paradigm, records data,
       and shows real-time ERP/ERDS feedback after each trial.
    2. REVIEW MODE: Loads a previously recorded .npz session and
       displays the offline ERP analysis.

Usage:
    # Live collection with real-time feedback
    python scripts/erp_trainer.py

    # Review a previous recording
    python scripts/erp_trainer.py --review data/raw/session_20260326.npz

    # Collection with specific config
    python scripts/erp_trainer.py --config config/settings.yaml --debug
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyArrowPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib is required: pip install matplotlib")
    sys.exit(1)

from src.config import load_config
from src.acquisition.board import BoardManager
from src.preprocessing.filters import bandpass_filter, common_average_reference
from src.analysis.erp import ERPAccumulator
from src.analysis.time_frequency import ERDSComputer
from src.analysis.topography import TopoMapper, CHANNEL_NAMES_16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ERP Signal Training & Data Collection Tool."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to settings.yaml.",
    )
    parser.add_argument(
        "--review", type=str, default=None,
        help="Path to a .npz recording to review offline (skips live collection).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save recordings (default: data/raw).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=None,
        help="Override trials per class (default: from config).",
    )
    parser.add_argument(
        "--classes", type=str, nargs="+", default=None,
        help="Override class list (e.g., --classes left_hand right_hand rest).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Windowed paradigm display (not fullscreen).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    return parser.parse_args()


# ======================================================================
# ERP Display Engine
# ======================================================================

class ERPDisplay:
    """Real-time ERP visualization dashboard.

    Shows four synchronized panels:
        1. ERP waveforms (C3, C4, Cz) with running average ± std
        2. Time-frequency ERDS% spectrogram
        3. Mu and beta band power timecourses
        4. Scalp topographic map at selected timepoint

    All panels share the same time axis (seconds relative to cue onset).
    """

    def __init__(
        self,
        n_channels: int,
        sf: int,
        epoch_tmin: float = -1.0,
        epoch_tmax: float = 5.0,
        class_names: List[str] = None,
        channel_names: List[str] = None,
        motor_channels: List[int] = None,
    ) -> None:
        self.sf = sf
        self.epoch_tmin = epoch_tmin
        self.epoch_tmax = epoch_tmax
        self.class_names = class_names or ["rest", "left_hand", "right_hand"]
        self.channel_names = channel_names or CHANNEL_NAMES_16[:n_channels]

        # Motor cortex channels to highlight
        # Default: C3(idx 2), C4(idx 3), Cz(idx 14) for 16-ch
        if motor_channels is not None:
            self.motor_channels = motor_channels
        elif n_channels >= 15:
            self.motor_channels = [2, 3, 14]  # C3, C4, Cz
        elif n_channels >= 4:
            self.motor_channels = [0, 1, 2]
        else:
            self.motor_channels = list(range(n_channels))

        n_samples = int((epoch_tmax - epoch_tmin) * sf)
        baseline_samples = int(abs(epoch_tmin) * sf)

        # Analysis engines
        self.erp_accum = ERPAccumulator(
            n_channels=n_channels,
            n_samples=n_samples,
            sf=sf,
            baseline_samples=baseline_samples,
            class_names=self.class_names,
        )

        self.erds_computer = ERDSComputer(
            sf=sf,
            freqs=np.arange(1, 41, 1.0),
            n_cycles=5.0,
            baseline_tmin=0.0,
            baseline_tmax=abs(epoch_tmin),
        )

        self.topo_mapper = TopoMapper(
            channel_names=self.channel_names[:n_channels],
        )

        self.times = self.erp_accum.get_epoch_times()
        self.n_channels = n_channels
        self.n_samples = n_samples

        # Class colors
        self._colors = {
            "rest": "#888888",
            "left_hand": "#2196F3",
            "right_hand": "#F44336",
            "feet": "#4CAF50",
            "tongue": "#FF9800",
        }

        # State
        self._fig = None
        self._axes = {}
        self._initialized = False

    def init_figure(self) -> None:
        """Create the matplotlib figure with all panels."""
        plt.ion()
        self._fig = plt.figure(figsize=(18, 11), facecolor="#1a1a2e")
        self._fig.canvas.manager.set_window_title("ERP Signal Trainer")

        gs = GridSpec(
            3, 3, figure=self._fig,
            hspace=0.35, wspace=0.3,
            left=0.06, right=0.96, top=0.93, bottom=0.06,
        )

        # Panel 1: ERP waveforms (top-left, spans 2 columns)
        ax_erp = self._fig.add_subplot(gs[0, 0:2])
        ax_erp.set_facecolor("#16213e")
        ax_erp.set_title("ERP Waveforms (Motor Cortex)", color="white", fontsize=12, fontweight="bold")
        ax_erp.set_xlabel("Time (s relative to cue)", color="#aaa")
        ax_erp.set_ylabel("Amplitude (µV)", color="#aaa")
        ax_erp.tick_params(colors="#888")
        ax_erp.axvline(0, color="#FFD700", linestyle="--", alpha=0.7, label="Cue onset")
        ax_erp.axhline(0, color="#444", linestyle="-", alpha=0.3)
        ax_erp.set_xlim(self.epoch_tmin, self.epoch_tmax)
        self._axes["erp"] = ax_erp

        # Panel 2: Topomap (top-right)
        ax_topo = self._fig.add_subplot(gs[0, 2])
        ax_topo.set_facecolor("#16213e")
        ax_topo.set_title("Scalp Map", color="white", fontsize=12, fontweight="bold")
        self._axes["topo"] = ax_topo

        # Panel 3: ERDS% spectrogram (middle, spans 2 columns)
        ax_tf = self._fig.add_subplot(gs[1, 0:2])
        ax_tf.set_facecolor("#16213e")
        ax_tf.set_title("ERDS% Time-Frequency Map", color="white", fontsize=12, fontweight="bold")
        ax_tf.set_xlabel("Time (s relative to cue)", color="#aaa")
        ax_tf.set_ylabel("Frequency (Hz)", color="#aaa")
        ax_tf.tick_params(colors="#888")
        ax_tf.axvline(0, color="#FFD700", linestyle="--", alpha=0.7)
        ax_tf.set_xlim(self.epoch_tmin, self.epoch_tmax)
        self._axes["tf"] = ax_tf

        # Panel 4: Signal quality / r² (middle-right)
        ax_r2 = self._fig.add_subplot(gs[1, 2])
        ax_r2.set_facecolor("#16213e")
        ax_r2.set_title("Class Discriminability (r²)", color="white", fontsize=12, fontweight="bold")
        ax_r2.tick_params(colors="#888")
        self._axes["r2"] = ax_r2

        # Panel 5: Band power timecourse (bottom, spans 2 columns)
        ax_band = self._fig.add_subplot(gs[2, 0:2])
        ax_band.set_facecolor("#16213e")
        ax_band.set_title("Band Power ERDS% (Mu 8-12Hz & Beta 13-30Hz)", color="white", fontsize=12, fontweight="bold")
        ax_band.set_xlabel("Time (s relative to cue)", color="#aaa")
        ax_band.set_ylabel("ERDS%", color="#aaa")
        ax_band.tick_params(colors="#888")
        ax_band.axvline(0, color="#FFD700", linestyle="--", alpha=0.7)
        ax_band.axhline(0, color="#666", linestyle="-", alpha=0.5)
        ax_band.set_xlim(self.epoch_tmin, self.epoch_tmax)
        self._axes["band"] = ax_band

        # Panel 6: Feedback text (bottom-right)
        ax_fb = self._fig.add_subplot(gs[2, 2])
        ax_fb.set_facecolor("#16213e")
        ax_fb.set_title("Training Feedback", color="white", fontsize=12, fontweight="bold")
        ax_fb.axis("off")
        self._axes["feedback"] = ax_fb

        # Super title
        self._fig.suptitle(
            "ERP Signal Trainer — Motor Imagery Training & Data Collection",
            color="#00d4ff", fontsize=14, fontweight="bold",
        )

        plt.draw()
        plt.pause(0.1)
        self._initialized = True

    def update(self, epoch: np.ndarray, class_name: str) -> None:
        """Process a new trial epoch and update all displays.

        Args:
            epoch: EEG epoch, shape (n_channels, n_samples).
            class_name: Class label for this trial.
        """
        if not self._initialized:
            self.init_figure()

        # Add to accumulator
        self.erp_accum.add_epoch(epoch, class_name)

        # Update each panel
        self._update_erp_panel(class_name)
        self._update_tf_panel(epoch, class_name)
        self._update_band_panel(class_name)
        self._update_topo_panel(epoch, class_name)
        self._update_r2_panel()
        self._update_feedback_panel(class_name)

        plt.draw()
        plt.pause(0.05)

    def _update_erp_panel(self, current_class: str) -> None:
        """Update ERP waveform panel with running averages."""
        ax = self._axes["erp"]
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.axvline(0, color="#FFD700", linestyle="--", alpha=0.7, label="Cue")
        ax.axhline(0, color="#444", linestyle="-", alpha=0.3)

        for cls_name in self.class_names:
            n_trials = self.erp_accum.get_trial_count(cls_name)
            if n_trials == 0:
                continue

            mean_erp, std_erp = self.erp_accum.get_erp(cls_name)
            color = self._colors.get(cls_name, "#ffffff")

            # Show primary motor channel (C3 = index 2 typically)
            ch = self.motor_channels[0] if self.motor_channels else 0
            ax.plot(
                self.times, mean_erp[ch],
                color=color, linewidth=2,
                label=f"{cls_name} (n={n_trials})",
            )
            ax.fill_between(
                self.times,
                mean_erp[ch] - std_erp[ch],
                mean_erp[ch] + std_erp[ch],
                color=color, alpha=0.15,
            )

        ch_name = self.channel_names[self.motor_channels[0]] if self.motor_channels else "ch0"
        ax.set_title(f"ERP Waveforms — {ch_name}", color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", color="#aaa", fontsize=9)
        ax.set_ylabel("Amplitude (µV)", color="#aaa", fontsize=9)
        ax.legend(fontsize=8, loc="upper right", facecolor="#16213e", edgecolor="#444", labelcolor="white")
        ax.set_xlim(self.epoch_tmin, self.epoch_tmax)
        ax.tick_params(colors="#888", labelsize=8)

    def _update_tf_panel(self, epoch: np.ndarray, class_name: str) -> None:
        """Update time-frequency ERDS% spectrogram."""
        ax = self._axes["tf"]
        ax.clear()
        ax.set_facecolor("#16213e")

        ch = self.motor_channels[0] if self.motor_channels else 0

        # Use average across all trials of this class
        all_trials = self.erp_accum.get_all_trials(class_name)
        if all_trials is not None and all_trials.shape[0] >= 2:
            mean_erds, _ = self.erds_computer.compute_erds_average(
                all_trials, channel=ch, epoch_tmin=self.epoch_tmin,
            )
        else:
            mean_erds = self.erds_computer.compute_erds(
                epoch, channel=ch, epoch_tmin=self.epoch_tmin,
            )

        im = ax.imshow(
            mean_erds,
            aspect="auto",
            origin="lower",
            cmap="RdBu_r",
            vmin=-100, vmax=100,
            extent=[self.epoch_tmin, self.epoch_tmax,
                    self.erds_computer.freqs[0], self.erds_computer.freqs[-1]],
        )

        ax.axvline(0, color="#FFD700", linestyle="--", alpha=0.7)

        # Mark mu and beta bands
        ax.axhspan(8, 12, color="cyan", alpha=0.1)
        ax.axhspan(13, 30, color="lime", alpha=0.05)
        ax.text(self.epoch_tmax - 0.3, 10, "mu", color="cyan", fontsize=8, ha="right")
        ax.text(self.epoch_tmax - 0.3, 20, "beta", color="lime", fontsize=8, ha="right")

        ch_name = self.channel_names[ch] if ch < len(self.channel_names) else f"ch{ch}"
        ax.set_title(
            f"ERDS% — {ch_name} ({class_name})", color="white", fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Time (s)", color="#aaa", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", color="#aaa", fontsize=9)
        ax.tick_params(colors="#888", labelsize=8)

        # Colorbar
        if hasattr(self, "_tf_cbar"):
            self._tf_cbar.remove()
        self._tf_cbar = self._fig.colorbar(im, ax=ax, shrink=0.8, label="ERDS%")
        self._tf_cbar.ax.tick_params(labelsize=7, colors="#888")
        self._tf_cbar.set_label("ERDS%", color="#aaa", fontsize=8)

    def _update_band_panel(self, class_name: str) -> None:
        """Update mu/beta band power timecourse."""
        ax = self._axes["band"]
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.axvline(0, color="#FFD700", linestyle="--", alpha=0.7)
        ax.axhline(0, color="#666", linestyle="-", alpha=0.5)

        ch = self.motor_channels[0] if self.motor_channels else 0

        for cls_name in self.class_names:
            all_trials = self.erp_accum.get_all_trials(cls_name)
            if all_trials is None or all_trials.shape[0] < 1:
                continue

            color = self._colors.get(cls_name, "#ffffff")

            # Mu band (8-12 Hz)
            _, mean_mu, std_mu = self.erds_computer.compute_band_power_average(
                all_trials, channel=ch, band=(8, 12), epoch_tmin=self.epoch_tmin,
            )
            ax.plot(self.times[:len(mean_mu)], mean_mu, color=color, linewidth=2,
                    label=f"{cls_name} mu", linestyle="-")
            ax.fill_between(
                self.times[:len(mean_mu)],
                mean_mu - std_mu, mean_mu + std_mu,
                color=color, alpha=0.1,
            )

        ax.set_title("Mu Band ERDS% (8-12 Hz)", color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", color="#aaa", fontsize=9)
        ax.set_ylabel("ERDS%", color="#aaa", fontsize=9)
        ax.legend(fontsize=7, loc="lower left", facecolor="#16213e", edgecolor="#444", labelcolor="white")
        ax.set_xlim(self.epoch_tmin, self.epoch_tmax)
        ax.tick_params(colors="#888", labelsize=8)

        # Annotate ERD region
        ax.annotate(
            "ERD\n(desync)", xy=(1.5, -30), fontsize=8, color="#66bb6a",
            ha="center", style="italic",
        )

    def _update_topo_panel(self, epoch: np.ndarray, class_name: str) -> None:
        """Update scalp topographic map at peak ERD timepoint."""
        ax = self._axes["topo"]
        ax.clear()
        ax.set_facecolor("#16213e")

        mean_erp, _ = self.erp_accum.get_erp(class_name, baseline_correct=True)

        # Show amplitude at t=2.0s (typical peak ERD)
        t_target = 2.0
        t_idx = int((t_target - self.epoch_tmin) * self.sf)
        t_idx = min(t_idx, mean_erp.shape[1] - 1)

        values = mean_erp[:, t_idx]

        if len(values) >= len(self.topo_mapper.channel_names):
            values = values[:len(self.topo_mapper.channel_names)]
        else:
            padded = np.zeros(len(self.topo_mapper.channel_names))
            padded[:len(values)] = values
            values = padded

        try:
            self.topo_mapper.plot(
                values, ax=ax,
                title=f"t={t_target:.1f}s ({class_name})",
                cmap="RdBu_r", colorbar=False, show_names=True,
            )
        except Exception as e:
            ax.text(0.5, 0.5, f"Topo error:\n{e}", transform=ax.transAxes,
                    ha="center", va="center", color="red", fontsize=8)

    def _update_r2_panel(self) -> None:
        """Update signed r² discriminability display."""
        ax = self._axes["r2"]
        ax.clear()
        ax.set_facecolor("#16213e")

        # Find two classes with the most trials
        trial_counts = {
            cls: self.erp_accum.get_trial_count(cls)
            for cls in self.class_names
        }
        sorted_classes = sorted(trial_counts.keys(), key=lambda k: trial_counts[k], reverse=True)
        eligible = [c for c in sorted_classes if trial_counts[c] >= 2]

        if len(eligible) < 2:
            ax.text(
                0.5, 0.5, "Need 2+ trials\nper class\nfor r² analysis",
                transform=ax.transAxes, ha="center", va="center",
                color="#888", fontsize=10,
            )
            ax.set_title("Class Discriminability", color="white", fontsize=11, fontweight="bold")
            return

        cls_a, cls_b = eligible[0], eligible[1]
        r2 = self.erp_accum.compute_signed_r2(cls_a, cls_b)

        # Show r² for motor channels as bar chart
        ch_labels = []
        ch_r2_peak = []
        for ch_idx in range(min(self.n_channels, len(self.channel_names))):
            ch_labels.append(self.channel_names[ch_idx])
            # Peak absolute r² in the post-stimulus window
            post_start = int(abs(self.epoch_tmin) * self.sf)
            ch_r2_peak.append(float(np.abs(r2[ch_idx, post_start:]).max()))

        colors_bar = ["#00d4ff" if ch in self.motor_channels else "#555" for ch in range(len(ch_labels))]

        ax.barh(range(len(ch_labels)), ch_r2_peak, color=colors_bar, height=0.7)
        ax.set_yticks(range(len(ch_labels)))
        ax.set_yticklabels(ch_labels, fontsize=7, color="#aaa")
        ax.set_xlabel("Peak |r²|", color="#aaa", fontsize=8)
        ax.set_title(f"r² : {cls_a} vs {cls_b}", color="white", fontsize=10, fontweight="bold")
        ax.tick_params(colors="#888", labelsize=7)
        ax.set_xlim(0, 1)
        ax.axvline(0.1, color="#666", linestyle=":", alpha=0.5)

    def _update_feedback_panel(self, class_name: str) -> None:
        """Update training feedback text."""
        ax = self._axes["feedback"]
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.axis("off")

        lines = []
        n_total = self.erp_accum.get_trial_count()
        lines.append(f"Total trials: {n_total}")

        for cls in self.class_names:
            n = self.erp_accum.get_trial_count(cls)
            lines.append(f"  {cls}: {n}")

        # SNR feedback
        if self.erp_accum.get_trial_count(class_name) >= 3:
            snr = self.erp_accum.compute_erp_snr(class_name)
            for ch_idx in self.motor_channels:
                if ch_idx < len(snr) and ch_idx < len(self.channel_names):
                    ch_name = self.channel_names[ch_idx]
                    quality = "STRONG" if snr[ch_idx] > 2.0 else "moderate" if snr[ch_idx] > 1.0 else "weak"
                    color = "#4CAF50" if quality == "STRONG" else "#FFD700" if quality == "moderate" else "#F44336"
                    lines.append(f"  {ch_name} SNR: {snr[ch_idx]:.1f} ({quality})")

        # Suggestions
        lines.append("")
        if n_total < 5:
            lines.append("Keep going! Need more trials")
            lines.append("for reliable ERPs.")
        elif n_total < 20:
            lines.append("Good progress. ERPs are")
            lines.append("starting to stabilize.")
        else:
            lines.append("ERPs well established.")
            lines.append("Check r² for best channels.")

        y_pos = 0.95
        for line in lines:
            color = "#aaa"
            if "STRONG" in line:
                color = "#4CAF50"
            elif "weak" in line:
                color = "#F44336"
            ax.text(0.05, y_pos, line, transform=ax.transAxes,
                    fontsize=9, color=color, verticalalignment="top",
                    fontfamily="monospace")
            y_pos -= 0.08

        ax.set_title("Training Feedback", color="white", fontsize=11, fontweight="bold")

    def show(self) -> None:
        """Block until the figure is closed."""
        if self._initialized:
            plt.ioff()
            plt.show()


# ======================================================================
# Live Data Collection
# ======================================================================

def run_live_collection(config: dict, args: argparse.Namespace) -> None:
    """Run live paradigm with real-time ERP feedback."""
    logger = logging.getLogger("erp_trainer")

    # Configuration
    train_cfg = config.get("training", {})
    preproc_cfg = config.get("preprocessing", {})

    classes = args.classes or train_cfg.get("classes", ["rest", "left_hand", "right_hand", "feet", "tongue"])
    n_trials = args.n_trials or train_cfg.get("n_trials_per_class", 40)
    paths_cfg = config.get("paths", {})
    output_dir = Path(args.output_dir or paths_cfg.get("raw_data_dir", "data/raw"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Epoch timing
    epoch_tmin = -1.0  # 1s before cue
    epoch_tmax = 5.0   # 5s after cue
    fixation_dur = train_cfg.get("fixation_duration", 2.0)
    cue_dur = train_cfg.get("cue_duration", 1.25)
    imagery_dur = train_cfg.get("imagery_duration", 4.0)

    # Board setup
    logger.info("Connecting to board...")
    board = BoardManager(config)
    board.connect()
    sf = board.get_sampling_rate()
    eeg_channels = board.get_eeg_channels()
    n_eeg = len(eeg_channels)
    logger.info("Board: %d Hz, %d channels, synthetic=%s", sf, n_eeg, board.is_synthetic())

    n_epoch_samples = int((epoch_tmax - epoch_tmin) * sf)
    baseline_samples = int(abs(epoch_tmin) * sf)

    # MI bandpass
    mi_bp_low = preproc_cfg.get("mi_bandpass_low", 8.0)
    mi_bp_high = preproc_cfg.get("mi_bandpass_high", 30.0)
    nyquist = sf / 2.0
    if mi_bp_high >= nyquist:
        mi_bp_high = nyquist - 1.0

    # ERP display
    display = ERPDisplay(
        n_channels=n_eeg,
        sf=sf,
        epoch_tmin=epoch_tmin,
        epoch_tmax=epoch_tmax,
        class_names=classes,
        channel_names=CHANNEL_NAMES_16[:n_eeg],
    )
    display.init_figure()

    # Data storage
    all_epochs = []
    all_labels = []
    events = []

    # Simple console + matplotlib paradigm
    import random

    print("\n" + "=" * 60)
    print("ERP SIGNAL TRAINER — DATA COLLECTION")
    print("=" * 60)
    print(f"  Classes:       {classes}")
    print(f"  Trials/class:  {n_trials}")
    print(f"  Total trials:  {n_trials * len(classes)}")
    print(f"  Board:         {sf} Hz, {n_eeg} ch")
    print(f"  Epoch:         [{epoch_tmin}, {epoch_tmax}] s")
    print(f"  Close the plot window or Ctrl+C to stop.")
    print("=" * 60 + "\n")

    # Build trial sequence (block-randomized)
    trial_sequence = []
    for _ in range(n_trials):
        block = list(classes)
        random.shuffle(block)
        trial_sequence.extend(block)

    total_trials = len(trial_sequence)
    start_time = time.time()

    # Flush board buffer
    board.get_board_data()

    try:
        for trial_idx, class_name in enumerate(trial_sequence):
            if not plt.fignum_exists(display._fig.number):
                logger.info("Display window closed. Stopping.")
                break

            trial_num = trial_idx + 1
            print(f"\r  Trial {trial_num}/{total_trials}: {class_name:15s}", end="", flush=True)

            # --- Fixation ---
            time.sleep(fixation_dur)

            # --- Cue onset: record timestamp ---
            cue_time = time.time()
            elapsed = cue_time - start_time
            cue_sample = int(elapsed * sf)

            events.append({
                "label": class_name,
                "timestamp": cue_time,
                "sample_index": cue_sample,
            })

            # --- Imagery period ---
            time.sleep(cue_dur + imagery_dur)

            # --- Rest ---
            rest_dur = random.uniform(
                train_cfg.get("rest_duration_min", 1.5),
                train_cfg.get("rest_duration_max", 3.0),
            )
            time.sleep(rest_dur)

            # --- Extract epoch ---
            # We need data from (cue - 1s) to (cue + 5s) = 6s total
            total_needed = n_epoch_samples + sf  # extra buffer
            raw = board.get_data(total_needed)

            if raw.shape[1] < n_epoch_samples:
                logger.warning("Not enough data for epoch (got %d, need %d). Skipping.",
                               raw.shape[1], n_epoch_samples)
                continue

            # Take the last n_epoch_samples from the buffer
            eeg_data = raw[eeg_channels, -n_epoch_samples:]

            # Preprocess
            filtered = bandpass_filter(
                eeg_data, sf=sf, low=mi_bp_low, high=mi_bp_high, causal=False
            )
            filtered = common_average_reference(filtered)

            # Store
            all_epochs.append(filtered)
            all_labels.append(class_name)

            # Update display
            display.update(filtered, class_name)

        print()  # newline after progress

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        board.disconnect()
        logger.info("Board disconnected.")

    # --- Save data ---
    if all_epochs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"erp_session_{timestamp}.npz"
        save_path = output_dir / filename

        epochs_array = np.stack(all_epochs, axis=0)  # (n_trials, n_ch, n_samples)
        labels_array = np.array(all_labels)

        np.savez(
            str(save_path),
            epochs=epochs_array,
            labels=labels_array,
            events_json=json.dumps(events),
            sf=np.array(sf),
            eeg_channels=np.array(eeg_channels),
            epoch_tmin=np.array(epoch_tmin),
            epoch_tmax=np.array(epoch_tmax),
            class_names=np.array(classes),
        )

        n_collected = len(all_epochs)
        event_counts = Counter(all_labels)

        print("\n" + "=" * 60)
        print("DATA COLLECTION COMPLETE")
        print("=" * 60)
        print(f"  Trials collected: {n_collected}/{total_trials}")
        for cls, cnt in sorted(event_counts.items()):
            print(f"    {cls:15s}: {cnt}")
        print(f"  Saved to: {save_path}")
        print(f"  Epoch shape: {epochs_array.shape}")
        print("=" * 60)

        logger.info("Saved %d epochs to %s", n_collected, save_path)
    else:
        print("No epochs collected.")

    # Keep display open for review
    print("\nClose the plot window to exit.")
    display.show()


# ======================================================================
# Offline Review Mode
# ======================================================================

def run_review(review_path: str, config: dict) -> None:
    """Load a recorded session and display ERP analysis."""
    logger = logging.getLogger("erp_trainer")

    path = Path(review_path)
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    logger.info("Loading %s...", path)
    archive = np.load(str(path), allow_pickle=False)

    # Handle both formats: pre-epoched and raw continuous
    if "epochs" in archive:
        epochs = archive["epochs"]
        labels = archive["labels"]
        sf = int(archive.get("sf", 125))
        epoch_tmin = float(archive.get("epoch_tmin", -1.0))
        epoch_tmax = float(archive.get("epoch_tmax", 5.0))
        class_names = list(archive.get("class_names", ["rest", "left_hand", "right_hand"]))
    elif "data" in archive:
        # Raw continuous format — need to epoch it
        from src.training.recorder import DataRecorder
        raw_data, events, metadata = DataRecorder.load(str(path))
        sf = metadata.get("sf", 125)
        eeg_channels = metadata.get("eeg_channels", list(range(min(16, raw_data.shape[0]))))
        epoch_tmin = -1.0
        epoch_tmax = 5.0
        n_samples = int((epoch_tmax - epoch_tmin) * sf)

        epochs_list = []
        labels_list = []
        for ev in events:
            start = ev["sample_index"] + int(epoch_tmin * sf)
            end = start + n_samples
            if start >= 0 and end <= raw_data.shape[1]:
                ep = raw_data[eeg_channels, start:end]
                # Preprocess
                preproc_cfg = config.get("preprocessing", {})
                mi_low = preproc_cfg.get("mi_bandpass_low", 8.0)
                mi_high = preproc_cfg.get("mi_bandpass_high", 30.0)
                nyq = sf / 2.0
                if mi_high >= nyq:
                    mi_high = nyq - 1.0
                ep = bandpass_filter(ep, sf=sf, low=mi_low, high=mi_high, causal=False)
                ep = common_average_reference(ep)
                epochs_list.append(ep)
                labels_list.append(ev["label"])

        if not epochs_list:
            logger.error("No valid epochs extracted from recording.")
            sys.exit(1)

        epochs = np.stack(epochs_list, axis=0)
        labels = np.array(labels_list)
        class_names = sorted(set(labels_list))
    else:
        logger.error("Unrecognized archive format. Expected 'epochs' or 'data' key.")
        sys.exit(1)

    n_ch = epochs.shape[1]
    logger.info("Loaded %d epochs, %d channels, %d samples, sf=%d Hz",
                epochs.shape[0], n_ch, epochs.shape[2], sf)

    # Create display and feed all epochs
    display = ERPDisplay(
        n_channels=n_ch,
        sf=sf,
        epoch_tmin=epoch_tmin,
        epoch_tmax=epoch_tmax,
        class_names=list(class_names),
        channel_names=CHANNEL_NAMES_16[:n_ch],
    )
    display.init_figure()

    for i in range(epochs.shape[0]):
        label = str(labels[i])
        display.erp_accum.add_epoch(epochs[i], label)

    # Final update with last trial
    last_label = str(labels[-1])
    display.update(epochs[-1], last_label)

    print(f"\nLoaded {epochs.shape[0]} epochs from {path.name}")
    print("Close the plot window to exit.")
    display.show()


# ======================================================================
# Entry point
# ======================================================================

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)

    if args.review:
        run_review(args.review, config)
    else:
        run_live_collection(config, args)


if __name__ == "__main__":
    main()
