#!/usr/bin/env python3
"""Real-time EEG Cursor Control application.

Connects to the EEG board, loads a trained classifier, and runs a
real-time control loop that translates 5-class motor imagery into
4-directional cursor movement with click via sustained imagery.

Classes:
    - rest: no movement
    - left_hand: cursor LEFT
    - right_hand: cursor RIGHT
    - feet: cursor DOWN
    - tongue: cursor UP

Usage:
    python scripts/run_eeg_cursor.py --model models/csp_lda_20260325.pkl
    python scripts/run_eeg_cursor.py --model models/eegnet.pt --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# CPU thread reservation — set BEFORE importing numpy/scipy so that
# MKL/OpenBLAS respect the limits.
# ---------------------------------------------------------------------------
_TOTAL_CORES = os.cpu_count() or 4
_MAX_WORKER_THREADS = max(1, _TOTAL_CORES - 1)
_COMPUTE_THREADS = str(max(1, _MAX_WORKER_THREADS - 1))

os.environ.setdefault("OMP_NUM_THREADS", _COMPUTE_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", _COMPUTE_THREADS)

# ---------------------------------------------------------------------------
# Path hack: allow running from the repo root or the scripts/ directory
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.config import load_config
from src.acquisition.board import BoardManager
from src.preprocessing.filters import bandpass_filter, common_average_reference
from src.classification.base import BaseClassifier
from src.control.cursor_control import EEGCursorController
from src.adaptation.errp_detector import ErrPP300Detector
from src.adaptation.seal_engine import SEALAdaptationEngine
from src.analysis.state_monitor import BCIStateMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the EEG Cursor BCI in real time."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to a trained classifier file (from train_model.py).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to settings.yaml (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Threaded EEG acquisition
# ---------------------------------------------------------------------------

class EEGAcquisitionThread:
    """Continuously reads EEG data from the board on a background thread.

    Maintains a rolling buffer of the most recent ``window_samples``
    samples, updated every ``poll_interval`` seconds.  The main loop
    reads the latest window via ``get_window()`` without blocking.
    """

    def __init__(
        self,
        board: BoardManager,
        window_seconds: float,
        poll_interval: float = 0.02,
    ) -> None:
        self._board = board
        self._sf = board.get_sampling_rate()
        self._window_samples = int(window_seconds * self._sf)
        self._poll_interval = poll_interval

        self._buffer: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Board error tracking
        self._consecutive_failures = 0
        self._board_error = False
        self._board_error_time: Optional[float] = None
        self._acq_logger = logging.getLogger("run_eeg_cursor.acq")

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def get_window(self) -> Optional[np.ndarray]:
        """Return the latest EEG window, or None if not yet available."""
        with self._lock:
            return self._buffer.copy() if self._buffer is not None else None

    @property
    def has_error(self) -> bool:
        """True if the board has had 10+ consecutive read failures."""
        return self._board_error

    @property
    def error_duration(self) -> float:
        """Seconds since the board error flag was first set, or 0.0 if no error."""
        if self._board_error and self._board_error_time is not None:
            return time.monotonic() - self._board_error_time
        return 0.0

    def _run(self) -> None:
        while self._running:
            try:
                data = self._board.get_data(self._window_samples)
                if data.shape[1] >= self._window_samples:
                    with self._lock:
                        self._buffer = data[:, -self._window_samples:]
                elif data.shape[1] > 0:
                    with self._lock:
                        self._buffer = data
                # Successful read — reset failure counter
                if self._consecutive_failures > 0:
                    self._acq_logger.info(
                        "Board recovered after %d consecutive failures.",
                        self._consecutive_failures,
                    )
                self._consecutive_failures = 0
                self._board_error = False
                self._board_error_time = None
            except Exception as exc:
                self._consecutive_failures += 1
                if self._consecutive_failures <= 10 or self._consecutive_failures % 50 == 0:
                    self._acq_logger.debug(
                        "Board read failure #%d: %s",
                        self._consecutive_failures, exc,
                    )
                if self._consecutive_failures >= 10 and not self._board_error:
                    self._board_error = True
                    self._board_error_time = time.monotonic()
                    self._acq_logger.warning(
                        "Board error flag SET — %d consecutive read failures. "
                        "Board may be disconnected.",
                        self._consecutive_failures,
                    )
            time.sleep(self._poll_interval)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_eeg_cursor")

    # Graceful shutdown on Ctrl+C
    shutdown_event = threading.Event()

    def _signal_handler(sig, frame):
        logger.info("Received signal %d, shutting down...", sig)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ------------------------------------------------------------------
    # 0. Limit PyTorch intra-op threads
    # ------------------------------------------------------------------
    try:
        import torch
        torch.set_num_threads(max(1, _MAX_WORKER_THREADS - 1))
        logger.info(
            "Thread budget: %d cores, %d max workers, PyTorch=%d, OMP/MKL=%s",
            _TOTAL_CORES, _MAX_WORKER_THREADS,
            torch.get_num_threads(), _COMPUTE_THREADS,
        )
    except ImportError:
        logger.debug("PyTorch not available; thread limit not set.")

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    logger.info("Loading configuration...")
    config = load_config(args.config)

    control_cfg = config.get("control", {})
    update_rate_hz = control_cfg.get("update_rate_hz", 16)
    update_interval = 1.0 / update_rate_hz

    # ------------------------------------------------------------------
    # 2. Load trained model
    # ------------------------------------------------------------------
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        sys.exit(1)

    logger.info("Loading trained model from %s...", model_path)
    classifier = BaseClassifier.load(str(model_path))
    logger.info("Classifier loaded: %r", classifier)

    # Load label mapping (must match training)
    label_map_path = model_path.with_suffix('.labels.json')
    if label_map_path.exists():
        import json
        label_map = json.loads(label_map_path.read_text())
        # Invert: {0: "left_hand", 1: "rest", ...}
        class_names = [None] * len(label_map)
        for name, idx in label_map.items():
            class_names[idx] = name
        logger.info("Label mapping loaded: %s", label_map)
    else:
        logger.warning(
            "No label mapping file found at %s. Using config order (may be wrong!).",
            label_map_path,
        )
        class_names = config.get("training", {}).get(
            "classes", ["rest", "left_hand", "right_hand", "feet", "tongue"]
        )

    # ------------------------------------------------------------------
    # 3. Create BoardManager, connect
    # ------------------------------------------------------------------
    logger.info("Initializing board...")
    board = BoardManager(config)
    board.connect()

    sf = board.get_sampling_rate()
    eeg_channels = board.get_eeg_channels()
    logger.info("Board ready: %d Hz, %d EEG channels, synthetic=%s",
                sf, len(eeg_channels), board.is_synthetic())

    # ------------------------------------------------------------------
    # 4. MI bandpass parameters (must match training)
    # ------------------------------------------------------------------
    preproc_cfg = config.get("preprocessing", {})
    mi_bp_low = preproc_cfg.get("mi_bandpass_low", 8.0)
    mi_bp_high = preproc_cfg.get("mi_bandpass_high", 30.0)
    mi_bp_order = preproc_cfg.get("bandpass_order", 4)

    # Clamp to Nyquist
    nyquist = sf / 2.0
    if mi_bp_high >= nyquist:
        mi_bp_high = nyquist - 1.0
        logger.warning("Clamped mi_bandpass_high to %.1f Hz (Nyquist=%.1f Hz).",
                        mi_bp_high, nyquist)

    logger.info("MI bandpass filter: [%.1f, %.1f] Hz, order %d (non-causal, per-window)",
                mi_bp_low, mi_bp_high, mi_bp_order)

    # ------------------------------------------------------------------
    # 5. Create EEG cursor controller
    # ------------------------------------------------------------------
    cursor = EEGCursorController(config)
    logger.info("Cursor controller: %r", cursor)

    # ------------------------------------------------------------------
    # 5b. Create self-adaptation system (ErrP/P300 + SEAL)
    # ------------------------------------------------------------------
    adapt_cfg = config.get("adaptation", {})
    adaptation_enabled = adapt_cfg.get("enabled", True)

    errp_detector = ErrPP300Detector(
        sf=sf,
        fcz_idx=adapt_cfg.get("fcz_channel", 14),
        fz_idx=adapt_cfg.get("fz_channel", 15),
        p3_idx=adapt_cfg.get("p3_channel", 12),
        p4_idx=adapt_cfg.get("p4_channel", 13),
        erp_window_ms=adapt_cfg.get("erp_window_ms", 600),
        baseline_ms=adapt_cfg.get("baseline_ms", 200),
        errp_threshold=adapt_cfg.get("errp_threshold", 8.0),
        p300_threshold=adapt_cfg.get("p300_threshold", 5.0),
    )

    seal_engine = SEALAdaptationEngine(config)
    seal_engine.set_classifier(classifier, class_names)

    # Optional GFlowNet strategy optimizer for SEAL hyperparameters
    gflownet_optimizer = None
    gflownet_enabled = config.get("advanced", {}).get("gflownet_enabled", False)
    if adaptation_enabled and gflownet_enabled:
        try:
            from src.adaptation.gflownet_strategy import GFlowNetSEALOptimizer
            gflownet_optimizer = GFlowNetSEALOptimizer(
                config.get("advanced", {})
            )
            logger.info("GFlowNet SEAL optimizer ENABLED")
        except ImportError:
            logger.warning("GFlowNet requested but PyTorch not available")

    if adaptation_enabled:
        logger.info("Self-adaptation ENABLED (ErrP/P300 + SEAL)")
    else:
        logger.info("Self-adaptation DISABLED")

    # ------------------------------------------------------------------
    # 5c. Create state monitor (fatigue/attention EWS)
    # ------------------------------------------------------------------
    adv_cfg = config.get("advanced", {})
    state_monitor_enabled = adv_cfg.get("state_monitor", False)
    state_monitor: Optional[BCIStateMonitor] = None

    if state_monitor_enabled:
        state_monitor = BCIStateMonitor(
            sf=sf,
            n_channels=len(eeg_channels),
            window_s=adv_cfg.get("state_monitor_window_s", 30.0),
            update_interval_s=adv_cfg.get("state_monitor_interval_s", 5.0),
        )
        logger.info("State monitor ENABLED (EWS fatigue/attention)")
    else:
        logger.info("State monitor DISABLED")

    # ------------------------------------------------------------------
    # 6. Start EEG acquisition thread
    # ------------------------------------------------------------------
    train_cfg = config.get("training", {})
    window_seconds = (
        train_cfg.get("classification_window_end", 4.0)
        - train_cfg.get("classification_window_start", 1.5)
    )
    logger.info("Classification window: %.1f seconds (matching training).", window_seconds)
    acq_thread = EEGAcquisitionThread(board, window_seconds=window_seconds)
    acq_thread.start()
    logger.info("EEG acquisition thread started (window=%.1f s).", window_seconds)

    # Wait briefly for the buffer to fill
    logger.info("Waiting for initial data buffer to fill...")
    time.sleep(window_seconds + 0.5)

    # ------------------------------------------------------------------
    # 7. Main control loop
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EEG CURSOR - RUNNING")
    print("=" * 60)
    print(f"  Mode:          Pure EEG (5-class MI)")
    print(f"  Model:         {model_path.name}")
    print(f"  Classes:       {class_names}")
    print(f"  Update rate:   {update_rate_hz} Hz")
    print(f"  Board:         {sf} Hz, {len(eeg_channels)} ch, synthetic={board.is_synthetic()}")
    print(f"  Press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    loop_count = 0
    status_interval = int(update_rate_hz * 3)  # Print status every ~3 seconds
    total_latency = 0.0
    classifications = 0
    seal_prev_accuracy = 0.0
    session_start_time = datetime.now(timezone.utc)

    try:
        while not shutdown_event.is_set():
            loop_start = time.monotonic()

            # ----- a. Get latest EEG window -----
            # Check for board disconnection
            if acq_thread.has_error:
                err_dur = acq_thread.error_duration
                if err_dur >= 30.0:
                    logger.error(
                        "Board has been unresponsive for %.0f seconds. "
                        "Consider stopping the session and reconnecting the board.",
                        err_dur,
                    )
                else:
                    logger.warning(
                        "Board read errors detected (%.0f s). "
                        "Data may be stale — check the board connection.",
                        err_dur,
                    )
                time.sleep(update_interval)
                continue

            raw_window = acq_thread.get_window()
            if raw_window is None:
                time.sleep(update_interval)
                continue

            # Extract EEG channels only
            eeg_window = raw_window[eeg_channels, :]

            # ----- a2. State monitor (fatigue/attention EWS) -----
            if state_monitor is not None:
                state_result = state_monitor.update(
                    eeg_chunk=eeg_window,
                    current_time=time.monotonic(),
                )
                if state_result is not None:
                    sm_state = state_result["state"]
                    if sm_state == "degraded":
                        logger.warning(
                            "STATE MONITOR: %s — fatigue=%.2f, attention=%.2f, "
                            "electrodes=%.2f | %s",
                            sm_state.upper(),
                            state_result["fatigue_score"],
                            state_result["attention_score"],
                            state_result["electrode_quality"],
                            state_result.get("recommendation", ""),
                        )
                    elif sm_state == "warning":
                        logger.warning(
                            "STATE MONITOR: %s — fatigue=%.2f, attention=%.2f, "
                            "electrodes=%.2f",
                            sm_state.upper(),
                            state_result["fatigue_score"],
                            state_result["attention_score"],
                            state_result["electrode_quality"],
                        )
                    else:
                        logger.debug(
                            "STATE MONITOR: %s — fatigue=%.2f, attention=%.2f",
                            sm_state,
                            state_result["fatigue_score"],
                            state_result["attention_score"],
                        )

            # ----- b. Preprocess (same pipeline as training) -----
            filtered = bandpass_filter(eeg_window, sf=sf, low=mi_bp_low,
                                       high=mi_bp_high, order=mi_bp_order,
                                       causal=False)
            filtered = common_average_reference(filtered)

            # ----- c. Classify -----
            trial = filtered[np.newaxis, :, :]

            try:
                prediction, probabilities, decision = classifier.predict_all(trial)
            except Exception as exc:
                logger.debug("Classification error: %s", exc)
                time.sleep(update_interval)
                continue

            proba = probabilities[0] if probabilities.ndim > 1 else probabilities

            # ----- d. Update cursor (movement + click detection) -----
            status = cursor.update(proba, class_names)

            # ----- d2. Self-adaptation (ErrP/P300 + SEAL) -----
            if adaptation_enabled:
                now = time.monotonic()
                predicted_int = int(np.argmax(proba))

                # Feed EEG to the ErrP detector's continuous buffer
                errp_detector.update_buffer(eeg_window, now)

                # Record this action for ErrP evaluation
                action_executed = (
                    status["direction"] is not None or
                    status["click_event"] is not None
                )
                if action_executed:
                    errp_detector.record_action(
                        timestamp=now,
                        predicted_class=status["predicted_class"],
                        eeg_epoch=filtered,
                    )
                    seal_engine.on_prediction(
                        eeg_epoch=filtered,
                        predicted_class=predicted_int,
                        action_time=now,
                        action_type="click" if status["click_event"] else "move",
                    )

                # Check for ErrP/P300 responses to past actions
                erp_results = errp_detector.detect(now)
                for erp_res in erp_results:
                    seal_info = seal_engine.on_errp_result(
                        erp_res["timestamp"],
                        erp_res["result"],
                        erp_res["confidence"],
                    )
                    if seal_info and seal_info.get("should_undo"):
                        if adapt_cfg.get("auto_undo", True):
                            # Auto-correct: reverse cursor by a fixed undo step
                            # (we cannot replay the exact erroneous velocity since
                            # it occurred 650ms+ ago across multiple loop iterations)
                            undo_px = control_cfg.get("max_velocity", 25) * 2
                            erroneous_class = seal_info.get("predicted_class", -1)
                            direction_map = control_cfg.get("direction_map", {})
                            # Reverse the erroneous direction
                            for cls_name, direction in direction_map.items():
                                cls_idx = class_names.index(cls_name) if cls_name in class_names else -1
                                if cls_idx == erroneous_class:
                                    dx, dy = 0.0, 0.0
                                    if direction == "left": dx = undo_px
                                    elif direction == "right": dx = -undo_px
                                    elif direction == "up": dy = undo_px
                                    elif direction == "down": dy = -undo_px
                                    cursor._mouse.move_relative(dx=dx, dy=dy)
                                    break
                            logger.info(
                                "AUTO-UNDO: ErrP detected (conf=%.2f), reversed erroneous %s",
                                erp_res["confidence"],
                                erp_res.get("predicted_class", "unknown"),
                            )

                # Periodically update model from reward signals
                if seal_engine.maybe_update(now):
                    stats = seal_engine.get_stats()
                    logger.info(
                        "SEAL: Model self-adapted (#%d) — confirmed=%d, corrected=%d",
                        stats["n_updates"], stats["n_confirmed"], stats["n_corrected"],
                    )
                    # GFlowNet: update with reward and propose next config
                    if gflownet_optimizer is not None:
                        n_total = stats["n_confirmed"] + stats["n_corrected"]
                        accuracy_now = stats["n_confirmed"] / max(n_total, 1)
                        gflownet_optimizer.update(
                            config_used={
                                "blend_ratio": seal_engine.blend_ratio,
                                "update_interval": int(seal_engine.update_interval_s),
                                "min_samples": seal_engine.min_samples_for_update,
                                "learning_rate": 1e-4,
                            },
                            accuracy_before=seal_prev_accuracy,
                            accuracy_after=accuracy_now,
                        )
                        seal_prev_accuracy = accuracy_now
                        new_cfg = gflownet_optimizer.propose_config(
                            accuracy_now, n_total
                        )
                        seal_engine.blend_ratio = new_cfg["blend_ratio"]
                        seal_engine.update_interval_s = float(new_cfg["update_interval"])
                        seal_engine.min_samples_for_update = new_cfg["min_samples"]
                        logger.info(
                            "GFlowNet: updated SEAL config — blend=%.2f, interval=%ds, min_samples=%d",
                            new_cfg["blend_ratio"], new_cfg["update_interval"], new_cfg["min_samples"],
                        )

            # ----- e. Timing and status -----
            loop_elapsed = time.monotonic() - loop_start
            total_latency += loop_elapsed
            classifications += 1
            loop_count += 1

            if loop_count % status_interval == 0:
                avg_latency_ms = (total_latency / max(classifications, 1)) * 1000
                cur_x, cur_y = cursor.position
                logger.info(
                    "Status: class=%s dir=%s (p=%.2f), cursor=(%d,%d), "
                    "vel=(%.1f,%.1f), latency=%.1f ms, clicks=%d",
                    status["predicted_class"],
                    status["direction"],
                    status["confidence"],
                    cur_x, cur_y,
                    status["velocity"][0], status["velocity"][1],
                    avg_latency_ms,
                    cursor.total_clicks,
                )

            # Sleep for the remainder of the update interval
            sleep_time = update_interval - loop_elapsed
            if sleep_time > 0:
                shutdown_event.wait(timeout=sleep_time)

    except Exception:
        logger.exception("Fatal error in control loop.")
    finally:
        # ------------------------------------------------------------------
        # 8. Cleanup
        # ------------------------------------------------------------------
        logger.info("Shutting down...")

        acq_thread.stop()
        logger.info("EEG acquisition thread stopped.")

        board.disconnect()
        logger.info("Board disconnected.")

        # Final stats
        session_end_time = datetime.now(timezone.utc)
        session_duration = (session_end_time - session_start_time).total_seconds()

        if classifications > 0:
            avg_latency_ms = (total_latency / classifications) * 1000
            print("\n" + "=" * 60)
            print("SESSION ENDED")
            print("=" * 60)
            print(f"  Total classifications: {classifications}")
            print(f"  Average latency:       {avg_latency_ms:.1f} ms")
            print(f"  Total movements:       {cursor.total_movements}")
            print(f"  Total clicks:          {cursor.total_clicks}")
            print(f"  Total loop iterations: {loop_count}")
            print("=" * 60)
        else:
            avg_latency_ms = 0.0
            print("\nSession ended (no classifications performed).")

        # ------------------------------------------------------------------
        # 9. Save session stats to JSON
        # ------------------------------------------------------------------
        try:
            project_root = Path(__file__).parent.parent
            sessions_dir = project_root / "data" / "sessions"
            sessions_dir.mkdir(parents=True, exist_ok=True)

            session_filename = (
                f"session_{session_start_time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            session_path = sessions_dir / session_filename

            session_stats = {
                "start_time": session_start_time.isoformat(),
                "end_time": session_end_time.isoformat(),
                "duration_seconds": round(session_duration, 2),
                "total_classifications": classifications,
                "avg_latency_ms": round(avg_latency_ms, 2),
                "total_movements": cursor.total_movements,
                "total_clicks": cursor.total_clicks,
                "total_loop_iterations": loop_count,
                "model_used": str(model_path),
            }

            # Include SEAL adaptation stats if adaptation was enabled
            if adaptation_enabled:
                try:
                    seal_stats = seal_engine.get_stats()
                    session_stats["seal_stats"] = seal_stats
                except Exception:
                    session_stats["seal_stats"] = None
                if gflownet_optimizer is not None:
                    try:
                        session_stats["gflownet_stats"] = gflownet_optimizer.get_stats()
                    except Exception:
                        session_stats["gflownet_stats"] = None

            session_path.write_text(
                json.dumps(session_stats, indent=2, default=str) + "\n"
            )
            logger.info("Session stats saved to %s", session_path)
        except Exception:
            logger.exception("Failed to save session stats to JSON.")

        logger.info("EEG Cursor stopped.")


if __name__ == "__main__":
    main()
