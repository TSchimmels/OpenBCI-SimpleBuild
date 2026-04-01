"""EEG data recorder with event markers.

Records continuous EEG data from a :class:`~src.acquisition.board.BoardManager`
and stores time-locked event markers for later epoch extraction.  Supports
saving and loading sessions as NumPy ``.npz`` archives.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..acquisition.board import BoardManager

logger = logging.getLogger(__name__)


class DataRecorder:
    """Record EEG data with event markers for offline analysis.

    Works in tandem with :class:`~src.acquisition.board.BoardManager`:

    1. Call :meth:`start` to begin a recording session.
    2. The training paradigm calls :meth:`add_event` at each cue onset.
    3. Call :meth:`stop` to end the session and retrieve all data.
    4. Optionally :meth:`save` to disk for later use.

    Args:
        board_manager: A connected :class:`BoardManager` instance that
            is already streaming data.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, board_manager: "BoardManager") -> None:
        self._board: "BoardManager" = board_manager
        self._events: List[Dict] = []
        self._start_time: Optional[float] = None
        self._recording: bool = False
        self._last_raw_data: Optional[np.ndarray] = None
        self._last_events: Optional[List[Dict]] = None
        self._accumulated_chunks: List[np.ndarray] = []

        logger.debug("DataRecorder created (board=%r).", board_manager)

    # ------------------------------------------------------------------
    # Recording lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin recording.

        Stores the current wall-clock time as the session reference and
        clears the board's ring buffer so that only data from this point
        forward is captured.
        """
        # Flush the ring buffer so we start with a clean slate
        self._board.get_board_data()

        self._events = []
        self._accumulated_chunks = []
        self._start_time = time.time()
        self._recording = True

        logger.info("Recording started at %.3f.", self._start_time)

    def drain(self) -> int:
        """Drain the BrainFlow ring buffer into Python-side accumulator.

        MUST be called periodically during long recordings to prevent
        the 45,000-sample ring buffer from overflowing (6 minutes at
        125 Hz). The Graz paradigm calls this after each trial.

        Returns:
            Number of new samples drained.
        """
        if not self._recording:
            return 0
        chunk = self._board.get_board_data()  # destructive read
        n_new = chunk.shape[1] if chunk.ndim == 2 else 0
        if n_new > 0:
            self._accumulated_chunks.append(chunk)
        return n_new

    def add_event(self, label: str) -> None:
        """Record an event marker at the current moment.

        The event stores the label, the wall-clock timestamp, and an
        *approximate* sample index computed from the elapsed time and
        the board's sampling rate.  The sample index is approximate
        because it relies on wall-clock timing rather than the board's
        hardware clock; for most practical purposes (epoch extraction
        with generous windows) this is more than sufficient.

        Args:
            label: Human-readable class label for this event
                (e.g. ``'left_hand'``, ``'right_hand'``, ``'rest'``).

        Raises:
            RuntimeError: If recording has not been started.
        """
        if not self._recording or self._start_time is None:
            raise RuntimeError(
                "Cannot add event: recording has not been started. "
                "Call start() first."
            )

        now = time.time()
        elapsed = now - self._start_time
        sf = self._board.get_sampling_rate()
        sample_idx = int(elapsed * sf)

        event = {
            "label": label,
            "timestamp": now,
            "sample_index": sample_idx,
        }
        self._events.append(event)

        logger.debug(
            "Event '%s' at %.3f s (sample ~%d).",
            label,
            elapsed,
            sample_idx,
        )

    def stop(self) -> Tuple[np.ndarray, List[Dict]]:
        """Stop recording and return collected data.

        Retrieves all data accumulated in the board's ring buffer since
        :meth:`start` was called, then resets internal state.

        Returns:
            A 2-tuple ``(raw_data, events)`` where:

            - **raw_data** -- 2-D NumPy array of shape
              ``(n_channels, n_samples)`` containing all channels
              (EEG + auxiliary) as returned by BrainFlow.
            - **events** -- list of event dictionaries, each with keys
              ``'label'``, ``'timestamp'``, and ``'sample_index'``.

        Raises:
            RuntimeError: If recording has not been started.
        """
        if not self._recording:
            raise RuntimeError(
                "Cannot stop: recording is not active. Call start() first."
            )

        # Final drain of remaining buffer data
        final_chunk = self._board.get_board_data()
        if final_chunk.ndim == 2 and final_chunk.shape[1] > 0:
            self._accumulated_chunks.append(final_chunk)

        # Combine all accumulated chunks into one continuous array
        if self._accumulated_chunks:
            raw_data = np.hstack(self._accumulated_chunks)
        else:
            raw_data = final_chunk

        events = list(self._events)
        self._recording = False

        # Cache data so save() can access it after stop()
        self._last_raw_data = raw_data
        self._last_events = events
        self._accumulated_chunks = []  # free memory

        logger.info(
            "Recording stopped. Captured %d samples (%d chunks), %d events.",
            raw_data.shape[1] if raw_data.ndim == 2 else 0,
            len(self._accumulated_chunks) + 1,
            len(events),
        )

        return raw_data, events

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the most recently recorded session to a ``.npz`` file.

        The archive contains:

        - ``data`` -- the raw 2-D data array.
        - ``events_json`` -- the events list serialised as a JSON string.

        If the recording has not been stopped yet, this method stops it
        first and caches the result.

        Args:
            path: Destination file path (should end in ``.npz``).
        """
        if self._recording:
            logger.warning(
                "save() called while still recording. "
                "Stopping recording first."
            )
            raw_data, events = self.stop()
        else:
            if self._last_raw_data is not None:
                raw_data = self._last_raw_data
                events = self._last_events or list(self._events)
            else:
                logger.warning("No cached data from stop(). Data may be empty.")
                raw_data = self._board.get_board_data()
                events = list(self._events)

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sf = self._board.get_sampling_rate()
        eeg_channels = self._board.get_eeg_channels()

        events_json = json.dumps(events)
        np.savez(str(out_path), data=raw_data, events_json=events_json,
                 sf=np.array(sf), eeg_channels=np.array(eeg_channels))

        logger.info("Session saved to %s.", out_path)

    @staticmethod
    def load(path: str) -> Tuple[np.ndarray, List[Dict], Dict]:
        """Load a previously saved recording session.

        Args:
            path: Path to a ``.npz`` file created by :meth:`save`.

        Returns:
            A 3-tuple ``(raw_data, events, metadata)`` where:

            - **raw_data** -- 2-D NumPy array as saved by :meth:`save`.
            - **events** -- list of event dictionaries.
            - **metadata** -- dictionary with optional keys ``'sf'``
              (sampling frequency, int) and ``'eeg_channels'`` (list of
              int).  Empty dict for archives saved before this feature.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError: If the archive does not contain the expected keys.
        """
        archive = np.load(path, allow_pickle=False)
        raw_data = archive["data"]
        events = json.loads(str(archive["events_json"]))

        metadata: Dict = {}
        if "sf" in archive:
            metadata["sf"] = int(archive["sf"])
        if "eeg_channels" in archive:
            metadata["eeg_channels"] = archive["eeg_channels"].tolist()

        logger.info(
            "Loaded session from %s: %d samples, %d events, metadata=%s.",
            path,
            raw_data.shape[1] if raw_data.ndim == 2 else 0,
            len(events),
            metadata,
        )

        return raw_data, events, metadata

    # ------------------------------------------------------------------
    # Epoch extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_epochs(
        raw_data: np.ndarray,
        events: List[Dict],
        sf: int,
        tmin: float = 0.0,
        tmax: float = 4.0,
        eeg_channels: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """Cut fixed-length epochs from continuous data.

        Each epoch is time-locked to an event's ``sample_index`` and spans
        from ``tmin`` to ``tmax`` seconds relative to that marker.

        Args:
            raw_data: Continuous recording, shape
                ``(n_total_channels, n_samples)``.
            events: List of event dictionaries (as returned by
                :meth:`stop` or :meth:`load`).  Each must contain at
                least ``'label'`` and ``'sample_index'`` keys.
            sf: Sampling frequency in Hz.
            tmin: Epoch start relative to event onset, in seconds.
                Default 0.0 (event onset).
            tmax: Epoch end relative to event onset, in seconds.
                Default 4.0.
            eeg_channels: List of channel indices to extract.  If
                ``None``, all channels in *raw_data* are used.

        Returns:
            A 3-tuple ``(epochs, labels, label_map)`` where:

            - **epochs** -- 3-D array of shape
              ``(n_epochs, n_channels, n_samples)`` containing the
              extracted epochs.
            - **labels** -- 1-D integer array of shape ``(n_epochs,)``
              with class labels encoded as integers.  The mapping is
              determined by the sorted unique set of label strings in
              *events*.
            - **label_map** -- dictionary mapping label strings to
              integer indices (e.g. ``{"left_hand": 0, "rest": 1}``).

        Notes:
            Events whose epoch window extends beyond the boundaries of
            *raw_data* are silently skipped.
        """
        if raw_data.ndim != 2:
            raise ValueError(
                f"raw_data must be 2-D (n_channels, n_samples), "
                f"got shape {raw_data.shape}."
            )

        # Select channels
        if eeg_channels is not None:
            data = raw_data[eeg_channels, :]
        else:
            data = raw_data

        n_channels = data.shape[0]
        total_samples = data.shape[1]

        # Epoch sample boundaries relative to event onset
        offset_start = int(round(tmin * sf))
        offset_end = int(round(tmax * sf))
        epoch_len = offset_end - offset_start

        # Build label-to-integer mapping from sorted unique labels
        unique_labels = sorted(set(ev["label"] for ev in events))
        label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}

        epochs_list: List[np.ndarray] = []
        labels_list: List[int] = []

        for ev in events:
            sample_idx = ev["sample_index"]
            start = sample_idx + offset_start
            end = sample_idx + offset_end

            # Skip if epoch falls outside data boundaries
            if start < 0 or end > total_samples:
                logger.warning(
                    "Skipping event '%s' at sample %d: epoch [%d, %d) "
                    "exceeds data range [0, %d).",
                    ev["label"],
                    sample_idx,
                    start,
                    end,
                    total_samples,
                )
                continue

            epoch = data[:, start:end].copy()
            epochs_list.append(epoch)
            labels_list.append(label_map[ev["label"]])

        if len(epochs_list) == 0:
            logger.warning("No valid epochs extracted.")
            return (
                np.empty((0, n_channels, epoch_len), dtype=raw_data.dtype),
                np.empty((0,), dtype=np.int64),
                label_map,
            )

        epochs = np.stack(epochs_list, axis=0)  # (n_epochs, n_channels, n_samples)
        labels = np.array(labels_list, dtype=np.int64)

        logger.info(
            "Extracted %d epochs (%d channels, %d samples each). "
            "Label mapping: %s.",
            epochs.shape[0],
            n_channels,
            epoch_len,
            label_map,
        )

        return epochs, labels, label_map

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        """Whether the recorder is currently active."""
        return self._recording

    @property
    def n_events(self) -> int:
        """Number of events recorded so far."""
        return len(self._events)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "recording" if self._recording else "idle"
        return (
            f"DataRecorder(status={status}, events={len(self._events)}, "
            f"board={self._board!r})"
        )
