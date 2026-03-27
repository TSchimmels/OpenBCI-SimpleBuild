"""BrainFlow data acquisition wrapper.

Provides BoardManager, a high-level wrapper around BrainFlow's BoardShim
for EEG data acquisition.  Supports both real hardware (OpenBCI Cyton/Daisy)
and the synthetic board for development and testing.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowError, BrainFlowInputParams, BoardIds

logger = logging.getLogger(__name__)


class BoardManager:
    """Manage a BrainFlow board session.

    Wraps BoardShim with connection lifecycle management, automatic
    fallback to the synthetic board, and context-manager support.

    Args:
        config: Configuration dictionary (typically loaded from
            ``config/settings.yaml``).  Relevant keys live under the
            ``board`` section: ``board_id``, ``serial_port``,
            ``sampling_rate_override``, ``channel_count``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Dict) -> None:
        board_cfg = config.get("board", {})

        board_id: int = board_cfg.get("board_id", BoardIds.SYNTHETIC_BOARD)
        serial_port: str = board_cfg.get("serial_port", "")
        self._sampling_rate_override: Optional[int] = board_cfg.get(
            "sampling_rate_override", None
        )
        self._channel_count: int = board_cfg.get("channel_count", 16)

        # Fall back to synthetic board when no real hardware is configured.
        if board_id == -1 or not serial_port:
            board_id = BoardIds.SYNTHETIC_BOARD
            serial_port = ""
            logger.info(
                "No hardware configured (board_id=%s, serial_port='%s'); "
                "using SYNTHETIC_BOARD.",
                board_cfg.get("board_id", -1),
                board_cfg.get("serial_port", ""),
            )

        self._board_id: int = int(board_id)
        self._serial_port: str = serial_port
        self._connected: bool = False

        # Build BrainFlow params and board instance.
        params = BrainFlowInputParams()
        if self._serial_port:
            params.serial_port = self._serial_port

        self._board: BoardShim = BoardShim(self._board_id, params)
        logger.debug(
            "BoardManager initialised: board_id=%d, serial_port='%s'",
            self._board_id,
            self._serial_port,
        )

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Prepare the board session and start the data stream.

        Raises:
            BrainFlowError: If the board cannot be prepared or started.
        """
        if self._connected:
            logger.warning("Board is already connected; skipping connect().")
            return

        try:
            logger.info("Preparing board session (board_id=%d)...", self._board_id)
            self._board.prepare_session()
            self._board.start_stream(45000)
            self._connected = True
            logger.info(
                "Board streaming at %d Hz on %d EEG channel(s).",
                self.get_sampling_rate(),
                len(self.get_eeg_channels()),
            )
        except BrainFlowError:
            logger.exception("Failed to connect to board.")
            raise

    def disconnect(self) -> None:
        """Stop the data stream and release the board session.

        Safe to call even if the board is not connected; errors during
        teardown are logged but not re-raised so that cleanup is
        best-effort.
        """
        if not self._connected:
            logger.debug("Board is not connected; nothing to disconnect.")
            return

        try:
            self._board.stop_stream()
            logger.debug("Stream stopped.")
        except BrainFlowError:
            logger.exception("Error stopping board stream.")

        try:
            self._board.release_session()
            logger.debug("Session released.")
        except BrainFlowError:
            logger.exception("Error releasing board session.")

        self._connected = False
        logger.info("Board disconnected.")

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_data(self, n_samples: int) -> np.ndarray:
        """Return the most recent *n_samples* from the ring buffer.

        The ring buffer is **not** cleared — the same samples may be
        returned on subsequent calls until they fall out of the buffer.

        Args:
            n_samples: Number of most-recent samples to retrieve.

        Returns:
            2-D array of shape ``(channels, samples)``.  If
            *n_samples* is <= 0 or the board returns fewer samples
            than requested, a warning is logged and the available
            data (possibly empty) is returned.

        Raises:
            BrainFlowError: If the board is not streaming.
        """
        if n_samples <= 0:
            logger.warning(
                "get_data() called with n_samples=%d; returning empty array.",
                n_samples,
            )
            n_channels = len(self.get_eeg_channels())
            return np.empty((n_channels, 0), dtype=np.float64)

        try:
            data: np.ndarray = self._board.get_current_board_data(n_samples)
        except BrainFlowError:
            if self._connected:
                # Board may have disconnected mid-stream
                logger.exception(
                    "Failed to get board data; marking board as disconnected."
                )
                self._connected = False
            else:
                logger.exception("Failed to get current board data.")
            raise

        # Guard: board returned empty data
        if data.size == 0 or data.shape[-1] == 0:
            logger.warning(
                "Board returned empty data (shape %s). Buffer may be empty.",
                data.shape,
            )
            return data

        # Guard: board returned fewer samples than requested
        actual = data.shape[-1]
        if actual < n_samples:
            logger.warning(
                "Requested %d samples but board returned only %d. "
                "Buffer may not have accumulated enough data yet.",
                n_samples,
                actual,
            )

        return data

    def get_board_data(self) -> np.ndarray:
        """Return **all** data accumulated in the ring buffer.

        This call **clears** the buffer — subsequent calls will only
        return data collected after this point.

        Returns:
            2-D array of shape ``(channels, samples)``.

        Raises:
            BrainFlowError: If the board is not streaming.
        """
        try:
            data: np.ndarray = self._board.get_board_data()
        except BrainFlowError:
            if self._connected:
                logger.exception(
                    "Failed to get board data; marking board as disconnected."
                )
                self._connected = False
            else:
                logger.exception("Failed to get board data.")
            raise

        if data.size == 0 or data.shape[-1] == 0:
            logger.warning(
                "get_board_data() returned empty data (shape %s).",
                data.shape,
            )

        return data

    # ------------------------------------------------------------------
    # Board metadata helpers
    # ------------------------------------------------------------------

    def get_sampling_rate(self) -> int:
        """Return the effective sampling rate in Hz.

        If ``sampling_rate_override`` is set in the config it takes
        precedence; otherwise the value is queried from BrainFlow for
        the active board id.
        """
        if self._sampling_rate_override is not None:
            return int(self._sampling_rate_override)
        return int(BoardShim.get_sampling_rate(self._board_id))

    def get_eeg_channels(self) -> List[int]:
        """Return the list of EEG channel indices for this board."""
        return list(BoardShim.get_eeg_channels(self._board_id))

    def get_board_id(self) -> int:
        """Return the BrainFlow board id in use."""
        return self._board_id

    def is_synthetic(self) -> bool:
        """Return ``True`` if the synthetic (simulated) board is active."""
        return self._board_id == BoardIds.SYNTHETIC_BOARD

    @property
    def is_connected(self) -> bool:
        """Whether the board session is currently active and streaming."""
        return self._connected

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "BoardManager":
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        self.disconnect()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return (
            f"BoardManager(board_id={self._board_id}, "
            f"serial_port='{self._serial_port}', "
            f"status={status})"
        )
