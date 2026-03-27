"""Artifact detection and epoch rejection for EEG data.

Provides functions to identify and remove noisy epochs and bad channels
before feature extraction. Designed for the Motor-Imagery BCI pipeline.
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def reject_epochs(
    epochs: np.ndarray,
    labels: np.ndarray,
    threshold_uv: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """Reject epochs where any channel exceeds a peak-to-peak threshold.

    Peak-to-peak amplitude is defined as ``max - min`` within each
    channel of each epoch.  An epoch is rejected if *any* of its
    channels exceeds *threshold_uv*.

    Args:
        epochs: Epoched EEG data, shape
            ``(n_epochs, n_channels, n_samples)``.
        labels: Class labels corresponding to each epoch, shape
            ``(n_epochs,)``.
        threshold_uv: Maximum allowed peak-to-peak amplitude in
            microvolts.  Epochs exceeding this on any channel are
            rejected.  Default 100.

    Returns:
        A 3-tuple ``(clean_epochs, clean_labels, rejected_indices)``
        where:

        - **clean_epochs** -- retained epochs, shape
          ``(n_clean, n_channels, n_samples)``.
        - **clean_labels** -- labels for the retained epochs, shape
          ``(n_clean,)``.
        - **rejected_indices** -- list of integer indices of the
          rejected epochs (relative to the input array).

    Raises:
        ValueError: If *epochs* is not 3-D or *labels* length does not
            match the number of epochs.
    """
    if epochs.ndim != 3:
        raise ValueError(
            f"epochs must be 3-D (n_epochs, n_channels, n_samples), "
            f"got shape {epochs.shape}."
        )
    if labels.shape[0] != epochs.shape[0]:
        raise ValueError(
            f"labels length ({labels.shape[0]}) must match the number "
            f"of epochs ({epochs.shape[0]})."
        )

    # Guard: 0 epochs — nothing to reject
    if epochs.shape[0] == 0:
        logger.warning("reject_epochs called with 0 epochs; returning empty.")
        return epochs, labels, []

    # Guard: 0 channels or 0 samples — ptp is meaningless
    if epochs.shape[1] == 0 or epochs.shape[2] == 0:
        logger.warning(
            "reject_epochs: epochs have 0 channels or 0 samples (shape %s). "
            "Returning all epochs unmodified.",
            epochs.shape,
        )
        return epochs, labels, []

    # Guard: NaN values corrupt ptp calculation
    if np.any(np.isnan(epochs)):
        logger.warning(
            "NaN values detected in epochs; replacing with 0.0 "
            "before artifact rejection."
        )
        epochs = np.nan_to_num(epochs, nan=0.0)

    # Peak-to-peak per channel per epoch: (n_epochs, n_channels)
    ptp = epochs.max(axis=2) - epochs.min(axis=2)

    # An epoch is bad if any channel exceeds the threshold
    epoch_max_ptp = ptp.max(axis=1)  # (n_epochs,)
    bad_mask = epoch_max_ptp > threshold_uv

    rejected_indices: List[int] = np.where(bad_mask)[0].tolist()
    good_mask = ~bad_mask

    clean_epochs = epochs[good_mask]
    clean_labels = labels[good_mask]

    if clean_epochs.shape[0] == 0:
        logger.warning(
            "ALL %d epochs were rejected (threshold=%.1f uV). "
            "Downstream code will receive empty arrays. Consider "
            "increasing the artifact threshold or checking electrode "
            "impedances.",
            epochs.shape[0],
            threshold_uv,
        )

    return clean_epochs, clean_labels, rejected_indices


def detect_bad_channels(
    data: np.ndarray,
    threshold_std: float = 3.0,
) -> List[int]:
    """Detect bad channels based on statistical outliers.

    A channel is flagged as *bad* if:

    1. Its standard deviation exceeds ``threshold_std * median_std``
       (excessively noisy), **or**
    2. Its standard deviation is below ``0.1 * median_std``
       (flatline / dead channel).

    Args:
        data: Continuous EEG data, shape ``(n_channels, n_samples)``.
        threshold_std: Multiplier for the median channel standard
            deviation.  Channels whose std exceeds
            ``threshold_std * median_std`` are flagged.  Default 3.

    Returns:
        Sorted list of integer indices of the bad channels.

    Raises:
        ValueError: If *data* is not 2-D.
    """
    if data.ndim != 2:
        raise ValueError(
            f"data must be 2-D (n_channels, n_samples), "
            f"got shape {data.shape}."
        )

    # Guard: empty data
    if data.shape[0] == 0 or data.shape[1] == 0:
        logger.warning(
            "detect_bad_channels: data has 0 channels or 0 samples "
            "(shape %s). Returning empty list.",
            data.shape,
        )
        return []

    # Guard: NaN values corrupt std calculation
    if np.any(np.isnan(data)):
        logger.warning(
            "NaN values detected in detect_bad_channels input; "
            "replacing with 0.0 before analysis."
        )
        data = np.nan_to_num(data, nan=0.0)

    channel_stds = data.std(axis=1)  # (n_channels,)
    median_std = np.median(channel_stds)

    # Guard against the degenerate case where median_std is zero
    # (e.g. all channels are constant). In that case every channel
    # with any variance at all looks noisy, so flag none.
    if median_std == 0:
        return []

    high_threshold = threshold_std * median_std
    low_threshold = 0.1 * median_std

    bad_indices: List[int] = []
    for ch_idx, std in enumerate(channel_stds):
        if std > high_threshold or std < low_threshold:
            bad_indices.append(ch_idx)

    return sorted(bad_indices)
