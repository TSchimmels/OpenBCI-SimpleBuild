"""Surface Laplacian spatial filter for EEG.

Computes the second spatial derivative of the electric potential at each
electrode, acting as a high-pass spatial filter that sharpens the
topographic distribution and reduces volume conduction effects.

For motor imagery BCI with 16 channels, the Surface Laplacian provides
better spatial resolution than Common Average Reference (CAR) by
estimating the local current density at each electrode rather than
subtracting a global mean.

Two methods are provided:
1. **Finite Difference (nearest-neighbor):** Fast, works with any montage.
   Each electrode's Laplacian is computed as the difference between its
   value and the mean of its nearest neighbors.
2. **Spherical Spline (Perrin et al., 1989):** More accurate, requires
   3D electrode coordinates. Uses the Legendre polynomial expansion on
   a unit sphere.

References:
    Perrin, F., Pernier, J., Bertrand, O., & Echallier, J.F. (1989).
    Spherical splines for scalp potential and current density mapping.
    Electroencephalography and Clinical Neurophysiology, 72(2), 184-187.

    Hjorth, B. (1975). An on-line transformation of EEG scalp
    potentials into orthogonal source derivations. Electroencephalography
    and Clinical Neurophysiology, 39(5), 526-530.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Standard 10-20 electrode positions (x, y, z on unit sphere)
# Used when no custom positions are provided
ELECTRODE_POSITIONS_10_20: Dict[str, Tuple[float, float, float]] = {
    "Fp1": (-0.31, 0.95, 0.00), "Fp2": (0.31, 0.95, 0.00),
    "F7":  (-0.81, 0.59, 0.00), "F3":  (-0.55, 0.67, 0.50),
    "Fz":  (0.00, 0.71, 0.71),  "F4":  (0.55, 0.67, 0.50),
    "F8":  (0.81, 0.59, 0.00),  "T7":  (-1.00, 0.00, 0.00),
    "C3":  (-0.71, 0.00, 0.71), "Cz":  (0.00, 0.00, 1.00),
    "C4":  (0.71, 0.00, 0.71),  "T8":  (1.00, 0.00, 0.00),
    "P7":  (-0.81, -0.59, 0.00), "P3": (-0.55, -0.67, 0.50),
    "Pz":  (0.00, -0.71, 0.71), "P4":  (0.55, -0.67, 0.50),
    "P8":  (0.81, -0.59, 0.00), "O1":  (-0.31, -0.95, 0.00),
    "O2":  (0.31, -0.95, 0.00), "Oz":  (0.00, -1.00, 0.00),
    # Additional 10-20 positions for OpenBCI 16ch
    "FC3": (-0.63, 0.34, 0.71), "FC4": (0.63, 0.34, 0.71),
    "CP3": (-0.63, -0.34, 0.71), "CP4": (0.63, -0.34, 0.71),
    "FCz": (0.00, 0.36, 0.93), "CPz": (0.00, -0.36, 0.93),
}

# Default 16-channel OpenBCI Ultracortex Mark IV montage
# (Cyton channels 1-8 + Daisy channels 9-16)
DEFAULT_16CH_MONTAGE: List[str] = [
    "C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4", "Fz",
    "F3", "F4", "P3", "P4", "P7", "P8", "O1", "O2",
]


def surface_laplacian_fdn(
    data: np.ndarray,
    channel_names: Optional[List[str]] = None,
    n_neighbors: int = 4,
) -> np.ndarray:
    """Compute the Surface Laplacian via finite differences (nearest-neighbor).

    For each channel, subtracts the mean of its N nearest neighbors
    (in 3D electrode space). This is Hjorth's source derivation
    generalized to arbitrary montages.

    Args:
        data: EEG data, shape ``(n_channels, n_samples)``.
        channel_names: List of electrode names (10-20 system).
            Defaults to :data:`DEFAULT_16CH_MONTAGE` if None.
        n_neighbors: Number of nearest neighbors per channel (default 4).

    Returns:
        Laplacian-filtered data, same shape as input.
    """
    n_ch = data.shape[0]
    names = channel_names or DEFAULT_16CH_MONTAGE[:n_ch]

    # Get 3D positions
    positions = np.array([
        ELECTRODE_POSITIONS_10_20.get(name, (0, 0, 0))
        for name in names
    ], dtype=np.float64)

    # Compute distance matrix
    dist = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(n_ch):
            dist[i, j] = np.linalg.norm(positions[i] - positions[j])

    # For each channel, find N nearest neighbors (excluding self)
    out = np.empty_like(data)
    for i in range(n_ch):
        # Sort by distance, skip self (distance=0)
        neighbor_idx = np.argsort(dist[i])[1:n_neighbors + 1]
        neighbor_mean = data[neighbor_idx].mean(axis=0)
        out[i] = data[i] - neighbor_mean

    logger.debug(
        "Surface Laplacian (FDN): %d channels, %d neighbors each",
        n_ch, n_neighbors,
    )
    return out


def surface_laplacian_spline(
    data: np.ndarray,
    channel_names: Optional[List[str]] = None,
    m: int = 4,
    smoothing: float = 1e-5,
) -> np.ndarray:
    """Compute the Surface Laplacian via spherical splines (Perrin et al., 1989).

    More accurate than finite differences but requires 3D electrode
    coordinates. Uses the Legendre polynomial expansion to compute
    the G and H matrices on a unit sphere.

    Args:
        data: EEG data, shape ``(n_channels, n_samples)``.
        channel_names: List of electrode names (10-20 system).
        m: Order of the spline (default 4, recommended by Perrin).
        smoothing: Regularization parameter (default 1e-5).

    Returns:
        Laplacian-filtered data, same shape as input.
    """
    from scipy.special import legendre

    n_ch = data.shape[0]
    names = channel_names or DEFAULT_16CH_MONTAGE[:n_ch]

    # Get 3D positions on unit sphere
    positions = np.array([
        ELECTRODE_POSITIONS_10_20.get(name, (0, 0, 0))
        for name in names
    ], dtype=np.float64)

    # Normalize to unit sphere
    norms = np.linalg.norm(positions, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    positions = positions / norms

    # Compute cosine of angles between all electrode pairs
    cos_angles = positions @ positions.T
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    # Compute G matrix (interpolation) and H matrix (Laplacian)
    # using Legendre polynomial expansion
    n_terms = 50  # number of Legendre terms
    G = np.zeros((n_ch, n_ch))
    H = np.zeros((n_ch, n_ch))

    for n in range(1, n_terms + 1):
        Pn = legendre(n)
        Pn_vals = Pn(cos_angles)
        denom = (2.0 * n + 1.0)
        G += (1.0 / (n ** m * (n + 1) ** m)) * (2.0 * n + 1.0) / (4.0 * np.pi) * Pn_vals
        H += ((2.0 * n + 1.0) / (4.0 * np.pi)) * (
            (-n * (n + 1.0)) / (n ** m * (n + 1) ** m)
        ) * Pn_vals

    # Add smoothing regularization
    G += smoothing * np.eye(n_ch)

    # Solve for spline coefficients
    # G @ C = data  =>  C = G^{-1} @ data
    # Laplacian = H @ C = H @ G^{-1} @ data
    try:
        G_inv = np.linalg.inv(G)
        transform = H @ G_inv
    except np.linalg.LinAlgError:
        logger.warning(
            "Spherical spline G matrix singular — falling back to FDN method"
        )
        return surface_laplacian_fdn(data, channel_names)

    out = transform @ data

    logger.debug(
        "Surface Laplacian (spline): %d channels, m=%d, smoothing=%.1e",
        n_ch, m, smoothing,
    )
    return out
