"""Scalp topographic mapping for 16-channel EEG.

Generates 2D interpolated scalp maps showing spatial distribution of
EEG features (amplitude, power, ERDS%, r²) across the 10-20 electrode
montage. Uses matplotlib for rendering.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ======================================================================
# Standard 10-20 positions for 16-channel Cyton+Daisy
# Coordinates: (x, y) on a unit circle, nose pointing up (+y)
# ======================================================================

CHANNEL_NAMES_16 = [
    "Fp1", "Fp2", "C3", "C4", "P7", "P8",
    "O1", "O2", "F3", "F4", "T7", "T8",
    "P3", "P4", "Cz", "Fz",
]

# 2D projected positions (azimuthal equidistant from vertex)
CHANNEL_POS_16 = {
    "Fp1": (-0.31, 0.95),
    "Fp2": (0.31, 0.95),
    "F3":  (-0.42, 0.58),
    "Fz":  (0.00, 0.58),
    "F4":  (0.42, 0.58),
    "T7":  (-0.87, 0.00),
    "C3":  (-0.42, 0.00),
    "Cz":  (0.00, 0.00),
    "C4":  (0.42, 0.00),
    "T8":  (0.87, 0.00),
    "P7":  (-0.71, -0.48),
    "P3":  (-0.42, -0.48),
    "P4":  (0.42, -0.48),
    "P8":  (0.71, -0.48),
    "O1":  (-0.31, -0.85),
    "O2":  (0.31, -0.85),
}


class TopoMapper:
    """Generate scalp topographic maps for 16-channel EEG.

    Creates matplotlib figures showing interpolated spatial distributions
    of EEG values across the scalp. Supports electrode highlighting,
    head outline, and nose/ear markers.

    Args:
        channel_names: List of channel names in data order.
            Defaults to the standard 16-channel Cyton+Daisy layout.
        channel_positions: Dict mapping channel names to (x, y) positions.
            Defaults to standard 10-20 projections.
    """

    def __init__(
        self,
        channel_names: Optional[List[str]] = None,
        channel_positions: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.channel_names = channel_names or CHANNEL_NAMES_16
        self.channel_positions = channel_positions or CHANNEL_POS_16

        # Build position arrays in data order
        self._pos_x = np.array([
            self.channel_positions.get(name, (0, 0))[0]
            for name in self.channel_names
        ])
        self._pos_y = np.array([
            self.channel_positions.get(name, (0, 0))[1]
            for name in self.channel_names
        ])

        # Interpolation grid
        self._grid_res = 64
        xi = np.linspace(-1.1, 1.1, self._grid_res)
        yi = np.linspace(-1.1, 1.1, self._grid_res)
        self._grid_x, self._grid_y = np.meshgrid(xi, yi)

        # Head mask (circle)
        self._head_radius = 1.0
        self._head_mask = (
            self._grid_x ** 2 + self._grid_y ** 2
        ) <= (self._head_radius * 1.05) ** 2

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        """Interpolate channel values onto a 2D grid.

        Uses inverse-distance weighting for fast, smooth interpolation.

        Args:
            values: 1D array of values, one per channel, shape (n_channels,).

        Returns:
            2D grid of interpolated values, shape (grid_res, grid_res).
            Points outside the head are masked to NaN.
        """
        from scipy.interpolate import griddata

        points = np.column_stack([self._pos_x, self._pos_y])
        grid = griddata(
            points, values,
            (self._grid_x, self._grid_y),
            method="cubic",
            fill_value=0.0,
        )

        # Mask outside head
        grid[~self._head_mask] = np.nan

        return grid

    def plot(
        self,
        values: np.ndarray,
        ax=None,
        title: str = "",
        cmap: str = "RdBu_r",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_names: bool = True,
        colorbar: bool = True,
    ):
        """Plot a topographic map.

        Args:
            values: 1D array of values per channel.
            ax: Matplotlib axes. If None, creates a new figure.
            title: Plot title.
            cmap: Colormap name.
            vmin: Minimum colorbar value. If None, uses data min.
            vmax: Maximum colorbar value. If None, uses data max.
            show_names: Whether to label electrode positions.
            colorbar: Whether to add a colorbar.

        Returns:
            The matplotlib axes object.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        grid = self.interpolate(values)

        # Symmetric limits for diverging colormaps
        if vmin is None and vmax is None:
            abs_max = np.nanmax(np.abs(grid))
            if abs_max > 0:
                vmin, vmax = -abs_max, abs_max
            else:
                vmin, vmax = -1, 1

        # Plot interpolated surface
        im = ax.contourf(
            self._grid_x, self._grid_y, grid,
            levels=32, cmap=cmap, vmin=vmin, vmax=vmax,
            extend="both",
        )

        # Head outline
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(
            self._head_radius * np.cos(theta),
            self._head_radius * np.sin(theta),
            "k-", linewidth=2,
        )

        # Nose
        nose_x = [0.0, -0.08, 0.0, 0.08, 0.0]
        nose_y = [1.0, 1.12, 1.20, 1.12, 1.0]
        ax.plot(nose_x, nose_y, "k-", linewidth=2)

        # Ears
        for side in [-1, 1]:
            ear_x = side * np.array([0.96, 1.02, 1.06, 1.06, 1.02, 0.96])
            ear_y = np.array([0.18, 0.16, 0.08, -0.08, -0.16, -0.18])
            ax.plot(ear_x, ear_y, "k-", linewidth=1.5)

        # Electrode markers
        ax.scatter(
            self._pos_x, self._pos_y,
            s=40, c="black", zorder=5, edgecolors="white", linewidth=0.5,
        )

        # Channel labels
        if show_names:
            for name, x, y in zip(self.channel_names, self._pos_x, self._pos_y):
                ax.annotate(
                    name, (x, y),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="black",
                    fontweight="bold",
                )

        if colorbar:
            plt.colorbar(im, ax=ax, shrink=0.6, label="Value")

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.2, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold")

        return ax

    def plot_erds_topo(
        self,
        erds_values: np.ndarray,
        ax=None,
        title: str = "ERDS%",
    ):
        """Plot ERDS% topographic map with ERD-specific colormap.

        Blue = ERD (desynchronization, negative), Red = ERS (synchronization, positive).

        Args:
            erds_values: ERDS% per channel, shape (n_channels,).
            ax: Matplotlib axes.
            title: Plot title.

        Returns:
            Axes object.
        """
        return self.plot(
            erds_values, ax=ax, title=title,
            cmap="RdBu_r", show_names=True, colorbar=True,
        )

    def get_channel_index(self, name: str) -> int:
        """Get the data index for a named channel."""
        return self.channel_names.index(name)

    def __repr__(self) -> str:
        return f"TopoMapper(channels={len(self.channel_names)})"
