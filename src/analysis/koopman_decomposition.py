"""
Koopman Spectral Decomposition for EEG

Replaces fixed frequency bands (alpha, beta, mu, etc.) with data-driven
oscillatory modes extracted via Dynamic Mode Decomposition (DMD).

The Koopman operator linearizes nonlinear dynamics in a lifted function space.
For EEG signals, this yields:
  - Koopman eigenvalues: encode frequency (imaginary part) and growth/decay rate (real part)
  - Koopman eigenfunctions: encode spatial patterns across channels

Reference:
    Brunton, S. L., Brunton, B. W., Proctor, J. L., & Kutz, J. N. (2016).
    "Extracting spatial-temporal coherent patterns in large-scale neural
    recordings using dynamic mode decomposition." Journal of Neuroscience
    Methods, 258, 1-15.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.linalg import svd, eig, pinv

logger = logging.getLogger(__name__)


class KoopmanEEGDecomposition:
    """
    Koopman spectral decomposition of multi-channel EEG via DMD.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    sf : float
        Sampling frequency in Hz.
    n_modes : int
        Number of dominant Koopman modes to retain (sorted by amplitude).
    delay_embedding_dim : int
        Number of time-delay copies for Takens embedding. Higher values
        enrich the observable space but increase computational cost.
    svd_rank : int or None
        Truncation rank for the SVD step. If None, uses n_modes.
    """

    def __init__(
        self,
        n_channels: int,
        sf: float,
        n_modes: int = 10,
        delay_embedding_dim: int = 5,
        svd_rank: Optional[int] = None,
    ):
        self.n_channels = n_channels
        self.sf = sf
        self.dt = 1.0 / sf
        self.n_modes = n_modes
        self.delay_dim = delay_embedding_dim
        self.svd_rank = svd_rank if svd_rank is not None else n_modes

        # Fitted state
        self._eigenvalues: Optional[np.ndarray] = None
        self._modes: Optional[np.ndarray] = None
        self._amplitudes: Optional[np.ndarray] = None
        self._frequencies: Optional[np.ndarray] = None
        self._growth_rates: Optional[np.ndarray] = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_delay_embedding(self, data: np.ndarray) -> np.ndarray:
        """
        Construct a time-delay embedded matrix from multi-channel data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)

        Returns
        -------
        X_embedded : ndarray, shape (n_channels * delay_dim, n_snapshots)
            Each column is the concatenation [x(t), x(t-1), ..., x(t-k+1)].
        """
        n_ch, n_samples = data.shape
        n_snapshots = n_samples - self.delay_dim + 1
        if n_snapshots < 2:
            raise ValueError(
                f"Not enough samples ({n_samples}) for delay embedding "
                f"dimension {self.delay_dim}. Need at least {self.delay_dim + 1}."
            )

        X = np.zeros((n_ch * self.delay_dim, n_snapshots))
        for k in range(self.delay_dim):
            row_start = k * n_ch
            row_end = row_start + n_ch
            X[row_start:row_end, :] = data[
                :, (self.delay_dim - 1 - k):(self.delay_dim - 1 - k + n_snapshots)
            ]
        return X

    def _compute_dmd(self, X: np.ndarray) -> None:
        """
        Exact DMD on the delay-embedded snapshot matrix.

        Steps:
            1. Split X into X1 = X[:, :-1], X2 = X[:, 1:]
            2. SVD of X1: X1 = U Sigma V*
            3. Project onto POD basis: A_tilde = U* X2 V Sigma^{-1}
            4. Eigendecompose A_tilde -> eigenvalues lam, eigenvectors w
            5. Koopman modes: Phi = X2 V Sigma^{-1} w
            6. Extract frequencies and growth rates from continuous-time eigenvalues.
        """
        X1 = X[:, :-1]
        X2 = X[:, 1:]

        # Step 2: truncated SVD
        U, s, Vh = svd(X1, full_matrices=False)
        rank = min(self.svd_rank, len(s))
        U_r = U[:, :rank]
        s_r = s[:rank]
        V_r = Vh[:rank, :].conj().T  # shape (n_snapshots-1, rank)

        S_inv = np.diag(1.0 / s_r)

        # Step 3: reduced-order linear operator
        A_tilde = U_r.conj().T @ X2 @ V_r @ S_inv  # (rank, rank)

        # Step 4: eigendecompose
        eigenvalues, w = eig(A_tilde)

        # Step 5: exact DMD modes (project back to full state space)
        Phi = X2 @ V_r @ S_inv @ w  # (state_dim, rank)

        # Step 6: continuous-time eigenvalues
        # lambda_c = log(lambda_d) / dt
        # Protect against zero eigenvalues
        safe_eigs = np.where(np.abs(eigenvalues) > 1e-15, eigenvalues, 1e-15)
        lambda_c = np.log(safe_eigs.astype(complex)) / self.dt

        frequencies = np.imag(lambda_c) / (2.0 * np.pi)  # Hz
        growth_rates = np.real(lambda_c)  # 1/s

        # Amplitudes: use the norm of each mode column
        amplitudes = np.abs(np.linalg.norm(Phi, axis=0))

        # Sort by amplitude descending, keep top n_modes
        sort_idx = np.argsort(amplitudes)[::-1][:self.n_modes]

        self._eigenvalues = eigenvalues[sort_idx]
        self._modes = Phi[:, sort_idx]
        self._amplitudes = amplitudes[sort_idx]
        self._frequencies = frequencies[sort_idx]
        self._growth_rates = growth_rates[sort_idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, data: np.ndarray) -> "KoopmanEEGDecomposition":
        """
        Fit the Koopman decomposition to multi-channel EEG data.

        Parameters
        ----------
        data : ndarray, shape (n_channels, n_samples)
            Continuous EEG time series (channels x time).

        Returns
        -------
        self
        """
        if data.ndim != 2:
            raise ValueError(
                f"Expected 2-D array (n_channels, n_samples), got shape {data.shape}"
            )
        if data.shape[0] != self.n_channels:
            raise ValueError(
                f"Channel count mismatch: expected {self.n_channels}, "
                f"got {data.shape[0]}"
            )

        logger.info(
            "Fitting Koopman decomposition: %d channels, %d samples, delay_dim=%d",
            data.shape[0], data.shape[1], self.delay_dim,
        )

        X = self._build_delay_embedding(data)
        self._compute_dmd(X)
        self._is_fitted = True

        logger.info(
            "Fitted %d modes. Top frequency: %.2f Hz, top amplitude: %.4f",
            self.n_modes,
            self._frequencies[0],
            self._amplitudes[0],
        )
        return self

    def get_modes(self) -> List[Dict]:
        """
        Retrieve the extracted Koopman modes sorted by amplitude.

        Returns
        -------
        modes : list of dict
            Each dict contains:
            - 'frequency_hz': oscillation frequency in Hz
            - 'growth_rate': exponential growth/decay rate (1/s).
              Negative means decaying (stable), positive means growing.
            - 'amplitude': mode energy (L2 norm of the mode vector)
            - 'spatial_pattern': ndarray of shape (n_channels,) -- the
              magnitude of the mode projected back to channel space
              (first n_channels components of the full mode vector).
        """
        self._check_fitted()

        modes = []
        for i in range(len(self._eigenvalues)):
            spatial = np.abs(self._modes[:self.n_channels, i])
            modes.append({
                "frequency_hz": float(np.abs(self._frequencies[i])),
                "growth_rate": float(self._growth_rates[i]),
                "amplitude": float(self._amplitudes[i]),
                "spatial_pattern": spatial,
            })
        return modes

    def get_subject_mu_band(self) -> Tuple[float, float]:
        """
        Find the subject's actual mu rhythm frequency from the strongest
        mode in the 6-15 Hz range and return (center_freq, bandwidth).

        The standard mu band is 8-12 Hz, but individual subjects can
        differ by 1-2 Hz. This method uses the data-driven Koopman
        decomposition to locate the true peak.

        Returns
        -------
        center_freq : float
            Peak mu frequency in Hz.
        bandwidth : float
            Estimated bandwidth (half-power width) in Hz, derived from
            the growth rate of the mode (faster decay = broader peak).
        """
        self._check_fitted()

        mu_candidates = []
        for i, freq in enumerate(self._frequencies):
            abs_freq = abs(freq)
            if 6.0 <= abs_freq <= 15.0:
                mu_candidates.append((i, abs_freq, self._amplitudes[i]))

        if not mu_candidates:
            logger.warning(
                "No Koopman modes found in 6-15 Hz range. "
                "Falling back to default 10 Hz +/- 2 Hz."
            )
            return (10.0, 2.0)

        # Pick the strongest candidate
        best = max(mu_candidates, key=lambda c: c[2])
        idx, center_freq, _ = best

        # Bandwidth from growth rate: larger |growth_rate| = faster decay = broader peak
        # Empirical mapping: bandwidth ~ |growth_rate| / pi
        growth = abs(self._growth_rates[idx])
        bandwidth = max(growth / np.pi, 0.5)  # floor at 0.5 Hz
        bandwidth = min(bandwidth, 4.0)  # cap at 4.0 Hz

        logger.info(
            "Subject mu band: %.2f Hz +/- %.2f Hz (mode %d)",
            center_freq, bandwidth, idx,
        )
        return (float(center_freq), float(bandwidth))

    def reconstruct(self, mode_indices: List[int]) -> np.ndarray:
        """
        Reconstruct a time series from selected Koopman modes.

        Parameters
        ----------
        mode_indices : list of int
            Indices into the sorted mode list (0 = most energetic).

        Returns
        -------
        reconstructed : ndarray, shape (n_channels, n_time)
            Reconstructed signal using only the selected modes.
            n_time depends on the original data length and delay embedding.
        """
        self._check_fitted()

        if not mode_indices:
            raise ValueError("mode_indices must be non-empty")

        for idx in mode_indices:
            if idx < 0 or idx >= len(self._eigenvalues):
                raise IndexError(
                    f"Mode index {idx} out of range "
                    f"[0, {len(self._eigenvalues) - 1}]"
                )

        # Reconstruct in the delay-embedded space, then extract the
        # first n_channels rows (corresponding to x(t) at the latest delay).
        n_time = self._modes.shape[0] // (self.n_channels * self.delay_dim) \
            if self._modes.shape[0] % (self.n_channels * self.delay_dim) == 0 \
            else self._modes.shape[0] // self.n_channels

        # Build time dynamics: each mode evolves as b_k * lambda_k^t
        # Use initial amplitudes from projection
        Phi_sel = self._modes[:, mode_indices]
        lam_sel = self._eigenvalues[mode_indices]

        # Estimate initial amplitudes via pseudoinverse
        b = np.ones(len(mode_indices), dtype=complex)

        # Build the Vandermonde matrix for time evolution
        n_steps = max(n_time, 100)
        vander = np.zeros((len(mode_indices), n_steps), dtype=complex)
        for t in range(n_steps):
            vander[:, t] = lam_sel ** t * b

        # Full reconstruction in embedded space
        X_recon = Phi_sel @ vander

        # Extract the top n_channels rows (latest time in delay stack)
        reconstructed = np.real(X_recon[:self.n_channels, :])
        return reconstructed

    def compute_erds_koopman(
        self,
        epoch: np.ndarray,
        baseline_epoch: np.ndarray,
    ) -> Dict:
        """
        Compute Event-Related Desynchronization/Synchronization using
        Koopman mode amplitudes instead of FFT band power.

        Parameters
        ----------
        epoch : ndarray, shape (n_channels, n_samples)
            EEG epoch during the event (e.g., motor imagery).
        baseline_epoch : ndarray, shape (n_channels, n_samples)
            EEG epoch from the baseline (rest) period.

        Returns
        -------
        result : dict
            - 'erds_percent': ndarray (n_channels,) -- ERDS% per channel
              for the mu band. Negative = desynchronization (ERD),
              positive = synchronization (ERS).
            - 'mu_center': float -- detected mu center frequency
            - 'mu_bandwidth': float -- detected mu bandwidth
            - 'baseline_power': ndarray (n_channels,) -- baseline mode amplitudes
            - 'epoch_power': ndarray (n_channels,) -- epoch mode amplitudes
        """
        # Fit on baseline to find subject's mu band
        baseline_decomp = KoopmanEEGDecomposition(
            n_channels=self.n_channels,
            sf=self.sf,
            n_modes=self.n_modes,
            delay_embedding_dim=self.delay_dim,
        )
        baseline_decomp.fit(baseline_epoch)
        mu_center, mu_bw = baseline_decomp.get_subject_mu_band()

        # Get mu-band spatial power from baseline
        baseline_mu_power = self._extract_mu_power(
            baseline_decomp, mu_center, mu_bw
        )

        # Fit on the event epoch
        epoch_decomp = KoopmanEEGDecomposition(
            n_channels=self.n_channels,
            sf=self.sf,
            n_modes=self.n_modes,
            delay_embedding_dim=self.delay_dim,
        )
        epoch_decomp.fit(epoch)

        epoch_mu_power = self._extract_mu_power(epoch_decomp, mu_center, mu_bw)

        # ERDS% = (epoch - baseline) / baseline * 100
        safe_baseline = np.where(
            baseline_mu_power > 1e-12, baseline_mu_power, 1e-12
        )
        erds_pct = (epoch_mu_power - baseline_mu_power) / safe_baseline * 100.0

        return {
            "erds_percent": erds_pct,
            "mu_center": mu_center,
            "mu_bandwidth": mu_bw,
            "baseline_power": baseline_mu_power,
            "epoch_power": epoch_mu_power,
        }

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    def _extract_mu_power(
        self,
        decomp: "KoopmanEEGDecomposition",
        mu_center: float,
        mu_bw: float,
    ) -> np.ndarray:
        """
        Sum spatial-pattern amplitudes of all modes within
        [mu_center - bw, mu_center + bw].

        Returns ndarray of shape (n_channels,).
        """
        modes = decomp.get_modes()
        power = np.zeros(self.n_channels)
        lo = mu_center - mu_bw
        hi = mu_center + mu_bw
        for m in modes:
            if lo <= m["frequency_hz"] <= hi:
                power += m["spatial_pattern"] * m["amplitude"]
        # If no modes fell in the band, return a small floor
        if np.sum(power) < 1e-15:
            power[:] = 1e-12
        return power

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "KoopmanEEGDecomposition has not been fitted yet. "
                "Call fit() first."
            )
