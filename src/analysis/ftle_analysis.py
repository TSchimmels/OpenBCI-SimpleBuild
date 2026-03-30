"""
FTLE (Finite-Time Lyapunov Exponent) ridge analysis for EEG time-frequency space.

Finds Lagrangian Coherent Structures (LCS) — the dynamical boundaries between
different motor imagery states in the time-frequency plane.

Reference:
    Haller, G. (2015). "Lagrangian Coherent Structures."
    Annual Review of Fluid Mechanics, 47, 137-162.

Adapted for EEG time-frequency analysis: treats the wavelet power spectrum as a
2D flow field over (time, frequency) and computes the FTLE to reveal ridges that
mark transitions between distinct brain states.
"""

import numpy as np
from scipy.signal import morlet2
from scipy.ndimage import gaussian_filter, maximum_filter, label
from scipy.interpolate import RectBivariateSpline
from typing import List, Dict, Optional, Tuple


class FTLEAnalyzer:
    """
    Computes Finite-Time Lyapunov Exponents on EEG time-frequency
    representations to identify dynamical boundaries (ridges) between
    motor imagery states.

    The key insight is that high-FTLE ridges in (time, frequency) space
    correspond to Lagrangian Coherent Structures — boundaries where the
    neural dynamics are maximally separating, which aligns with state
    transitions in motor imagery tasks.

    Parameters
    ----------
    sf : float
        Sampling frequency in Hz.
    freqs : np.ndarray or None
        Frequency vector for the time-frequency decomposition.
        If None, defaults to 1–45 Hz in 0.5 Hz steps.
    n_cycles : float
        Number of cycles for Morlet wavelet (controls time-frequency
        trade-off). Default 5.0.
    """

    def __init__(
        self,
        sf: float,
        freqs: Optional[np.ndarray] = None,
        n_cycles: float = 5.0,
    ):
        self.sf = sf
        self.freqs = freqs if freqs is not None else np.arange(1, 45.5, 0.5)
        self.n_cycles = n_cycles

    # ------------------------------------------------------------------
    # Time-frequency decomposition
    # ------------------------------------------------------------------

    def _compute_tfr(self, signal: np.ndarray) -> np.ndarray:
        """Compute Morlet wavelet time-frequency power.

        Parameters
        ----------
        signal : 1-D array, shape (n_samples,)

        Returns
        -------
        power : 2-D array, shape (n_freqs, n_samples)
        """
        n_samples = signal.shape[0]
        n_freqs = len(self.freqs)
        power = np.zeros((n_freqs, n_samples))

        for i, freq in enumerate(self.freqs):
            w = self.n_cycles
            s = w / (2.0 * np.pi * freq)  # wavelet width in seconds
            M = int(10 * s * self.sf)  # wavelet length in samples
            if M % 2 == 0:
                M += 1
            wavelet = morlet2(M, s * self.sf, w)
            # convolve via FFT
            n_conv = n_samples + M - 1
            fft_sig = np.fft.fft(signal, n=n_conv)
            fft_wav = np.fft.fft(wavelet, n=n_conv)
            conv = np.fft.ifft(fft_sig * fft_wav)
            # trim to original length
            half = M // 2
            analytic = conv[half: half + n_samples]
            power[i, :] = np.abs(analytic) ** 2

        return power

    # ------------------------------------------------------------------
    # Flow map via particle advection
    # ------------------------------------------------------------------

    def _advect_particles(
        self,
        power: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Advect particles on the (time, frequency) grid using power
        gradients as the velocity field.

        The velocity at each grid point is:
            dx/dt = dP/dt   (power gradient in time direction)
            df/dt = dP/df   (power gradient in frequency direction)

        Advection is performed with 4th-order Runge-Kutta.

        Parameters
        ----------
        power : shape (n_freqs, n_samples)
        dt : integration time in normalised units

        Returns
        -------
        phi_t, phi_f : displaced positions, each shape (n_freqs, n_samples)
        """
        n_freqs, n_samples = power.shape

        # Smooth lightly so gradients are well-defined
        smoothed = gaussian_filter(power, sigma=1.0)

        # Central-difference gradients
        grad_t = np.gradient(smoothed, axis=1)  # dP/dt
        grad_f = np.gradient(smoothed, axis=0)  # dP/df

        # Normalise velocity so advection stays within the grid
        max_speed = max(np.max(np.abs(grad_t)), np.max(np.abs(grad_f)), 1e-12)
        grad_t /= max_speed
        grad_f /= max_speed

        # Build interpolators for the velocity field
        freq_idx = np.arange(n_freqs, dtype=float)
        time_idx = np.arange(n_samples, dtype=float)
        interp_vt = RectBivariateSpline(freq_idx, time_idx, grad_t)
        interp_vf = RectBivariateSpline(freq_idx, time_idx, grad_f)

        # Initial particle positions on the grid
        ff, tt = np.meshgrid(freq_idx, time_idx, indexing="ij")
        pos_f = ff.copy()
        pos_t = tt.copy()

        def _velocity(pf, pt):
            """Evaluate velocity at arbitrary positions (clamped to grid)."""
            pf_c = np.clip(pf, 0, n_freqs - 1)
            pt_c = np.clip(pt, 0, n_samples - 1)
            vt = interp_vt.ev(pf_c.ravel(), pt_c.ravel()).reshape(pf.shape)
            vf = interp_vf.ev(pf_c.ravel(), pt_c.ravel()).reshape(pf.shape)
            return vf, vt

        # RK4 integration
        n_steps = max(int(np.ceil(abs(dt) / 0.05)), 1)
        h = dt / n_steps
        for _ in range(n_steps):
            k1f, k1t = _velocity(pos_f, pos_t)
            k2f, k2t = _velocity(pos_f + 0.5 * h * k1f, pos_t + 0.5 * h * k1t)
            k3f, k3t = _velocity(pos_f + 0.5 * h * k2f, pos_t + 0.5 * h * k2t)
            k4f, k4t = _velocity(pos_f + h * k3f, pos_t + h * k3t)
            pos_f += (h / 6.0) * (k1f + 2 * k2f + 2 * k3f + k4f)
            pos_t += (h / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)

        return pos_t, pos_f

    # ------------------------------------------------------------------
    # FTLE computation
    # ------------------------------------------------------------------

    def compute_ftle(
        self,
        epoch: np.ndarray,
        channel: int = 0,
        dt: float = 0.1,
    ) -> np.ndarray:
        """Compute the FTLE field for one EEG epoch.

        Steps:
            1. Compute Morlet wavelet power (time-frequency representation).
            2. Treat (time, frequency) as a 2D flow field with velocities
               derived from power gradients.
            3. Advect particles for integration time *dt* via RK4.
            4. Compute the deformation gradient tensor F = d phi / d x.
            5. Compute the Cauchy-Green strain tensor C = F^T F.
            6. FTLE = 1/(2|T|) * log(lambda_max(C)).

        Parameters
        ----------
        epoch : np.ndarray
            EEG data, shape (n_channels, n_samples) or (n_samples,).
        channel : int
            Channel index to analyse (ignored if epoch is 1-D).
        dt : float
            Advection integration time (normalised). Larger values
            reveal longer-lived structures.

        Returns
        -------
        ftle : np.ndarray, shape (n_freqs, n_samples)
            FTLE field. High values mark ridges (LCS boundaries).
        """
        if epoch.ndim == 1:
            signal = epoch
        else:
            signal = epoch[channel]

        power = self._compute_tfr(signal)

        # Advect particles
        phi_t, phi_f = self._advect_particles(power, dt)

        n_freqs, n_samples = power.shape

        # Deformation gradient F = [[dphi_t/dt, dphi_t/df],
        #                           [dphi_f/dt, dphi_f/df]]
        dphi_t_dt = np.gradient(phi_t, axis=1)
        dphi_t_df = np.gradient(phi_t, axis=0)
        dphi_f_dt = np.gradient(phi_f, axis=1)
        dphi_f_df = np.gradient(phi_f, axis=0)

        # Cauchy-Green tensor C = F^T F  —  compute max eigenvalue pointwise
        # C = [[a, b], [b, d]]
        a = dphi_t_dt ** 2 + dphi_f_dt ** 2
        b = dphi_t_dt * dphi_t_df + dphi_f_dt * dphi_f_df
        d = dphi_t_df ** 2 + dphi_f_df ** 2

        # Max eigenvalue of 2x2 symmetric matrix via closed-form expression
        trace = a + d
        det = a * d - b ** 2
        discriminant = np.maximum(trace ** 2 - 4.0 * det, 0.0)
        lambda_max = 0.5 * (trace + np.sqrt(discriminant))
        lambda_max = np.maximum(lambda_max, 1e-12)

        T = abs(dt) if abs(dt) > 0 else 1.0
        ftle = (1.0 / (2.0 * T)) * np.log(lambda_max)

        return ftle

    # ------------------------------------------------------------------
    # Ridge extraction
    # ------------------------------------------------------------------

    def find_ridges(
        self,
        ftle_map: np.ndarray,
        threshold: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Extract ridge lines from the FTLE map via height-ridge detection.

        A point is on a ridge if it is a local maximum in the direction of
        the smallest curvature (Hessian eigenvector).

        Parameters
        ----------
        ftle_map : shape (n_freqs, n_samples)
        threshold : float or None
            Minimum FTLE value for a point to be considered part of a
            ridge. Defaults to mean + 1.5 * std.

        Returns
        -------
        ridges : list of np.ndarray
            Each element is shape (N, 2) with columns [time_idx, freq_idx].
        """
        if threshold is None:
            threshold = float(np.mean(ftle_map) + 1.5 * np.std(ftle_map))

        # Local maxima: points equal to their neighbourhood maximum
        footprint = np.ones((3, 3))
        local_max = maximum_filter(ftle_map, footprint=footprint)
        ridge_mask = (ftle_map == local_max) & (ftle_map >= threshold)

        # Label connected components
        labelled, n_labels = label(ridge_mask)
        ridges: List[np.ndarray] = []
        for lbl in range(1, n_labels + 1):
            coords = np.argwhere(labelled == lbl)  # (freq_idx, time_idx)
            # Sort by time index for a coherent ridge line
            order = np.argsort(coords[:, 1])
            coords = coords[order]
            # Store as (time_idx, freq_idx)
            ridges.append(coords[:, ::-1].copy())

        return ridges

    # ------------------------------------------------------------------
    # Transition classification
    # ------------------------------------------------------------------

    def classify_transitions(
        self,
        ftle_map: np.ndarray,
        times: np.ndarray,
    ) -> List[Dict]:
        """Identify state transitions based on FTLE ridge crossings.

        For each detected ridge, the method estimates the time at which
        the ridge is strongest (peak FTLE along the ridge) and reports it
        as a candidate transition point.

        Parameters
        ----------
        ftle_map : shape (n_freqs, n_samples)
        times : 1-D array of time stamps (seconds), length n_samples.

        Returns
        -------
        transitions : list of dict
            Each dict contains:
            - 'time_s': estimated transition time in seconds
            - 'time_idx': sample index of the transition
            - 'freq_range_hz': (low, high) frequency band of the ridge
            - 'peak_ftle': maximum FTLE value along the ridge
            - 'ridge_length': number of points in the ridge
        """
        ridges = self.find_ridges(ftle_map)
        transitions: List[Dict] = []

        for ridge in ridges:
            if len(ridge) < 2:
                continue
            t_indices = ridge[:, 0]
            f_indices = ridge[:, 1]
            # FTLE values along the ridge
            ftle_vals = ftle_map[f_indices, t_indices]
            peak_idx = int(np.argmax(ftle_vals))
            t_peak = int(t_indices[peak_idx])

            # Map frequency indices back to Hz
            freq_low = float(self.freqs[int(np.min(f_indices))])
            freq_high = float(self.freqs[min(int(np.max(f_indices)),
                                              len(self.freqs) - 1)])

            transitions.append({
                "time_s": float(times[min(t_peak, len(times) - 1)]),
                "time_idx": t_peak,
                "freq_range_hz": (freq_low, freq_high),
                "peak_ftle": float(np.max(ftle_vals)),
                "ridge_length": len(ridge),
            })

        # Sort by time
        transitions.sort(key=lambda d: d["time_s"])
        return transitions
