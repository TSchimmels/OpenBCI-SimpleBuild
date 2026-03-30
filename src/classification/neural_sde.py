"""Latent Neural SDE classifier for EEG signals.

Models brain dynamics as stochastic differential equations with jump
processes for state transitions between rest and motor imagery onset:

    dz = f(z,t)dt + g(z,t)dW + J(z,t)dN

Where:
    f(z,t) = drift network (deterministic MI pattern evolution)
    g(z,t) = diffusion network (trial-to-trial variability)
    J(z,t)dN = jump process (sudden state changes: rest -> imagery onset)

The SDE latent space provides interpretable dynamics: drift captures the
average neural trajectory during MI, diffusion captures between-trial
variability, and jumps capture abrupt transitions (e.g., the moment a
subject initiates motor imagery).

References:
    Li, X., Wong, T.-K. L., Chen, R. T. Q., & Duvenaud, D. (2020).
    "Scalable Gradients for Stochastic Differential Equations."
    Advances in Neural Information Processing Systems (NeurIPS).

    Kidger, P., Foster, J., Li, X., & Lyons, T. (2021).
    "Neural SDEs as Infinite-Dimensional GANs." ICML.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard PyTorch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    """Raise a helpful ImportError if PyTorch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for NeuralSDEClassifier but was not found. "
            "Install it with:\n"
            "  pip install torch torchvision torchaudio --index-url "
            "https://download.pytorch.org/whl/cu121\n"
            "or for CPU-only:\n"
            "  pip install torch torchvision torchaudio --index-url "
            "https://download.pytorch.org/whl/cpu\n"
            "See https://pytorch.org/get-started/locally/ for details."
        )


def _resolve_device(device: str) -> "torch.device":
    """Resolve ``'auto'`` to the best available device."""
    _require_torch()
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            logger.info("NeuralSDE using CUDA: %s", torch.cuda.get_device_name(0))
        else:
            dev = torch.device("cpu")
            logger.info("NeuralSDE using CPU (CUDA not available)")
        return dev
    return torch.device(device)


# ======================================================================
# Sub-networks for the SDE components
# ======================================================================

class _Encoder(nn.Module):
    """Conv1d encoder: maps (n_channels, n_samples) -> latent_dim.

    Two Conv1d layers with BatchNorm and ELU activations, followed by
    adaptive average pooling and a linear projection to the latent space.
    Outputs both mean and log-variance for the initial latent distribution
    q(z0 | x).
    """

    def __init__(self, n_channels: int, n_samples: int, latent_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Output mean and log-variance for the reparameterisation trick
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Encode EEG to latent distribution parameters.

        Args:
            x: Input tensor, shape (batch, n_channels, n_samples).

        Returns:
            (mu, logvar) each of shape (batch, latent_dim).
        """
        h = F.elu(self.bn1(self.conv1(x)))
        h = F.elu(self.bn2(self.conv2(h)))
        h = F.elu(self.bn3(self.conv3(h)))
        h = self.pool(h).squeeze(-1)  # (batch, 128)
        return self.fc_mu(h), self.fc_logvar(h)


class _DriftNet(nn.Module):
    """Drift network f(z, t): deterministic dynamics in latent space.

    A 2-layer MLP that takes the current latent state and (optionally)
    the current time, producing the deterministic drift component.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        # +1 for time input
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
        """Compute drift f(z, t).

        Args:
            z: Latent state, shape (batch, latent_dim).
            t: Current time scalar broadcast to (batch, 1).

        Returns:
            Drift vector, shape (batch, latent_dim).
        """
        t_expanded = t.expand(z.shape[0], 1)
        return self.net(torch.cat([z, t_expanded], dim=-1))


class _DiffusionNet(nn.Module):
    """Diffusion network g(z, t): state-dependent noise magnitude.

    Outputs strictly positive values via softplus to ensure the diffusion
    coefficient is well-defined. Models trial-to-trial variability in
    neural dynamics.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus(),  # ensure positive diffusion
        )

    def forward(self, z: "torch.Tensor", t: "torch.Tensor") -> "torch.Tensor":
        """Compute diffusion g(z, t).

        Args:
            z: Latent state, shape (batch, latent_dim).
            t: Current time scalar broadcast to (batch, 1).

        Returns:
            Diffusion coefficients (positive), shape (batch, latent_dim).
        """
        t_expanded = t.expand(z.shape[0], 1)
        return self.net(torch.cat([z, t_expanded], dim=-1))


class _JumpDetector(nn.Module):
    """Jump detector J(z, t): predicts probability and magnitude of jumps.

    Models sudden state transitions (e.g., rest -> motor imagery onset)
    as a compound Poisson process with state-dependent intensity and
    jump size.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        # Jump probability (intensity of the Poisson process)
        self.prob_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Jump magnitude
        self.mag_net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self, z: "torch.Tensor", t: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Compute jump probability and magnitude.

        Args:
            z: Latent state, shape (batch, latent_dim).
            t: Current time scalar broadcast to (batch, 1).

        Returns:
            (jump_prob, jump_magnitude):
                jump_prob: shape (batch, 1), values in [0, 1]
                jump_magnitude: shape (batch, latent_dim)
        """
        t_expanded = t.expand(z.shape[0], 1)
        zt = torch.cat([z, t_expanded], dim=-1)
        return self.prob_net(zt), self.mag_net(zt)


# ======================================================================
# Full Neural SDE Model
# ======================================================================

class NeuralSDEModel(nn.Module):
    """Complete Neural SDE model for EEG classification.

    Combines encoder, SDE dynamics (drift + diffusion + jumps), and a
    classifier head. The forward pass encodes EEG into an initial latent
    state, integrates the SDE using Euler-Maruyama, and classifies the
    terminal state.

    Args:
        n_channels: Number of EEG channels.
        n_samples: Number of time samples per epoch.
        n_classes: Number of output classes.
        latent_dim: Dimension of the latent SDE state.
        n_steps: Number of Euler-Maruyama integration steps.
        dt: Time step for integration (total time = n_steps * dt).
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int = 5,
        latent_dim: int = 32,
        n_steps: int = 20,
        dt: float = 0.05,
    ) -> None:
        _require_torch()
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        self.dt = dt

        # Sub-networks
        self.encoder = _Encoder(n_channels, n_samples, latent_dim)
        self.drift = _DriftNet(latent_dim)
        self.diffusion = _DiffusionNet(latent_dim)
        self.jump_detector = _JumpDetector(latent_dim)
        self.classifier = nn.Linear(latent_dim, n_classes)

    def _reparameterise(
        self, mu: "torch.Tensor", logvar: "torch.Tensor"
    ) -> "torch.Tensor":
        """Sample z0 from q(z0|x) using the reparameterisation trick.

        Args:
            mu: Mean of q(z0|x), shape (batch, latent_dim).
            logvar: Log-variance of q(z0|x), shape (batch, latent_dim).

        Returns:
            Sampled z0, shape (batch, latent_dim).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(
        self, x: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Full forward pass: encode -> integrate SDE -> classify.

        Args:
            x: EEG input, shape (batch, n_channels, n_samples).

        Returns:
            (logits, mu, logvar, jump_probs):
                logits: class scores, shape (batch, n_classes)
                mu: encoder mean, shape (batch, latent_dim)
                logvar: encoder log-variance, shape (batch, latent_dim)
                jump_probs: accumulated jump probabilities, shape (batch, 1)
        """
        # 1. Encode EEG -> initial latent distribution
        mu, logvar = self.encoder(x)
        z = self._reparameterise(mu, logvar)

        # 2. Integrate SDE via Euler-Maruyama
        sqrt_dt = float(self.dt) ** 0.5
        total_jump_prob = torch.zeros(z.shape[0], 1, device=z.device)

        for step in range(self.n_steps):
            t_val = step * self.dt
            t = torch.tensor([[t_val]], device=z.device, dtype=z.dtype)

            # Drift: deterministic dynamics
            f_z = self.drift(z, t)

            # Diffusion: stochastic noise
            g_z = self.diffusion(z, t)
            noise = torch.randn_like(z)

            # Jump: sudden state transitions
            jump_prob, jump_mag = self.jump_detector(z, t)
            total_jump_prob = total_jump_prob + jump_prob

            if self.training:
                # Stochastic: sample Bernoulli for jump events
                jump_mask = torch.bernoulli(jump_prob)
            else:
                # Deterministic at inference: use expected value
                jump_mask = jump_prob

            # Euler-Maruyama update:
            #   z_{t+1} = z_t + f(z,t)*dt + g(z,t)*sqrt(dt)*eps
            #           + J(z,t)*mask*dt
            z = (
                z
                + f_z * self.dt
                + g_z * sqrt_dt * noise
                + jump_mag * jump_mask * self.dt
            )

        # 3. Classify the terminal latent state
        logits = self.classifier(z)

        # Average jump probability over steps for regularisation
        avg_jump_prob = total_jump_prob / self.n_steps

        return logits, mu, logvar, avg_jump_prob


# ======================================================================
# Neural SDE Classifier (BaseClassifier interface)
# ======================================================================

class NeuralSDEClassifier(BaseClassifier):
    """Latent Neural SDE classifier for EEG motor imagery.

    Wraps :class:`NeuralSDEModel` and provides the same API as other
    classifiers in the pipeline (CSP+LDA, EEGNet) so they can be
    swapped transparently.

    The loss function combines three terms::

        L = CrossEntropy + beta * KL(q(z0|x) || p(z0)) + gamma * JumpSparsity

    Where:
        - CrossEntropy drives classification accuracy
        - KL divergence regularises the initial latent distribution
        - JumpSparsity encourages sparse (infrequent) state transitions

    Args:
        n_channels: Number of EEG channels.
        n_samples: Number of time samples per epoch.
        n_classes: Number of output classes.
        latent_dim: Dimension of the latent SDE state.
        n_steps: Number of Euler-Maruyama integration steps.
        dt: Time step for integration.
        device: ``'auto'`` selects CUDA when available.
    """

    def __init__(
        self,
        n_channels: int = 16,
        n_samples: int = 125,
        n_classes: int = 5,
        latent_dim: int = 32,
        n_steps: int = 20,
        dt: float = 0.05,
        device: str = "auto",
    ) -> None:
        _require_torch()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.n_steps = n_steps
        self.dt = dt
        self.device: "torch.device" = _resolve_device(device)

        # Model (created fresh in fit, or loaded via load())
        self._model: Optional[NeuralSDEModel] = None
        self._training_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> NeuralSDEModel:
        """Instantiate and move model to the configured device."""
        model = NeuralSDEModel(
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            n_classes=self.n_classes,
            latent_dim=self.latent_dim,
            n_steps=self.n_steps,
            dt=self.dt,
        )
        return model.to(self.device)

    def _numpy_to_tensor(self, X: np.ndarray) -> "torch.Tensor":
        """Convert numpy EEG array to a 3-D float32 tensor on device.

        Handles 2-D single-trial input (n_channels, n_samples) by
        prepending a batch dimension.

        Args:
            X: EEG data, shape (n_trials, n_channels, n_samples) or
                (n_channels, n_samples).

        Returns:
            Tensor of shape (batch, n_channels, n_samples).
        """
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
        return torch.from_numpy(X.astype(np.float32)).to(self.device)

    def _validate_model(self) -> None:
        """Raise if no model is available."""
        if self._model is None:
            raise RuntimeError(
                "NeuralSDE model has not been fitted or loaded. "
                "Call fit() or load() first."
            )

    @staticmethod
    def _kl_divergence(
        mu: "torch.Tensor", logvar: "torch.Tensor"
    ) -> "torch.Tensor":
        """KL divergence KL(q(z0|x) || N(0,I)).

        Args:
            mu: Mean, shape (batch, latent_dim).
            logvar: Log-variance, shape (batch, latent_dim).

        Returns:
            Scalar KL divergence averaged over the batch.
        """
        # KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()

    # ------------------------------------------------------------------
    # BaseClassifier interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        batch_size: int = 32,
        lr: float = 1e-3,
        beta: float = 0.1,
        gamma: float = 0.01,
        patience: int = 30,
        validation_fraction: float = 0.1,
    ) -> "NeuralSDEClassifier":
        """Train the Neural SDE classifier.

        Uses Adam with gradient clipping for SDE stability. Training
        includes early stopping on a held-out validation split.

        Args:
            X: Training epochs, shape (n_trials, n_channels, n_samples).
            y: Integer class labels, shape (n_trials,).
            epochs: Maximum training epochs.
            batch_size: Mini-batch size.
            lr: Learning rate for Adam optimiser.
            beta: Weight for KL divergence regularisation.
            gamma: Weight for jump sparsity regularisation.
            patience: Early stopping patience.
            validation_fraction: Fraction of data for validation.

        Returns:
            self
        """
        _require_torch()

        # --- Prepare data ---
        n_trials = X.shape[0]
        indices = np.random.permutation(n_trials)
        n_val = max(1, int(n_trials * validation_fraction))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train_t = self._numpy_to_tensor(X[train_idx])
        y_train_t = torch.from_numpy(y[train_idx].astype(np.int64)).to(self.device)
        X_val_t = self._numpy_to_tensor(X[val_idx])
        y_val_t = torch.from_numpy(y[val_idx].astype(np.int64)).to(self.device)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
        )

        # --- Build model & optimiser ---
        self._model = self._build_model()
        optimiser = torch.optim.Adam(self._model.parameters(), lr=lr)
        ce_criterion = nn.CrossEntropyLoss()

        # --- Training loop with early stopping ---
        best_val_loss = float("inf")
        best_state: Optional[Dict[str, Any]] = None
        epochs_no_improve = 0
        self._training_history = []

        for epoch in range(1, epochs + 1):
            # -- Train --
            self._model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                optimiser.zero_grad()

                logits, mu, logvar, avg_jump_prob = self._model(X_batch)

                # Combined loss: CE + KL regularisation + jump sparsity
                ce_loss = ce_criterion(logits, y_batch)
                kl_loss = self._kl_divergence(mu, logvar)
                jump_sparsity = avg_jump_prob.mean()
                loss = ce_loss + beta * kl_loss + gamma * jump_sparsity

                loss.backward()
                # Gradient clipping for SDE stability
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), max_norm=5.0
                )
                optimiser.step()

                train_loss_sum += loss.item() * X_batch.size(0)
                train_correct += (
                    (logits.argmax(dim=1) == y_batch).sum().item()
                )
                train_total += X_batch.size(0)

            train_loss = train_loss_sum / train_total
            train_acc = train_correct / train_total

            # -- Validate --
            self._model.eval()
            with torch.no_grad():
                val_logits, val_mu, val_logvar, val_jump = self._model(X_val_t)
                val_ce = ce_criterion(val_logits, y_val_t).item()
                val_kl = self._kl_divergence(val_mu, val_logvar).item()
                val_loss = val_ce + beta * val_kl
                val_acc = (
                    (val_logits.argmax(dim=1) == y_val_t).float().mean().item()
                )

            self._training_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

            # -- Early stopping --
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self._model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epoch % 25 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  loss=%.4f  acc=%.3f  "
                    "val_loss=%.4f  val_acc=%.3f",
                    epoch, epochs, train_loss, train_acc,
                    val_loss, val_acc,
                )

            if epochs_no_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d). "
                    "Best val_loss=%.4f",
                    epoch, patience, best_val_loss,
                )
                break

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(self.device)

        self._model.eval()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(
            "NeuralSDE training complete. Best val_loss=%.4f", best_val_loss
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Input epochs, shape (n_trials, n_channels, n_samples)
                or (n_channels, n_samples) for a single trial.

        Returns:
            Predicted integer labels, shape (n_trials,).
        """
        self._validate_model()
        X_t = self._numpy_to_tensor(X)

        self._model.eval()
        with torch.no_grad():
            logits, _, _, _ = self._model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via softmax.

        Args:
            X: Input epochs.

        Returns:
            Probabilities, shape (n_trials, n_classes).
        """
        self._validate_model()
        X_t = self._numpy_to_tensor(X)

        self._model.eval()
        with torch.no_grad():
            logits, _, _, _ = self._model(X_t)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw logits as continuous decision scores.

        For proportional control the logits provide a continuous signal
        reflecting classifier confidence for each class.

        Args:
            X: Input epochs.

        Returns:
            Logits, shape (n_trials, n_classes).
        """
        self._validate_model()
        X_t = self._numpy_to_tensor(X)

        self._model.eval()
        with torch.no_grad():
            logits, _, _, _ = self._model(X_t)
        return logits.cpu().numpy()

    def predict_all(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels, probabilities, and decision scores in one pass.

        Converts numpy data to a tensor once and runs a single forward
        pass, avoiding redundant transfers when all three outputs are
        needed (e.g. in the real-time control loop).

        Args:
            X: Input epochs, shape (n_trials, n_channels, n_samples)
                or (n_channels, n_samples) for a single trial.

        Returns:
            (predictions, probabilities, logits)
        """
        self._validate_model()
        X_t = self._numpy_to_tensor(X)

        self._model.eval()
        with torch.no_grad():
            logits, _, _, _ = self._model(X_t)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

        return (
            preds.cpu().numpy(),
            probs.cpu().numpy(),
            logits.cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Save / Load overrides (state_dict instead of joblib)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model state_dict and hyper-parameters.

        Args:
            path: File path (e.g. 'models/neural_sde.pt').
        """
        _require_torch()
        self._validate_model()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "n_channels": self.n_channels,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "latent_dim": self.latent_dim,
            "n_steps": self.n_steps,
            "dt": self.dt,
            "state_dict": self._model.state_dict(),
            "training_history": self._training_history,
        }
        torch.save(checkpoint, path)
        logger.info("NeuralSDE model saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "auto") -> "NeuralSDEClassifier":
        """Load a saved Neural SDE model.

        Args:
            path: Path to the saved checkpoint.
            device: Device to load the model onto.

        Returns:
            A fitted NeuralSDEClassifier ready for inference.
        """
        _require_torch()

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        clf = cls(
            n_channels=checkpoint["n_channels"],
            n_samples=checkpoint["n_samples"],
            n_classes=checkpoint["n_classes"],
            latent_dim=checkpoint["latent_dim"],
            n_steps=checkpoint["n_steps"],
            dt=checkpoint["dt"],
            device=device,
        )

        clf._model = clf._build_model()
        clf._model.load_state_dict(checkpoint["state_dict"])
        clf._model.eval()
        clf._training_history = checkpoint.get("training_history", [])

        logger.info("NeuralSDE model loaded from %s", path)
        return clf

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self._model is not None
        return (
            f"NeuralSDEClassifier(n_channels={self.n_channels}, "
            f"n_samples={self.n_samples}, n_classes={self.n_classes}, "
            f"latent_dim={self.latent_dim}, n_steps={self.n_steps}, "
            f"dt={self.dt}, device={self.device}, fitted={fitted})"
        )
