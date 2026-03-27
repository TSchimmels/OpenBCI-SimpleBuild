"""EEGNet classifier for motor imagery BCI (PyTorch implementation).

Implements the EEGNet architecture from Lawhern et al. (2018) as a
compact convolutional neural network designed specifically for EEG-based
brain-computer interfaces. The architecture uses depthwise and separable
convolutions to learn spatial and temporal features directly from raw
(bandpass-filtered) EEG epochs.

References:
    Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M.,
    Hung, C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional
    neural network for EEG-based brain-computer interfaces. Journal of
    Neural Engineering, 15(5), 056013.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import BaseClassifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard PyTorch import — give a clear message if it's missing
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
            "PyTorch is required for EEGNet but was not found. "
            "Install it with:\n"
            "  pip install torch torchvision torchaudio --index-url "
            "https://download.pytorch.org/whl/cu121\n"
            "or for CPU-only:\n"
            "  pip install torch torchvision torchaudio --index-url "
            "https://download.pytorch.org/whl/cpu\n"
            "See https://pytorch.org/get-started/locally/ for details."
        )


def _resolve_device(device: str) -> "torch.device":
    """Resolve ``'auto'`` to the best available device.

    Args:
        device: One of ``'auto'``, ``'cuda'``, ``'cpu'``, or a specific
            CUDA device string like ``'cuda:0'``.

    Returns:
        A ``torch.device`` instance.
    """
    _require_torch()
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            logger.info("EEGNet using CUDA: %s", torch.cuda.get_device_name(0))
        else:
            dev = torch.device("cpu")
            logger.info("EEGNet using CPU (CUDA not available)")
        return dev
    return torch.device(device)


# ======================================================================
# EEGNet PyTorch Module
# ======================================================================

class EEGNetModel(nn.Module):
    """EEGNet architecture (Lawhern et al., 2018).

    A compact convolutional network with three blocks:

    1. **Temporal convolution** — learns frequency filters.
    2. **Depthwise spatial convolution** — learns spatial filters
       (one per temporal filter), analogous to CSP.
    3. **Separable convolution** — combines features across time.

    Input shape:  ``(batch, 1, n_channels, n_samples)``
    Output shape: ``(batch, n_classes)`` (raw logits, no softmax).

    Args:
        n_channels: Number of EEG channels.
        n_samples: Number of time samples per epoch.
        n_classes: Number of output classes.
        F1: Number of temporal filters (Block 1).
        D: Depth multiplier for depthwise convolution (Block 2).
        F2: Number of pointwise filters (Block 3).  Typically ``F1 * D``.
        kernel_length: Length of the temporal convolution kernel.
            Should be roughly half the sampling rate to capture the
            lowest frequency of interest (~2 Hz at 125 Hz).
        dropout: Dropout probability applied after Blocks 2 and 3.
    """

    def __init__(
        self,
        n_channels: int = 16,
        n_samples: int = 125,
        n_classes: int = 3,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ) -> None:
        _require_torch()
        super().__init__()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes

        # ---- Block 1: Temporal convolution ----
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length), padding="same", bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # ---- Block 2: Depthwise spatial convolution ----
        self.depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        # ---- Block 3: Separable convolution ----
        self.separable_depthwise = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), groups=F1 * D, padding="same", bias=False,
        )
        self.separable_pointwise = nn.Conv2d(
            F1 * D, F2, (1, 1), bias=False,
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # ---- Classifier head ----
        # After Block 2 pool: time dimension = n_samples // 4
        # After Block 3 pool: time dimension = (n_samples // 4) // 8 = n_samples // 32
        flatten_size = F2 * (n_samples // 32)
        self.classifier = nn.Linear(flatten_size, n_classes)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, 1, n_channels, n_samples)``.

        Returns:
            Raw logits of shape ``(batch, n_classes)``.
        """
        # Block 1 — temporal
        x = self.conv1(x)
        x = self.bn1(x)

        # Block 2 — spatial (depthwise)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 3 — separable
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Classifier
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x


# ======================================================================
# EEGNet Classifier Wrapper (BaseClassifier interface)
# ======================================================================

class EEGNetClassifier(BaseClassifier):
    """EEGNet classifier with fit/predict interface.

    Wraps :class:`EEGNetModel` and provides the same API as
    :class:`CSPLDAClassifier` so classifiers can be swapped
    transparently.

    Training uses Adam with weight decay and early stopping on a
    held-out validation split (10 % of training data).

    Args:
        n_channels: Number of EEG channels.
        n_samples: Number of time samples per epoch.
        n_classes: Number of output classes.
        device: ``'auto'`` (default) selects CUDA when available.
        F1: Temporal filters.
        D: Depth multiplier.
        F2: Pointwise filters.
        kernel_length: Temporal kernel size.
        dropout: Dropout probability.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        weight_decay: Adam L2 penalty.
        patience: Early stopping patience (epochs without improvement).
        validation_fraction: Fraction of training data for validation.
    """

    def __init__(
        self,
        n_channels: int = 16,
        n_samples: int = 125,
        n_classes: int = 3,
        device: str = "auto",
        *,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
        epochs: int = 300,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        patience: int = 50,
        validation_fraction: float = 0.1,
    ) -> None:
        _require_torch()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.device: "torch.device" = _resolve_device(device)

        # Architecture hyper-parameters
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.kernel_length = kernel_length
        self.dropout = dropout

        # Training hyper-parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction

        # Model (created fresh in fit, or loaded via load())
        self._model: Optional[EEGNetModel] = None
        self._training_history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> EEGNetModel:
        """Instantiate and move model to the configured device."""
        model = EEGNetModel(
            n_channels=self.n_channels,
            n_samples=self.n_samples,
            n_classes=self.n_classes,
            F1=self.F1,
            D=self.D,
            F2=self.F2,
            kernel_length=self.kernel_length,
            dropout=self.dropout,
        )
        return model.to(self.device)

    def _numpy_to_tensor(self, X: np.ndarray) -> "torch.Tensor":
        """Convert numpy EEG array to a 4-D float32 tensor on device.

        Handles 2-D single-trial input ``(n_channels, n_samples)`` by
        prepending a batch dimension.

        Args:
            X: EEG data, shape ``(n_trials, n_channels, n_samples)`` or
                ``(n_channels, n_samples)``.

        Returns:
            Tensor of shape ``(batch, 1, n_channels, n_samples)``.
        """
        if X.ndim == 2:
            X = X[np.newaxis, :, :]
        # Add the "1-channel image" dimension: (batch, 1, C, T)
        X = X[:, np.newaxis, :, :].astype(np.float32)
        return torch.from_numpy(X).to(self.device)

    def _validate_model(self) -> None:
        """Raise if no model is available."""
        if self._model is None:
            raise RuntimeError(
                "EEGNet model has not been fitted or loaded. "
                "Call fit() or load() first."
            )

    # ------------------------------------------------------------------
    # BaseClassifier interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> EEGNetClassifier:
        """Train EEGNet with Adam and early stopping.

        A random 10 % of the training data is held out for validation.
        Training stops early if validation loss does not improve for
        ``patience`` consecutive epochs, and the best model weights are
        restored.

        Args:
            X: Training epochs, shape ``(n_trials, n_channels,
                n_samples)``.
            y: Integer class labels, shape ``(n_trials,)``.

        Returns:
            ``self``
        """
        _require_torch()

        # --- Prepare data ---
        n_trials = X.shape[0]
        indices = np.random.permutation(n_trials)
        n_val = max(1, int(n_trials * self.validation_fraction))
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_train_t = self._numpy_to_tensor(X[train_idx])
        y_train_t = torch.from_numpy(y[train_idx].astype(np.int64)).to(self.device)
        X_val_t = self._numpy_to_tensor(X[val_idx])
        y_val_t = torch.from_numpy(y[val_idx].astype(np.int64)).to(self.device)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False,
        )

        # --- Build model & optimiser ---
        self._model = self._build_model()
        optimiser = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # --- Training loop with early stopping ---
        best_val_loss = float("inf")
        best_state: Optional[dict[str, Any]] = None
        epochs_without_improvement = 0
        self._training_history = []

        for epoch in range(1, self.epochs + 1):
            # -- Train --
            self._model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                optimiser.zero_grad()
                logits = self._model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimiser.step()

                train_loss_sum += loss.item() * X_batch.size(0)
                train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                train_total += X_batch.size(0)

            train_loss = train_loss_sum / train_total
            train_acc = train_correct / train_total

            # -- Validate --
            self._model.eval()
            with torch.no_grad():
                val_logits = self._model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                val_acc = (val_logits.argmax(dim=1) == y_val_t).float().mean().item()

            self._training_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

            # -- Early stopping check --
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self._model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch % 25 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  train_acc=%.3f  "
                    "val_loss=%.4f  val_acc=%.3f",
                    epoch, self.epochs, train_loss, train_acc,
                    val_loss, val_acc,
                )

            if epochs_without_improvement >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d). "
                    "Best val_loss=%.4f",
                    epoch, self.patience, best_val_loss,
                )
                break

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)
            self._model.to(self.device)

        self._model.eval()

        # Free GPU memory used by training intermediates
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(
            "EEGNet training complete. Best val_loss=%.4f", best_val_loss,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Args:
            X: Input epochs, shape ``(n_trials, n_channels, n_samples)``
                or ``(n_channels, n_samples)`` for a single trial.

        Returns:
            Predicted integer labels, shape ``(n_trials,)``.
        """
        self._validate_model()
        X_t = self._numpy_to_tensor(X)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via softmax.

        Args:
            X: Input epochs.

        Returns:
            Probabilities, shape ``(n_trials, n_classes)``.
        """
        self._validate_model()
        X_t = self._numpy_to_tensor(X)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
            probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw logits as continuous decision scores.

        For proportional cursor control the logits provide a continuous
        signal reflecting classifier confidence for each class.

        Args:
            X: Input epochs.

        Returns:
            Logits, shape ``(n_trials, n_classes)``.
        """
        self._validate_model()
        X_t = self._numpy_to_tensor(X)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
        return logits.cpu().numpy()

    def predict_all(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict labels, probabilities, and decision scores in one pass.

        Converts numpy data to a GPU tensor **once** and runs a single
        forward pass, avoiding redundant CPU-to-GPU transfers when all
        three outputs are needed (e.g. in the real-time control loop).

        Args:
            X: Input epochs, shape ``(n_trials, n_channels, n_samples)``
                or ``(n_channels, n_samples)`` for a single trial.

        Returns:
            A 3-tuple ``(predictions, probabilities, logits)`` where:

            - **predictions** -- integer labels, shape ``(n_trials,)``.
            - **probabilities** -- softmax probs, shape
              ``(n_trials, n_classes)``.
            - **logits** -- raw decision scores, shape
              ``(n_trials, n_classes)``.
        """
        self._validate_model()
        X_t = self._numpy_to_tensor(X)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(X_t)
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
            path: File path (e.g. ``'models/eegnet.pt'``).
        """
        _require_torch()
        self._validate_model()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            # Architecture params needed to reconstruct the model
            "n_channels": self.n_channels,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "F1": self.F1,
            "D": self.D,
            "F2": self.F2,
            "kernel_length": self.kernel_length,
            "dropout": self.dropout,
            # Trained weights
            "state_dict": self._model.state_dict(),
            # Training history for diagnostics
            "training_history": self._training_history,
        }
        torch.save(checkpoint, path)
        logger.info("EEGNet model saved to %s", path)

    @classmethod
    def load(cls, path: str, device: str = "auto") -> EEGNetClassifier:
        """Load a saved EEGNet model.

        Args:
            path: Path to the saved checkpoint.
            device: Device to load the model onto.

        Returns:
            A fitted :class:`EEGNetClassifier` ready for inference.
        """
        _require_torch()

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        clf = cls(
            n_channels=checkpoint["n_channels"],
            n_samples=checkpoint["n_samples"],
            n_classes=checkpoint["n_classes"],
            device=device,
            F1=checkpoint["F1"],
            D=checkpoint["D"],
            F2=checkpoint["F2"],
            kernel_length=checkpoint["kernel_length"],
            dropout=checkpoint["dropout"],
        )

        clf._model = clf._build_model()
        clf._model.load_state_dict(checkpoint["state_dict"])
        clf._model.eval()
        clf._training_history = checkpoint.get("training_history", [])

        logger.info("EEGNet model loaded from %s", path)
        return clf

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self._model is not None
        return (
            f"EEGNetClassifier(n_channels={self.n_channels}, "
            f"n_samples={self.n_samples}, n_classes={self.n_classes}, "
            f"device={self.device}, fitted={fitted})"
        )
