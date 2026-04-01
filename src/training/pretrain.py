"""
TS-JEPA Inspired Self-Supervised Pre-Training for EEG

Learns EEG structure from UNLABELED data to reduce calibration burden.
The encoder discovers what mu rhythms, artifacts, and baseline activity
look like without any task labels. Then a simple linear classifier can
be fine-tuned with minimal labeled data.

Architecture:
    - Encoder: 1D CNN mapping (n_channels, n_samples) -> (n_channels, embed_dim)
    - Predictor: small MLP predicting masked channel embeddings from visible ones
    - Target encoder: exponential moving average (EMA) of the encoder (BYOL-style)

The spatial masking strategy randomly drops 50% of channels, forcing the
encoder to learn cross-channel correlations (e.g., bilateral mu rhythms,
artifact propagation patterns).

Reference:
    Assran, M., Duval, Q., Misra, I., et al. (2023). "Self-Supervised
    Learning from Images with a Joint-Embedding Predictive Architecture."
    CVPR 2023. -- Adapted here for multi-channel time-series (EEG).

Requires: torch (with try/except guard)
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not available. JEPAPretrainer will raise an error on "
        "instantiation. Install with: pip install torch"
    )


def _require_torch() -> None:
    """Raise a clear error if torch is missing."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for JEPAPretrainer but was not found. "
            "Install with: pip install torch"
        )


# ======================================================================
# Network components
# ======================================================================

if TORCH_AVAILABLE:

    class _EEGEncoder(nn.Module):
        """
        1-D convolutional encoder:
            (batch, n_channels, n_samples) -> (batch, n_channels, embed_dim)

        Three conv blocks with GELU activation and group normalization.
        The temporal dimension is progressively reduced to embed_dim via
        adaptive average pooling.
        """

        def __init__(
            self, n_channels: int, n_samples: int, embed_dim: int = 64
        ):
            super().__init__()
            self.n_channels = n_channels
            self.embed_dim = embed_dim

            # Grouped temporal convolutions (groups=n_channels, 4 ch/group after first layer)
            self.conv1 = nn.Conv1d(
                n_channels, n_channels * 4,
                kernel_size=7, padding=3, groups=n_channels,
            )
            self.norm1 = nn.GroupNorm(n_channels, n_channels * 4)
            self.conv2 = nn.Conv1d(
                n_channels * 4, n_channels * 4,
                kernel_size=5, padding=2, groups=n_channels,
            )
            self.norm2 = nn.GroupNorm(n_channels, n_channels * 4)
            self.conv3 = nn.Conv1d(
                n_channels * 4, n_channels,
                kernel_size=3, padding=1, groups=n_channels,
            )
            self.norm3 = nn.GroupNorm(n_channels, n_channels)

            self.activation = nn.GELU()
            self.pool = nn.AdaptiveAvgPool1d(embed_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            x : Tensor, shape (batch, n_channels, n_samples)

            Returns
            -------
            embeddings : Tensor, shape (batch, n_channels, embed_dim)
            """
            h = self.activation(self.norm1(self.conv1(x)))
            h = self.activation(self.norm2(self.conv2(h)))
            h = self.activation(self.norm3(self.conv3(h)))
            h = self.pool(h)
            return h

    class _Predictor(nn.Module):
        """
        Small MLP that predicts target embeddings for masked channels
        from the context (visible channel embeddings).

        Input:  (batch, n_visible * embed_dim)
        Output: (batch, n_masked * embed_dim)
        """

        def __init__(
            self, input_dim: int, output_dim: int, hidden_dim: int = 256
        ):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


# ======================================================================
# Main pre-trainer
# ======================================================================

class JEPAPretrainer:
    """
    Joint-Embedding Predictive Architecture pre-trainer for EEG.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_samples : int
        Number of time samples per window.
    sf : float
        Sampling frequency in Hz.
    embed_dim : int
        Embedding dimension per channel produced by the encoder.
    mask_ratio : float
        Fraction of channels to mask during pre-training (0.0 to 1.0).
    ema_decay : float
        Exponential moving average decay for the target encoder update.
    lr : float
        Learning rate for Adam optimizer.
    """

    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        sf: float,
        embed_dim: int = 64,
        mask_ratio: float = 0.5,
        ema_decay: float = 0.99,
        lr: float = 1e-3,
    ):
        _require_torch()

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.sf = sf
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.lr = lr

        # Number of channels to mask
        self.n_masked = max(1, int(round(n_channels * mask_ratio)))
        self.n_visible = n_channels - self.n_masked

        # Build networks
        self.encoder = _EEGEncoder(n_channels, n_samples, embed_dim)
        self.target_encoder = _EEGEncoder(n_channels, n_samples, embed_dim)
        self.predictor = _Predictor(
            input_dim=self.n_visible * embed_dim,
            output_dim=self.n_masked * embed_dim,
            hidden_dim=min(256, self.n_visible * embed_dim),
        )

        # Initialize target encoder as a copy of encoder
        self._copy_params(self.encoder, self.target_encoder)
        # Target encoder does not require gradients
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self._loss_history: List[float] = []
        self._is_pretrained = False

        logger.info(
            "JEPAPretrainer initialized: %d channels, %d samples, "
            "embed_dim=%d, mask_ratio=%.2f",
            n_channels, n_samples, embed_dim, mask_ratio,
        )

    # ------------------------------------------------------------------
    # EMA and parameter utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_params(source: "nn.Module", target: "nn.Module") -> None:
        """Copy all parameters from source to target."""
        for s_param, t_param in zip(
            source.parameters(), target.parameters()
        ):
            t_param.data.copy_(s_param.data)

    def _ema_update(self) -> None:
        """Update target encoder with EMA of the online encoder."""
        decay = self.ema_decay
        for s_param, t_param in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)

    def _generate_channel_mask(
        self, batch_size: int
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Generate random channel masks for a batch.

        Returns
        -------
        visible_idx : Tensor, shape (batch_size, n_visible)
        masked_idx : Tensor, shape (batch_size, n_masked)
        """
        visible_list = []
        masked_list = []
        for _ in range(batch_size):
            perm = torch.randperm(self.n_channels)
            visible_list.append(perm[:self.n_visible])
            masked_list.append(perm[self.n_visible:])
        visible_idx = torch.stack(visible_list)
        masked_idx = torch.stack(masked_list)
        return visible_idx, masked_idx

    def _zero_masked_channels(
        self,
        x: "torch.Tensor",
        visible_idx: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Zero out masked channels in the input so the encoder only sees
        visible channels (input shape stays the same for conv compat).
        """
        x_masked = torch.zeros_like(x)
        for i in range(x.shape[0]):
            x_masked[i, visible_idx[i], :] = x[i, visible_idx[i], :]
        return x_masked

    def _gather_channel_embeddings(
        self,
        embeddings: "torch.Tensor",
        indices: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Gather embeddings for selected channel indices.

        Parameters
        ----------
        embeddings : Tensor, shape (batch, n_channels, embed_dim)
        indices : Tensor, shape (batch, n_select)

        Returns
        -------
        selected : Tensor, shape (batch, n_select * embed_dim)
        """
        batch_size = embeddings.shape[0]
        gathered = []
        for i in range(batch_size):
            gathered.append(embeddings[i, indices[i], :].reshape(-1))
        return torch.stack(gathered)

    # ------------------------------------------------------------------
    # Pre-training loop
    # ------------------------------------------------------------------

    def pretrain(
        self,
        unlabeled_data: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 32,
    ) -> Dict:
        """
        Pre-train the encoder on unlabeled EEG data.

        Parameters
        ----------
        unlabeled_data : ndarray, shape (n_windows, n_channels, n_samples)
            Collection of EEG windows. No labels needed.
        n_epochs : int
            Number of training epochs.
        batch_size : int
            Mini-batch size.

        Returns
        -------
        result : dict
            'loss_history': list of per-epoch average losses
            'encoder': the trained encoder module
        """
        _require_torch()

        if unlabeled_data.ndim != 3:
            raise ValueError(
                f"Expected 3-D array (n_windows, n_channels, n_samples), "
                f"got shape {unlabeled_data.shape}"
            )

        n_windows, n_ch, n_samp = unlabeled_data.shape
        if n_ch != self.n_channels or n_samp != self.n_samples:
            raise ValueError(
                f"Shape mismatch: expected "
                f"({self.n_channels}, {self.n_samples}), "
                f"got ({n_ch}, {n_samp})"
            )

        logger.info(
            "Starting JEPA pre-training: %d windows, %d epochs, "
            "batch_size=%d",
            n_windows, n_epochs, batch_size,
        )

        # Prepare data loader
        tensor_data = torch.tensor(unlabeled_data, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        )

        # Optimizer for encoder + predictor only (not target encoder)
        optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.predictor.parameters()),
            lr=self.lr,
        )
        loss_fn = nn.MSELoss()

        self.encoder.train()
        self.predictor.train()
        self.target_encoder.eval()

        self._loss_history = []

        for epoch_idx in range(n_epochs):
            epoch_losses = []

            for (batch,) in loader:
                bs = batch.shape[0]

                # Step a: generate random channel mask
                visible_idx, masked_idx = self._generate_channel_mask(bs)

                # Step b: encode visible channels (masked input)
                x_visible = self._zero_masked_channels(batch, visible_idx)
                context_embeddings = self.encoder(x_visible)
                context_flat = self._gather_channel_embeddings(
                    context_embeddings, visible_idx
                )

                # Step c: encode ALL channels with target encoder (no grad)
                with torch.no_grad():
                    target_embeddings = self.target_encoder(batch)
                target_flat = self._gather_channel_embeddings(
                    target_embeddings, masked_idx
                )

                # Step d: predict masked embeddings from context
                predicted = self.predictor(context_flat)

                # Step e: loss on masked channels only
                loss = loss_fn(predicted, target_flat)

                # Step f: gradient update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Step g: EMA update of target encoder
                self._ema_update()

                epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses))
            self._loss_history.append(avg_loss)

            if (epoch_idx + 1) % max(1, n_epochs // 10) == 0 or epoch_idx == 0:
                logger.info(
                    "Epoch %d/%d -- loss: %.6f",
                    epoch_idx + 1, n_epochs, avg_loss,
                )

        self._is_pretrained = True
        logger.info(
            "Pre-training complete. Final loss: %.6f",
            self._loss_history[-1],
        )

        return {
            "loss_history": self._loss_history,
            "encoder": self.encoder,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_encoder(self) -> "nn.Module":
        """
        Return the trained encoder for downstream use.

        Returns
        -------
        encoder : nn.Module
            The pre-trained 1D CNN encoder.
        """
        if not self._is_pretrained:
            logger.warning(
                "Encoder has not been pre-trained yet. Returning the "
                "randomly initialized encoder."
            )
        return self.encoder

    def extract_features(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Extract features from EEG data using the pre-trained encoder.

        These features can be fed to a simple linear classifier (e.g.,
        logistic regression) instead of training CSP+LDA from scratch,
        significantly reducing the labeled data requirement.

        Parameters
        ----------
        eeg_data : ndarray, shape (n_windows, n_channels, n_samples)
            or (n_channels, n_samples) for a single window.

        Returns
        -------
        features : ndarray, shape (n_windows, n_channels * embed_dim)
            Flattened channel embeddings for each window.
        """
        _require_torch()

        single = eeg_data.ndim == 2
        if single:
            eeg_data = eeg_data[np.newaxis, :, :]

        if eeg_data.ndim != 3:
            raise ValueError(
                f"Expected 2-D or 3-D array, got shape {eeg_data.shape}"
            )

        tensor_data = torch.tensor(eeg_data, dtype=torch.float32)

        self.encoder.eval()
        with torch.no_grad():
            embeddings = self.encoder(tensor_data)  # (n, n_ch, embed_dim)

        features = embeddings.reshape(embeddings.shape[0], -1).numpy()

        if single:
            features = features[0]

        return features
