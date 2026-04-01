"""
TFT-inspired variable selection for EEG features.

Learns which time steps, channels, and frequency bands matter most for each
classification decision using the Gated Residual Network (GRN) architecture
from the Temporal Fusion Transformer.

Reference:
    Lim, B., Arik, S.O., Loeff, N. & Pfister, T. (2021).
    "Temporal Fusion Transformers for Interpretable Multi-horizon
    Time Series Forecasting." International Journal of Forecasting,
    37(4), 1748-1764.

The variable selection mechanism computes softmax-weighted importance scores
over GRN outputs, yielding per-variable explanations for every prediction.
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch():
    """Raise a clear error when PyTorch is unavailable."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for VariableSelector but could not be "
            "imported.  Install it with:  pip install torch"
        )


# ======================================================================
# Building blocks
# ======================================================================

if _TORCH_AVAILABLE:

    class _GLU(nn.Module):
        """Gated Linear Unit: sigma(x_1) * x_2."""

        def __init__(self, d_model: int):
            super().__init__()
            self.fc = nn.Linear(d_model, d_model * 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.fc(x)
            x1, x2 = out.chunk(2, dim=-1)
            return torch.sigmoid(x1) * x2

    class _GRN(nn.Module):
        """Gated Residual Network.

        GRN(a, c) = LayerNorm(a + GLU(W1 * eta1 + W2 * c + b))
        where eta1 = ELU(W3 * a + W4 * c + b3)
        """

        def __init__(self, d_input: int, d_hidden: int, d_context: int = 0):
            super().__init__()
            self.fc1 = nn.Linear(d_input, d_hidden)
            self.fc_context = (
                nn.Linear(d_context, d_hidden, bias=False)
                if d_context > 0
                else None
            )
            self.fc2 = nn.Linear(d_hidden, d_input)
            self.glu = _GLU(d_input)
            self.layer_norm = nn.LayerNorm(d_input)

        def forward(
            self, a: torch.Tensor, c: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            eta = self.fc1(a)
            if self.fc_context is not None and c is not None:
                eta = eta + self.fc_context(c)
            eta = F.elu(eta)
            eta = self.fc2(eta)
            gate_out = self.glu(eta)
            return self.layer_norm(a + gate_out)

    class _VariableSelectionNetwork(nn.Module):
        """Softmax over per-variable GRNs to produce importance weights."""

        def __init__(
            self,
            n_variables: int,
            d_var: int,
            d_hidden: int,
            d_context: int = 0,
        ):
            super().__init__()
            self.n_variables = n_variables
            self.d_var = d_var

            # One GRN per variable to produce a scalar importance score
            self.grns = nn.ModuleList(
                [_GRN(d_var, d_hidden, d_context) for _ in range(n_variables)]
            )
            # Joint GRN that computes the selection weights
            self.weight_grn = _GRN(
                n_variables, d_hidden, d_context
            )

        def forward(
            self,
            variables: torch.Tensor,
            context: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Parameters
            ----------
            variables : (batch, n_variables, d_var)
            context   : (batch, d_context) or None

            Returns
            -------
            combined  : (batch, d_var) — weighted sum of transformed variables
            weights   : (batch, n_variables) — per-variable importance weights
            """
            # Transform each variable through its own GRN
            transformed = []
            for i in range(self.n_variables):
                xi = variables[:, i, :]  # (batch, d_var)
                transformed.append(self.grns[i](xi, context))
            transformed = torch.stack(transformed, dim=1)  # (batch, n_var, d_var)

            # Compute importance weights
            # Flatten variables to scalars per variable via mean pooling
            var_summaries = transformed.mean(dim=-1)  # (batch, n_var)
            weight_input = self.weight_grn(var_summaries, context)
            weights = F.softmax(weight_input, dim=-1)  # (batch, n_var)

            # Weighted combination
            combined = (transformed * weights.unsqueeze(-1)).sum(dim=1)
            return combined, weights

    class _Classifier(nn.Module):
        """End-to-end variable selection + classification head."""

        def __init__(
            self,
            n_variables: int,
            d_var: int,
            d_hidden: int,
            n_classes: int,
        ):
            super().__init__()
            self.vsn = _VariableSelectionNetwork(
                n_variables, d_var, d_hidden
            )
            self.head = nn.Sequential(
                nn.Linear(d_var, d_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(d_hidden, n_classes),
            )

        def forward(
            self, variables: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            combined, weights = self.vsn(variables)
            logits = self.head(combined)
            return logits, weights


# ======================================================================
# Public API
# ======================================================================


class VariableSelector:
    """TFT-inspired variable selection for EEG feature sets.

    Learns which variables (channels, frequency bands, time steps) are
    most informative for each motor-imagery class, using a Gated Residual
    Network with softmax variable weighting.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_features : int
        Dimensionality of the feature vector per channel / variable.
    n_classes : int
        Number of target classes (default 5 for OpenBCI MI).
    d_hidden : int
        Hidden layer size inside the GRN (default 64).
    lr : float
        Learning rate for Adam optimiser (default 1e-3).
    """

    def __init__(
        self,
        n_channels: int,
        n_features: int,
        n_classes: int = 5,
        d_hidden: int = 64,
        lr: float = 1e-3,
    ):
        _require_torch()
        self.n_channels = n_channels
        self.n_features = n_features
        self.n_classes = n_classes
        self.d_hidden = d_hidden
        self.lr = lr

        self._model = _Classifier(
            n_variables=n_channels,
            d_var=n_features,
            d_hidden=d_hidden,
            n_classes=n_classes,
        )
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        self._loss_fn = nn.CrossEntropyLoss()
        self._fitted = False
        self._importance_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_features: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = False,
    ) -> "VariableSelector":
        """Train the variable selection network.

        Parameters
        ----------
        X_features : shape (n_samples, n_channels, n_features)
        y : shape (n_samples,), integer class labels
        epochs : number of training epochs
        batch_size : mini-batch size
        verbose : print loss every 10 epochs

        Returns
        -------
        self
        """
        X_t = torch.tensor(X_features, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self._optimizer.zero_grad()
                logits, _ = self._model(xb)
                loss = self._loss_fn(logits, yb)
                loss.backward()
                self._optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            if verbose and (epoch + 1) % 10 == 0:
                avg = epoch_loss / len(dataset)
                print(f"Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

        self._fitted = True
        self._importance_cache = None
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def transform(self, X_features: np.ndarray) -> np.ndarray:
        """Apply learned importance weights to features.

        Parameters
        ----------
        X_features : shape (n_samples, n_channels, n_features)

        Returns
        -------
        weighted : shape (n_samples, n_features) — importance-weighted
            combination of per-channel features.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_features, dtype=torch.float32)
            combined, _ = self._model.vsn(X_t)
        return combined.numpy()

    def get_importance(self) -> Dict[str, float]:
        """Return average importance weight per channel / variable.

        Returns
        -------
        importance : dict mapping "channel_<i>" to its average weight
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before get_importance().")
        if self._importance_cache is not None:
            weights = self._importance_cache
        else:
            # Run a forward pass on a dummy input to extract static weights
            self._model.eval()
            with torch.no_grad():
                dummy = torch.randn(1, self.n_channels, self.n_features)
                _, w = self._model(dummy)
                weights = w.squeeze(0).numpy()
            self._importance_cache = weights

        return {
            f"channel_{i}": float(weights[i])
            for i in range(self.n_channels)
        }

    def explain(self, X_single: np.ndarray) -> Dict:
        """Per-sample explanation of which features drove the decision.

        Parameters
        ----------
        X_single : shape (n_channels, n_features) — one sample

        Returns
        -------
        explanation : dict with keys
            - 'predicted_class': int
            - 'class_probabilities': np.ndarray of shape (n_classes,)
            - 'variable_weights': dict mapping channel name to weight
            - 'top_variable': name of the highest-weighted channel
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before explain().")
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_single, dtype=torch.float32).unsqueeze(0)
            logits, weights = self._model(X_t)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
            w = weights.squeeze(0).numpy()

        var_weights = {
            f"channel_{i}": float(w[i]) for i in range(self.n_channels)
        }
        top = max(var_weights, key=var_weights.get)

        return {
            "predicted_class": int(np.argmax(probs)),
            "class_probabilities": probs,
            "variable_weights": var_weights,
            "top_variable": top,
        }
