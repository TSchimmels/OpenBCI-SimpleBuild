"""Uncertainty-weighted multi-objective loss balancing.

Learns per-task loss weights automatically via homoscedastic uncertainty,
eliminating the need for manual weight tuning in multi-loss training.

Two formulations are provided:

1. **Kendall (CVPR 2018):** Original uncertainty weighting.
   Each loss L_i is weighted by a learnable log-variance s_i:
       L_total = sum_i [ (1 / (2 * exp(s_i))) * L_i + 0.5 * s_i ]

2. **Analytical (arXiv 2408.07985, 2024):** Improved formulation.
   Uses softmax-normalized weights with tunable temperature for
   more stable convergence. Consistently outperforms 6 other
   weighting methods including the original Kendall approach.

Usage::

    uw = UncertaintyWeightedLoss(["ce", "kl", "sparsity"])
    # In training loop:
    losses = {"ce": ce_loss, "kl": kl_loss, "sparsity": sparsity_loss}
    total, info = uw(losses)
    total.backward()
    # info contains per-task effective weights for logging

Cross-project: Designed for reuse across OpenBCI SimpleBuild,
CANDLE_LIT, AlphaChaosQuant, and ISAC-Systems.

References:
    Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning
    Using Uncertainty to Weigh Losses. CVPR.

    Analytical Uncertainty-Based Loss Weighting in Multi-Task Learning.
    arXiv:2408.07985, 2024.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for UncertaintyWeightedLoss. "
            "Install with: pip install torch"
        )


# Conditional base class for no-torch environments
_TorchModule = nn.Module if _TORCH_AVAILABLE else object


class UncertaintyWeightedLoss(_TorchModule):
    """Learnable multi-objective loss balancer via uncertainty weighting.

    Wraps N loss terms and learns N scalar weights that automatically
    balance the loss magnitudes during training. Losses that are noisier
    (higher uncertainty) get lower weight; losses that are more precise
    get higher weight.

    The module adds N learnable parameters (log-variances) to the
    optimizer's parameter list. These converge alongside the model
    parameters with zero additional hyperparameter tuning.

    Args:
        task_names: List of loss names (e.g., ``["ce", "kl", "sparsity"]``).
        method: ``"kendall"`` (original, CVPR 2018) or ``"analytical"``
            (improved, arXiv 2408.07985). Default: ``"analytical"``.
        temperature: Softmax temperature for the analytical method.
            Lower = sharper weight distribution. Default: ``1.0``.
        initial_log_vars: Optional dict of ``{name: initial_value}``
            for the log-variance parameters. Default: ``0.0`` for all.

    Example::

        uw = UncertaintyWeightedLoss(["pred", "lyap", "basin", "koopman"])
        optimizer = Adam(list(model.parameters()) + list(uw.parameters()))

        for batch in dataloader:
            losses = {
                "pred": prediction_loss,
                "lyap": lyapunov_loss,
                "basin": basin_loss,
                "koopman": koopman_loss,
            }
            total, info = uw(losses)
            total.backward()
            optimizer.step()

            # info["effective_weights"] = {"pred": 0.42, "lyap": 0.18, ...}
            # info["log_vars"] = {"pred": -0.31, "lyap": 0.55, ...}
    """

    def __init__(
        self,
        task_names: List[str],
        method: str = "analytical",
        temperature: float = 1.0,
        initial_log_vars: Optional[Dict[str, float]] = None,
    ) -> None:
        _require_torch()
        super().__init__()

        if len(task_names) == 0:
            raise ValueError("task_names must contain at least one name")
        if len(task_names) != len(set(task_names)):
            raise ValueError("task_names must be unique")
        if method not in ("kendall", "analytical"):
            raise ValueError(f"method must be 'kendall' or 'analytical', got '{method}'")

        self._task_names = list(task_names)
        self._method = method
        self._temperature = temperature

        # Learnable log-variance per task
        init_vals = initial_log_vars or {}
        self._log_vars = nn.ParameterDict({
            name: nn.Parameter(
                torch.tensor(init_vals.get(name, 0.0), dtype=torch.float32)
            )
            for name in self._task_names
        })

        logger.info(
            "UncertaintyWeightedLoss: %d tasks (%s), method=%s, temperature=%.2f",
            len(self._task_names),
            ", ".join(self._task_names),
            self._method,
            self._temperature,
        )

    def forward(
        self, losses: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", Dict]:
        """Compute the uncertainty-weighted total loss.

        Args:
            losses: Dict mapping task name to its scalar loss tensor.
                Must contain exactly the names passed to ``__init__``.

        Returns:
            (total_loss, info) where info contains:
                - ``effective_weights``: dict of per-task weights (detached)
                - ``log_vars``: dict of current log-variance values (detached)
                - ``raw_losses``: dict of per-task loss values (detached)
        """
        # Validate all expected tasks are present
        missing = set(self._task_names) - set(losses.keys())
        if missing:
            raise KeyError(
                f"Missing losses: {missing}. "
                f"Expected: {self._task_names}"
            )

        if self._method == "kendall":
            total, weights = self._kendall_forward(losses)
        else:
            total, weights = self._analytical_forward(losses)

        # Build info dict for logging
        info = {
            "effective_weights": {
                name: weights[name].detach().item()
                for name in self._task_names
            },
            "log_vars": {
                name: self._log_vars[name].detach().item()
                for name in self._task_names
            },
            "raw_losses": {
                name: losses[name].detach().item()
                for name in self._task_names
            },
        }

        return total, info

    def _kendall_forward(
        self, losses: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """Kendall et al. (2018) uncertainty weighting.

        L_total = sum_i [ (1 / (2 * exp(s_i))) * L_i + 0.5 * s_i ]

        The regularisation term (0.5 * s_i) prevents the weights from
        growing unbounded — increasing s_i reduces the loss contribution
        but increases the penalty.
        """
        total = torch.tensor(0.0, device=next(iter(losses.values())).device)
        weights = {}

        for name in self._task_names:
            log_var = self._log_vars[name]
            precision = torch.exp(-log_var)  # 1 / exp(s_i) = exp(-s_i)
            w = 0.5 * precision
            total = total + w * losses[name] + 0.5 * log_var
            weights[name] = w.detach()

        return total, weights

    def _analytical_forward(
        self, losses: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        """Analytical uncertainty weighting (arXiv 2408.07985, 2024).

        Uses softmax-normalized inverse-variance weights with
        temperature scaling for more stable convergence:

            w_i = softmax(-log_var_i / temperature)
            L_total = sum_i [ w_i * L_i + 0.5 * log_var_i ]

        The softmax ensures weights sum to 1, preventing the
        degenerate solution where all variances grow large.
        """
        device = next(iter(losses.values())).device

        # Stack log-vars and compute softmax weights
        log_var_tensor = torch.stack([
            self._log_vars[name] for name in self._task_names
        ])

        # Negative log-vars through softmax: lower variance = higher weight
        raw_weights = torch.softmax(
            -log_var_tensor / max(self._temperature, 1e-6), dim=0
        )

        total = torch.tensor(0.0, device=device)
        weights = {}

        for i, name in enumerate(self._task_names):
            w = raw_weights[i]
            total = total + w * losses[name] + 0.5 * self._log_vars[name]
            weights[name] = w.detach()

        return total, weights

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def task_names(self) -> List[str]:
        """Return the list of task names."""
        return list(self._task_names)

    @property
    def method(self) -> str:
        """Return the weighting method."""
        return self._method

    def get_weights(self) -> Dict[str, float]:
        """Return current effective weights (detached, for logging).

        Returns:
            Dict mapping task name to its current weight.
        """
        with torch.no_grad():
            if self._method == "kendall":
                return {
                    name: (0.5 * torch.exp(-self._log_vars[name])).item()
                    for name in self._task_names
                }
            else:
                log_var_tensor = torch.stack([
                    self._log_vars[name] for name in self._task_names
                ])
                w = torch.softmax(
                    -log_var_tensor / max(self._temperature, 1e-6), dim=0
                )
                return {
                    name: w[i].item()
                    for i, name in enumerate(self._task_names)
                }

    def __repr__(self) -> str:
        weights = self.get_weights() if _TORCH_AVAILABLE else {}
        w_str = ", ".join(f"{k}={v:.3f}" for k, v in weights.items())
        return (
            f"UncertaintyWeightedLoss(method='{self._method}', "
            f"tasks={self._task_names}, weights=[{w_str}])"
        )
