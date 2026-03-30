"""
GFlowNet for SEAL adaptation strategy optimisation.

Instead of using fixed hyperparameters (blend_ratio=0.3, update_interval=30s),
this module learns the optimal adaptation strategy for each subject by treating
configuration selection as a generative flow problem.

The GFlowNet explores the discrete space of SEAL configurations and samples
new configurations with probability proportional to the observed reward
(classification accuracy), encouraging diverse high-performing strategies.

Reference:
    Bengio, E., Jain, M., Korablyov, M., Precup, D. & Bengio, Y. (2021).
    "Flow Network based Generative Models for Non-Iterative Diverse
    Candidate Generation." NeurIPS 2021.

Requires PyTorch. Guarded with try/except.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from itertools import product
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _require_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for GFlowNetSEALOptimizer but could not "
            "be imported.  Install it with:  pip install torch"
        )


# ======================================================================
# Configuration space
# ======================================================================

# Each dimension is a discrete set of choices
_CONFIG_SPACE = {
    "blend_ratio": [0.1, 0.2, 0.3, 0.4, 0.5],
    "update_interval": [15, 30, 45, 60],
    "min_samples": [3, 5, 10, 15],
    "learning_rate": [1e-5, 1e-4, 1e-3],
}

_PARAM_NAMES = list(_CONFIG_SPACE.keys())
_PARAM_SIZES = [len(_CONFIG_SPACE[k]) for k in _PARAM_NAMES]
_TOTAL_CONFIGS = int(np.prod(_PARAM_SIZES))


def _config_to_indices(config: Dict) -> List[int]:
    """Map a config dict to a list of per-dimension indices."""
    indices = []
    for name in _PARAM_NAMES:
        val = config[name]
        idx = _CONFIG_SPACE[name].index(val)
        indices.append(idx)
    return indices


def _indices_to_config(indices: List[int]) -> Dict:
    """Map per-dimension indices back to a config dict."""
    return {
        name: _CONFIG_SPACE[name][idx]
        for name, idx in zip(_PARAM_NAMES, indices)
    }


def _config_to_flat(config: Dict) -> int:
    """Map a config dict to a single flat index in [0, _TOTAL_CONFIGS)."""
    flat = 0
    stride = 1
    for name in reversed(_PARAM_NAMES):
        idx = _CONFIG_SPACE[name].index(config[name])
        flat += idx * stride
        stride *= len(_CONFIG_SPACE[name])
    return flat


def _flat_to_config(flat: int) -> Dict:
    """Map a flat index back to a config dict."""
    config = {}
    for name in reversed(_PARAM_NAMES):
        size = len(_CONFIG_SPACE[name])
        config[name] = _CONFIG_SPACE[name][flat % size]
        flat //= size
    return config


# ======================================================================
# Flow network
# ======================================================================

if _TORCH_AVAILABLE:

    class _FlowNetwork(nn.Module):
        """Parameterises the forward and backward policies plus log Z.

        The state is encoded as a one-hot vector per configuration
        dimension, concatenated into a single input vector.

        Forward policy:  P_F(s' | s)  — probability of transitioning to
                         a new configuration given the current one.
        Backward policy: P_B(s | s')  — reverse direction for detailed
                         balance.
        """

        def __init__(self, d_hidden: int = 128):
            super().__init__()
            self.d_state = sum(_PARAM_SIZES)

            # Shared encoder
            self.encoder = nn.Sequential(
                nn.Linear(self.d_state, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
            )

            # Forward policy head — outputs logits over all configs
            self.forward_head = nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, _TOTAL_CONFIGS),
            )

            # Backward policy head
            self.backward_head = nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, _TOTAL_CONFIGS),
            )

            # Learnable log-partition function
            self.log_Z = nn.Parameter(torch.tensor(0.0))

        def _encode_state(self, config: Dict) -> torch.Tensor:
            """One-hot encode a config into a state vector."""
            parts = []
            for name in _PARAM_NAMES:
                size = len(_CONFIG_SPACE[name])
                idx = _CONFIG_SPACE[name].index(config[name])
                oh = torch.zeros(size)
                oh[idx] = 1.0
                parts.append(oh)
            return torch.cat(parts).unsqueeze(0)  # (1, d_state)

        def forward_policy(self, config: Dict) -> torch.Tensor:
            """Return log P_F(s' | s) over all possible next configs."""
            state = self._encode_state(config)
            h = self.encoder(state)
            logits = self.forward_head(h)
            return F.log_softmax(logits, dim=-1).squeeze(0)

        def backward_policy(self, config: Dict) -> torch.Tensor:
            """Return log P_B(s | s') over all possible previous configs."""
            state = self._encode_state(config)
            h = self.encoder(state)
            logits = self.backward_head(h)
            return F.log_softmax(logits, dim=-1).squeeze(0)


# ======================================================================
# Public API
# ======================================================================


class GFlowNetSEALOptimizer:
    """GFlowNet-based optimizer for SEAL adaptation hyperparameters.

    Instead of fixing blend_ratio, update_interval, min_samples, and
    learning_rate, this module learns to sample configurations whose
    long-term reward (classification accuracy) is high, while
    maintaining diversity via the flow-matching objective.

    Parameters
    ----------
    config : dict or None
        Optional overrides.  Recognised keys:
        - 'd_hidden'    (int, default 128)
        - 'lr'          (float, default 1e-3)
        - 'temperature' (float, default 1.0)
        - 'reward_ema'  (float, default 0.9)  — EMA smoothing for reward
    """

    def __init__(self, config: Optional[Dict] = None):
        _require_torch()
        cfg = config or {}
        self.d_hidden = cfg.get("d_hidden", 128)
        self.lr = cfg.get("lr", 1e-3)
        self.temperature = cfg.get("temperature", 1.0)
        self.reward_ema_alpha = cfg.get("reward_ema", 0.9)

        self._net = _FlowNetwork(d_hidden=self.d_hidden)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        # Tracking
        self._reward_ema: Optional[float] = None
        self._history: List[Dict] = []
        self._best_config: Optional[Dict] = None
        self._best_reward: float = -np.inf

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def propose_config(
        self,
        current_accuracy: float,
        n_actions: int,
    ) -> Dict:
        """Sample a new SEAL configuration from the forward policy.

        Early in training the policy is close to uniform, encouraging
        exploration; as training progresses it concentrates on high-reward
        regions while maintaining diversity.

        Parameters
        ----------
        current_accuracy : float
            Most recent classification accuracy (0-1).
        n_actions : int
            Number of actions (classified trials) in the current session.

        Returns
        -------
        config : dict with keys blend_ratio, update_interval,
                 min_samples, learning_rate.
        """
        self._net.eval()

        # Use a default starting config for the forward policy input
        current_config = self._best_config or _flat_to_config(0)

        with torch.no_grad():
            log_probs = self._net.forward_policy(current_config)
            # Temperature scaling for exploration
            scaled = log_probs / max(self.temperature, 1e-6)
            probs = torch.softmax(scaled, dim=-1)

            # Add epsilon noise early on for exploration
            explore_eps = max(0.1 * (1.0 - n_actions / 500.0), 0.01)
            uniform = torch.ones_like(probs) / _TOTAL_CONFIGS
            probs = (1.0 - explore_eps) * probs + explore_eps * uniform

            flat_idx = torch.multinomial(probs, num_samples=1).item()

        config = _flat_to_config(flat_idx)
        logger.debug(
            "GFlowNet proposed config (acc=%.3f, n=%d): %s",
            current_accuracy,
            n_actions,
            config,
        )
        return config

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        config_used: Dict,
        accuracy_before: float,
        accuracy_after: float,
    ) -> float:
        """Update the flow network after observing a reward.

        Uses the trajectory balance objective:
            log Z + sum log P_F = log R + sum log P_B

        For our single-step transitions this simplifies to:
            loss = (log Z + log P_F(s' | s) - log R(s') - log P_B(s | s'))^2

        Parameters
        ----------
        config_used : the SEAL config that was applied
        accuracy_before : accuracy before applying this config
        accuracy_after : accuracy after applying this config

        Returns
        -------
        loss_value : float, the trajectory balance loss
        """
        # Compute reward as accuracy improvement + absolute accuracy
        delta = accuracy_after - accuracy_before
        reward = accuracy_after + max(delta, 0.0)
        reward = max(reward, 1e-6)  # ensure positive for log

        # Update EMA reward
        if self._reward_ema is None:
            self._reward_ema = reward
        else:
            alpha = self.reward_ema_alpha
            self._reward_ema = alpha * self._reward_ema + (1 - alpha) * reward

        # Track best
        if reward > self._best_reward:
            self._best_reward = reward
            self._best_config = dict(config_used)

        # Record history
        self._history.append({
            "config": dict(config_used),
            "accuracy_before": accuracy_before,
            "accuracy_after": accuracy_after,
            "reward": reward,
        })

        # Trajectory balance loss
        self._net.train()
        self._optimizer.zero_grad()

        # For single-step: source state is previous best (or default)
        source = self._best_config if self._best_config != config_used else _flat_to_config(0)
        target = config_used
        target_flat = _config_to_flat(target)
        source_flat = _config_to_flat(source)

        log_pf = self._net.forward_policy(source)       # (TOTAL_CONFIGS,)
        log_pb = self._net.backward_policy(target)       # (TOTAL_CONFIGS,)

        log_Z = self._net.log_Z
        log_R = torch.tensor(np.log(reward), dtype=torch.float32)

        # Trajectory balance: log Z + log P_F(s'|s) = log R(s') + log P_B(s|s')
        lhs = log_Z + log_pf[target_flat]
        rhs = log_R + log_pb[source_flat]
        loss = (lhs - rhs) ** 2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
        self._optimizer.step()

        loss_val = loss.item()
        logger.debug("GFlowNet TB loss: %.4f  reward: %.4f", loss_val, reward)
        return loss_val

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_best_config(self) -> Dict:
        """Return the highest-reward configuration observed so far.

        Returns
        -------
        config : dict with keys blend_ratio, update_interval,
                 min_samples, learning_rate.
                 Returns a default config if no updates have been made.
        """
        if self._best_config is not None:
            return dict(self._best_config)
        # Sensible defaults matching the original SEAL parameters
        return {
            "blend_ratio": 0.3,
            "update_interval": 30,
            "min_samples": 5,
            "learning_rate": 1e-4,
        }

    def get_stats(self) -> Dict:
        """Return summary statistics of the optimisation history.

        Returns
        -------
        stats : dict with keys
            - 'n_updates': number of update() calls
            - 'reward_ema': exponential moving average of reward
            - 'best_reward': highest reward observed
            - 'best_config': corresponding configuration
            - 'log_Z': current learned log-partition estimate
        """
        return {
            "n_updates": len(self._history),
            "reward_ema": self._reward_ema if self._reward_ema is not None else 0.0,
            "best_reward": float(self._best_reward),
            "best_config": self.get_best_config(),
            "log_Z": float(self._net.log_Z.item()),
        }
