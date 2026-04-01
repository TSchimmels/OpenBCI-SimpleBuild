"""Tests for UncertaintyWeightedLoss module."""

import numpy as np
import pytest

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

pytestmark = pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")


@pytest.fixture
def uw_kendall():
    from src.training.uncertainty_weights import UncertaintyWeightedLoss
    return UncertaintyWeightedLoss(["ce", "kl", "sparsity"], method="kendall")


@pytest.fixture
def uw_analytical():
    from src.training.uncertainty_weights import UncertaintyWeightedLoss
    return UncertaintyWeightedLoss(["ce", "kl", "sparsity"], method="analytical")


class TestInit:
    def test_create_kendall(self, uw_kendall):
        assert uw_kendall.method == "kendall"
        assert uw_kendall.task_names == ["ce", "kl", "sparsity"]

    def test_create_analytical(self, uw_analytical):
        assert uw_analytical.method == "analytical"

    def test_empty_names_raises(self):
        from src.training.uncertainty_weights import UncertaintyWeightedLoss
        with pytest.raises(ValueError, match="at least one"):
            UncertaintyWeightedLoss([])

    def test_duplicate_names_raises(self):
        from src.training.uncertainty_weights import UncertaintyWeightedLoss
        with pytest.raises(ValueError, match="unique"):
            UncertaintyWeightedLoss(["a", "a"])

    def test_invalid_method_raises(self):
        from src.training.uncertainty_weights import UncertaintyWeightedLoss
        with pytest.raises(ValueError, match="method"):
            UncertaintyWeightedLoss(["a"], method="bogus")

    def test_custom_initial_log_vars(self):
        from src.training.uncertainty_weights import UncertaintyWeightedLoss
        uw = UncertaintyWeightedLoss(
            ["a", "b"], initial_log_vars={"a": -1.0, "b": 1.0}
        )
        weights = uw.get_weights()
        # "a" has lower log-var -> higher weight
        assert weights["a"] > weights["b"]


class TestForward:
    def test_kendall_returns_scalar(self, uw_kendall):
        losses = {
            "ce": torch.tensor(1.0, requires_grad=True),
            "kl": torch.tensor(0.5, requires_grad=True),
            "sparsity": torch.tensor(0.1, requires_grad=True),
        }
        total, info = uw_kendall(losses)
        assert total.dim() == 0  # scalar
        assert total.requires_grad

    def test_analytical_returns_scalar(self, uw_analytical):
        losses = {
            "ce": torch.tensor(1.0, requires_grad=True),
            "kl": torch.tensor(0.5, requires_grad=True),
            "sparsity": torch.tensor(0.1, requires_grad=True),
        }
        total, info = uw_analytical(losses)
        assert total.dim() == 0
        assert total.requires_grad

    def test_info_contains_keys(self, uw_analytical):
        losses = {
            "ce": torch.tensor(1.0),
            "kl": torch.tensor(0.5),
            "sparsity": torch.tensor(0.1),
        }
        _, info = uw_analytical(losses)
        assert "effective_weights" in info
        assert "log_vars" in info
        assert "raw_losses" in info
        assert set(info["effective_weights"].keys()) == {"ce", "kl", "sparsity"}

    def test_missing_loss_raises(self, uw_analytical):
        losses = {"ce": torch.tensor(1.0), "kl": torch.tensor(0.5)}
        with pytest.raises(KeyError, match="sparsity"):
            uw_analytical(losses)

    def test_gradients_flow_to_log_vars(self, uw_analytical):
        losses = {
            "ce": torch.tensor(2.0, requires_grad=True),
            "kl": torch.tensor(1.0, requires_grad=True),
            "sparsity": torch.tensor(0.5, requires_grad=True),
        }
        total, _ = uw_analytical(losses)
        total.backward()
        for name in uw_analytical.task_names:
            param = uw_analytical._log_vars[name]
            assert param.grad is not None
            assert param.grad.item() != 0.0


class TestAnalyticalWeightsSumToOne:
    def test_weights_sum_approximately_one(self, uw_analytical):
        weights = uw_analytical.get_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-5


class TestLearning:
    def test_weights_adapt_to_loss_magnitudes(self):
        """After optimization, the task with larger loss should get lower weight."""
        from src.training.uncertainty_weights import UncertaintyWeightedLoss

        uw = UncertaintyWeightedLoss(["big", "small"], method="analytical")
        optimizer = torch.optim.Adam(uw.parameters(), lr=0.1)

        for _ in range(100):
            optimizer.zero_grad()
            losses = {
                "big": torch.tensor(100.0, requires_grad=True),
                "small": torch.tensor(0.01, requires_grad=True),
            }
            total, _ = uw(losses)
            total.backward()
            optimizer.step()

        weights = uw.get_weights()
        # The "big" loss should get lower weight to balance
        assert weights["big"] < weights["small"]

    def test_equal_losses_equal_weights(self):
        """Equal losses should converge to roughly equal weights."""
        from src.training.uncertainty_weights import UncertaintyWeightedLoss

        uw = UncertaintyWeightedLoss(["a", "b", "c"], method="analytical")
        optimizer = torch.optim.Adam(uw.parameters(), lr=0.05)

        for _ in range(200):
            optimizer.zero_grad()
            losses = {
                "a": torch.tensor(1.0, requires_grad=True),
                "b": torch.tensor(1.0, requires_grad=True),
                "c": torch.tensor(1.0, requires_grad=True),
            }
            total, _ = uw(losses)
            total.backward()
            optimizer.step()

        weights = uw.get_weights()
        values = list(weights.values())
        # All weights should be roughly 0.333
        assert all(abs(v - 1.0 / 3) < 0.15 for v in values)


class TestRepr:
    def test_repr_contains_method(self, uw_analytical):
        r = repr(uw_analytical)
        assert "analytical" in r
        assert "ce" in r
