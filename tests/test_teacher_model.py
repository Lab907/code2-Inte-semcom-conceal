"""Tests for the Step 4 fixed pre-compensator."""

from __future__ import annotations

import torch

from rfhide.fixed_precomp import FixedPrecompensator
from rfhide.modulation import generate_random_bits, modulate_16qam
from rfhide.utils import set_seed


def _cfg() -> dict:
    """Return a compact fixed precomp config for unit tests."""
    return {
        "impairments": {"num_tx": 3},
        "fixed_precomp": {
            "max_residual_power_ratio": 0.05,
        },
    }


def _inputs(batch_size: int = 4, num_symbols: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create common teacher forward inputs."""
    bits = generate_random_bits(batch_size=batch_size, num_symbols=num_symbols, device="cpu")
    x_clean = modulate_16qam(bits)
    tx_ids = torch.arange(3)
    snr_db = torch.full((batch_size,), 20.0)
    time_indices = torch.arange(batch_size, dtype=torch.float32).unsqueeze(0).expand(3, -1)
    return x_clean, tx_ids, snr_db, time_indices


def test_fixed_precomp_output_shape() -> None:
    """Fixed precomp output should be complex residuals with shape [3, B, N]."""
    set_seed(301)
    model = FixedPrecompensator(_cfg())
    x_clean, tx_ids, snr_db, time_indices = _inputs()

    p = model(x_clean, tx_ids, snr_db, time_indices)

    assert p.shape == (3, 4, 64)
    assert torch.is_complex(p)


def test_residual_power_projection_is_enforced() -> None:
    """Residual power ratio should stay below the configured threshold."""
    set_seed(302)
    model = FixedPrecompensator(_cfg())
    x_clean, tx_ids, snr_db, time_indices = _inputs()

    p = model(x_clean, tx_ids, snr_db, time_indices)
    ratio = model.residual_power_ratio(p, x_clean)

    assert torch.all(ratio <= 0.05001)


def test_different_tx_parameters_can_diverge() -> None:
    """Tx-specific parameters should allow different residuals for devices."""
    set_seed(303)
    model = FixedPrecompensator(_cfg())
    x_clean, tx_ids, snr_db, time_indices = _inputs()

    with torch.no_grad():
        model.phase_rad.copy_(torch.tensor([0.01, -0.02, 0.03]))
    p = model(x_clean, tx_ids, snr_db, time_indices)

    assert not torch.allclose(p[0], p[1])
    assert not torch.allclose(p[1], p[2])


def test_same_tx_ignores_time_index() -> None:
    """The same transmitter should produce the same residual for different times."""
    set_seed(304)
    model = FixedPrecompensator(_cfg())
    x_clean, tx_ids, snr_db, time_indices = _inputs()

    with torch.no_grad():
        model.phase_rad[0] = 0.02
    p_t0 = model(x_clean, tx_ids[:1], snr_db, torch.zeros_like(time_indices[:1]))
    p_t1 = model(x_clean, tx_ids[:1], snr_db, torch.full_like(time_indices[:1], 25.0))

    assert torch.allclose(p_t0, p_t1)


def test_forward_loss_backward_has_nonzero_gradients() -> None:
    """A forward pass and scalar loss should backpropagate nonzero gradients."""
    set_seed(305)
    model = FixedPrecompensator(_cfg())
    x_clean, tx_ids, snr_db, time_indices = _inputs()

    p = model(x_clean, tx_ids, snr_db, time_indices)
    target = 0.01 * x_clean.unsqueeze(0).expand_as(p)
    loss = (p - target).abs().pow(2).mean()
    loss.backward()
    grad_norm = sum(
        param.grad.abs().sum().item()
        for param in model.parameters()
        if param.grad is not None
    )

    assert grad_norm > 0.0
