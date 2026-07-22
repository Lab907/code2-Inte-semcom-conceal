"""Tests for fixed transmitter-side pre-compensation."""

from __future__ import annotations

import torch

from rfhide.fixed_precomp import FixedPrecompensator, complex_to_channels, project_residual_power
from rfhide.modulation import generate_random_bits, modulate_16qam
from rfhide.utils import set_seed


def _cfg() -> dict:
    """Return a compact fixed precomp config."""
    return {
        "signal": {"sample_rate": 1_000_000},
        "impairments": {"num_tx": 3},
        "fixed_precomp": {"max_residual_power_ratio": 0.05},
    }


def _inputs(batch_size: int = 5, num_symbols: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create common model inputs."""
    bits = generate_random_bits(batch_size=batch_size, num_symbols=num_symbols, device="cpu")
    x_clean = modulate_16qam(bits)
    tx_ids = torch.arange(3)
    snr_db = torch.full((batch_size,), 20.0)
    return x_clean, tx_ids, snr_db


def test_fixed_precomp_shape_and_dtype() -> None:
    """Fixed precomp should emit complex residuals in [D, B, N] format."""
    set_seed(501)
    model = FixedPrecompensator(_cfg())
    x_clean, tx_ids, snr_db = _inputs()

    p = model(x_clean, tx_ids, snr_db, time_indices=torch.zeros(3, x_clean.shape[0]))

    assert p.shape == (3, 5, 64)
    assert torch.is_complex(p)


def test_fixed_precomp_projection_limit() -> None:
    """Projected residuals should obey the configured power ratio."""
    set_seed(502)
    model = FixedPrecompensator(_cfg())
    x_clean, tx_ids, snr_db = _inputs()
    with torch.no_grad():
        model.dc_real.fill_(1.0)

    p = model(x_clean, tx_ids, snr_db, time_indices=None)
    ratio = model.residual_power_ratio(p, x_clean)

    assert torch.all(ratio <= 0.05001)


def test_fixed_precomp_is_time_invariant() -> None:
    """Time index should not change fixed precomp residuals."""
    set_seed(503)
    model = FixedPrecompensator(_cfg())
    x_clean, tx_ids, snr_db = _inputs()
    with torch.no_grad():
        model.phase_rad.copy_(torch.tensor([0.01, -0.02, 0.03]))

    p_t0 = model(x_clean, tx_ids, snr_db, torch.zeros(3, x_clean.shape[0]))
    p_t1 = model(x_clean, tx_ids, snr_db, torch.full((3, x_clean.shape[0]), 99.0))

    assert torch.allclose(p_t0, p_t1)


def test_projection_helper_and_channel_conversion() -> None:
    """Utility helpers should preserve expected shapes and limits."""
    set_seed(504)
    x_clean, _, _ = _inputs(batch_size=4, num_symbols=32)
    p_raw = torch.complex(torch.randn_like(x_clean.real), torch.randn_like(x_clean.real))

    p = project_residual_power(p_raw, x_clean, max_residual_power_ratio=0.02)
    ratio = p.abs().pow(2).mean(dim=-1) / x_clean.abs().pow(2).mean(dim=-1).clamp_min(1e-12)
    channels = complex_to_channels(x_clean)

    assert torch.all(ratio <= 0.02001)
    assert channels.shape == (4, 2, 32)
