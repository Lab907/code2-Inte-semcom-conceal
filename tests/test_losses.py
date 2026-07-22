"""Tests for Step 3 feature extraction and loss functions."""

from __future__ import annotations

import torch

from rfhide.features import extract_raw_features
from rfhide.losses import (
    cov_alignment_loss,
    device_distribution_alignment,
    evm_loss,
    mean_alignment_loss,
    power_loss,
    rbf_mmd_loss,
    soft_bit_loss,
)
from rfhide.modulation import generate_random_bits, modulate_16qam
from rfhide.utils import set_seed


def test_alignment_loss_zero_for_identical_features() -> None:
    """Identical transmitter features should have near-zero alignment loss."""
    set_seed(201)
    base = torch.randn(6, 12)
    features = base.unsqueeze(0).expand(3, -1, -1).contiguous()

    assert torch.allclose(mean_alignment_loss(features), torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(cov_alignment_loss(features), torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(device_distribution_alignment(features), torch.tensor(0.0), atol=1e-6)


def test_mean_alignment_loss_increases_with_tx_offsets() -> None:
    """Adding different Tx offsets should increase mean alignment loss."""
    set_seed(202)
    base = torch.randn(8, 10)
    identical = base.unsqueeze(0).expand(3, -1, -1).contiguous()
    offsets = torch.tensor([0.0, 1.0, 2.0]).view(3, 1, 1)
    shifted = identical + offsets

    assert mean_alignment_loss(shifted) > mean_alignment_loss(identical)


def test_rbf_mmd_is_nonnegative_and_symmetric() -> None:
    """Biased multi-kernel MMD should be nonnegative and symmetric."""
    set_seed(203)
    x = torch.randn(9, 7)
    y = torch.randn(11, 7) + 0.25

    mmd_xy = rbf_mmd_loss(x, y)
    mmd_yx = rbf_mmd_loss(y, x)

    assert mmd_xy >= 0
    assert torch.allclose(mmd_xy, mmd_yx, atol=1e-7)


def test_soft_bit_loss_clean_symbols_lower_than_noisy_symbols() -> None:
    """The differentiable demapper should prefer clean 16-QAM symbols."""
    set_seed(204)
    bits = generate_random_bits(batch_size=12, num_symbols=64, device="cpu")
    clean = modulate_16qam(bits)
    noise = 0.35 * torch.complex(torch.randn_like(clean.real), torch.randn_like(clean.real))
    noisy = clean + noise
    flat_bits = bits.reshape(bits.shape[0], -1)

    clean_loss = soft_bit_loss(clean, flat_bits, cfg={"soft_bit_temperature": 0.05})
    noisy_loss = soft_bit_loss(noisy, flat_bits, cfg={"soft_bit_temperature": 0.05})

    assert clean_loss < noisy_loss
    assert clean_loss < 0.01


def test_evm_clean_is_lower_than_impaired() -> None:
    """Clean symbols should have lower EVM than perturbed symbols."""
    set_seed(205)
    bits = generate_random_bits(batch_size=5, num_symbols=128, device="cpu")
    clean = modulate_16qam(bits).unsqueeze(0).expand(3, -1, -1).contiguous()
    impairment = 0.15 * torch.complex(torch.randn_like(clean.real), torch.randn_like(clean.real))
    impaired = clean + impairment

    assert evm_loss(clean, clean) < evm_loss(impaired, clean)


def test_power_loss_is_larger_for_large_residuals() -> None:
    """Power penalty should increase for larger residual tensors."""
    small = torch.full((3, 4, 16), 0.1, dtype=torch.complex64)
    large = torch.full((3, 4, 16), 1.0, dtype=torch.complex64)

    assert power_loss(large, max_power=0.05) > power_loss(small, max_power=0.05)


def test_extract_raw_features_shape_and_no_tx_id_leakage() -> None:
    """Raw features should have [D, B, F] shape and depend only on signal values."""
    set_seed(206)
    y_one = torch.complex(torch.randn(1, 4, 32), torch.randn(1, 4, 32))
    y = y_one.expand(3, -1, -1).contiguous()
    features = extract_raw_features(y, downsample=4, include_power=True)

    assert features.shape == (3, 4, 27)
    assert torch.equal(features[0], features[1])
    assert torch.equal(features[1], features[2])
