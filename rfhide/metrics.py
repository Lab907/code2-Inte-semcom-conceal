"""Metrics for 16-QAM signal-chain evaluation."""

from __future__ import annotations

import torch

from rfhide.modulation import demodulate_16qam_hard


def estimate_complex_gain(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Estimate least-squares complex gain ``g`` such that ``y ~= g * x``.

    Args:
        y: Observed complex tensor with shape ``[..., num_symbols]``.
        x: Reference complex tensor with the same shape.

    Returns:
        Complex gain tensor with shape ``[..., 1]``.
    """
    if y.shape != x.shape:
        raise ValueError(f"Expected y and x to share shape, got {tuple(y.shape)} and {tuple(x.shape)}")
    numerator = (y * x.conj()).sum(dim=-1, keepdim=True)
    denominator = x.abs().pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return numerator / denominator


def evm_linear(y: torch.Tensor, x: torch.Tensor, align_gain: bool = True) -> torch.Tensor:
    """Compute RMS EVM as a linear ratio over the last dimension."""
    reference = x
    observed = y
    if align_gain:
        gain = estimate_complex_gain(y, x)
        safe_gain = torch.where(gain.abs() < 1e-12, torch.ones_like(gain), gain)
        observed = y / safe_gain

    error_power = (observed - reference).abs().pow(2).mean(dim=-1)
    reference_power = reference.abs().pow(2).mean(dim=-1).clamp_min(1e-12)
    return torch.sqrt(error_power / reference_power)


def evm_db(evm: torch.Tensor) -> torch.Tensor:
    """Convert linear EVM values to dB."""
    return 20.0 * torch.log10(evm.clamp_min(1e-12))


def evm_db_or_linear(y: torch.Tensor, x: torch.Tensor, return_db: bool = False) -> torch.Tensor:
    """Compute EVM and optionally return it in dB.

    The default return value is linear EVM, matching the Step 1 requirement.
    """
    evm = evm_linear(y, x)
    if return_db:
        return evm_db(evm)
    return evm


def ber_16qam(y: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """Compute hard-decision 16-QAM bit error rate over the last two bit axes."""
    detected_bits = demodulate_16qam_hard(y)
    if detected_bits.shape != bits.shape:
        raise ValueError(
            f"Expected demodulated bits shape {tuple(bits.shape)}, got {tuple(detected_bits.shape)}"
        )
    errors = (detected_bits != bits.to(device=detected_bits.device, dtype=detected_bits.dtype)).float()
    return errors.flatten(start_dim=-2).mean(dim=-1)
