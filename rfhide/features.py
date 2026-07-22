"""Distribution feature extraction for multi-transmitter alignment losses.

The raw feature extractor uses only signal samples and power statistics. It
does not include transmitter ids or any explicit device labels.
"""

from __future__ import annotations

import torch


def extract_raw_features(y: torch.Tensor, downsample: int = 4, include_power: bool = True) -> torch.Tensor:
    """Extract raw per-sample distribution features from complex Rx signals.

    Args:
        y: Complex tensor with shape ``[D, B, N]``.
        downsample: Positive stride used to reduce the symbol axis.
        include_power: Whether to include downsampled power and whole-signal
            power summary statistics. When ``False``, only real and imaginary
            downsampled samples are returned.

    Returns:
        Real-valued feature tensor with shape ``[D, B, F]``.
    """
    if not torch.is_complex(y):
        raise TypeError("extract_raw_features expects a complex tensor.")
    if y.ndim != 3:
        raise ValueError(f"Expected y shape [D, B, N], got {tuple(y.shape)}")
    if downsample <= 0:
        raise ValueError("downsample must be a positive integer.")

    y_ds = y[..., ::downsample]
    parts = [y_ds.real, y_ds.imag]

    if include_power:
        power = y.abs().pow(2)
        power_ds = power[..., ::downsample]
        mean_power = power.mean(dim=-1, keepdim=True)
        std_power = power.std(dim=-1, keepdim=True, unbiased=False)
        peak_power = power.amax(dim=-1, keepdim=True)
        parts.extend([power_ds, mean_power, std_power, peak_power])

    return torch.cat([part.to(dtype=y.real.dtype) for part in parts], dim=-1)


def extract_features(y: torch.Tensor, downsample: int = 4, include_power: bool = True) -> torch.Tensor:
    """Backward-compatible alias for raw feature extraction."""
    return extract_raw_features(y, downsample=downsample, include_power=include_power)
