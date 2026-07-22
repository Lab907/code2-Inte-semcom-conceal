"""Channel models for RF signal-chain simulation."""

from __future__ import annotations

import torch


def add_awgn(x: torch.Tensor, snr_db: float | torch.Tensor) -> torch.Tensor:
    """Add complex AWGN according to the measured signal power.

    Args:
        x: Complex signal tensor with shape ``[..., num_symbols]``.
        snr_db: SNR in dB. Can be a scalar or a tensor broadcastable to the
            leading dimensions of ``x``.

    Returns:
        Noisy complex tensor with the same shape as ``x``.
    """
    if not torch.is_complex(x):
        raise TypeError("AWGN channel expects a complex input tensor.")

    signal_power = x.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    snr_db_tensor = torch.as_tensor(snr_db, device=x.device, dtype=x.real.dtype)
    if snr_db_tensor.ndim == 1 and x.ndim >= 2 and snr_db_tensor.shape[0] == x.shape[-2]:
        view_shape = (1,) * (x.ndim - 2) + (x.shape[-2], 1)
        snr_db_tensor = snr_db_tensor.reshape(view_shape)
    else:
        while snr_db_tensor.ndim < signal_power.ndim:
            snr_db_tensor = snr_db_tensor.unsqueeze(-1)

    snr_linear = torch.pow(torch.tensor(10.0, device=x.device, dtype=x.real.dtype), snr_db_tensor / 10.0)
    noise_power = signal_power / snr_linear
    real_noise = torch.randn_like(x.real) * torch.sqrt(noise_power / 2.0)
    imag_noise = torch.randn_like(x.real) * torch.sqrt(noise_power / 2.0)
    return x + torch.complex(real_noise, imag_noise)
