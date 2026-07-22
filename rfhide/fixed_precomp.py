"""Fixed transmitter-side RF pre-compensation parameters."""

from __future__ import annotations

from math import pi
from typing import Any

import torch
from torch import nn


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, N]`` complex tensors to ``[B, 2, N]`` real channels."""
    if not torch.is_complex(x):
        raise TypeError("Expected a complex tensor.")
    return torch.stack([x.real, x.imag], dim=1)


def project_residual_power(
    p: torch.Tensor,
    x_clean: torch.Tensor,
    max_residual_power_ratio: float,
) -> torch.Tensor:
    """Project residual power under a per-sample ratio limit."""
    if not torch.is_complex(p) or not torch.is_complex(x_clean):
        raise TypeError("Residual projection expects complex tensors.")
    x_power = x_clean.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    p_power = p.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    max_power = float(max_residual_power_ratio) * x_power * (1.0 - 1e-6)
    scale = torch.sqrt(max_power / p_power).clamp(max=1.0)
    return p * scale


class FixedPrecompensator(nn.Module):
    """Learn a fixed set of Tx-side pre-compensation parameters.

    The optimized parameters are transmitter-specific and do not depend on
    time, payload identity, or a diffusion sampler. The resulting residual is
    returned in the same ``[D, B, N]`` format used by the existing signal chain.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        fixed_cfg = cfg.get("fixed_precomp", {})
        teacher_cfg = cfg.get("teacher", {})
        impair_cfg = cfg.get("impairments", {})
        signal_cfg = cfg.get("signal", {})

        self.num_tx = int(impair_cfg.get("num_tx", fixed_cfg.get("num_tx", 6)))
        self.sample_rate = float(impair_cfg.get("sample_rate", signal_cfg.get("sample_rate", 1_000_000.0)))
        self.max_residual_power_ratio = float(
            fixed_cfg.get("max_residual_power_ratio", teacher_cfg.get("max_residual_power_ratio", 0.05))
        )

        self.log_amplitude = nn.Parameter(torch.zeros(self.num_tx))
        self.phase_rad = nn.Parameter(torch.zeros(self.num_tx))
        self.cfo_hz = nn.Parameter(torch.zeros(self.num_tx))
        self.image_real = nn.Parameter(torch.zeros(self.num_tx))
        self.image_imag = nn.Parameter(torch.zeros(self.num_tx))
        self.dc_real = nn.Parameter(torch.zeros(self.num_tx))
        self.dc_imag = nn.Parameter(torch.zeros(self.num_tx))

    def _check_inputs(self, x_clean: torch.Tensor, tx_ids: torch.Tensor) -> torch.Tensor:
        """Validate and normalize Tx IDs."""
        if not torch.is_complex(x_clean):
            raise TypeError("FixedPrecompensator expects complex x_clean.")
        if x_clean.ndim != 2:
            raise ValueError(f"Expected x_clean shape [B, N], got {tuple(x_clean.shape)}")
        ids = torch.as_tensor(tx_ids, device=x_clean.device, dtype=torch.long)
        if ids.ndim != 1:
            raise ValueError(f"Expected tx_ids shape [D], got {tuple(ids.shape)}")
        if torch.any((ids < 0) | (ids >= self.num_tx)):
            raise ValueError(f"tx_ids must be in [0, {self.num_tx - 1}]")
        return ids

    def precompensate(self, x_clean: torch.Tensor, tx_ids: torch.Tensor) -> torch.Tensor:
        """Apply the fixed Tx-side pre-compensation parameters."""
        ids = self._check_inputs(x_clean, tx_ids)
        batch_size, num_symbols = x_clean.shape
        x_tx = x_clean.unsqueeze(0).expand(ids.numel(), -1, -1)

        dtype = x_clean.real.dtype
        symbol_index = torch.arange(num_symbols, device=x_clean.device, dtype=dtype).reshape(1, 1, num_symbols)
        amplitude = torch.exp(self.log_amplitude[ids]).reshape(-1, 1, 1).to(dtype=dtype)
        phase = self.phase_rad[ids].reshape(-1, 1, 1).to(dtype=dtype)
        cfo = self.cfo_hz[ids].reshape(-1, 1, 1).to(dtype=dtype)
        rotation = torch.exp(1j * (phase + (2.0 * pi * cfo / self.sample_rate) * symbol_index))

        y = amplitude * x_tx * rotation
        image = torch.complex(self.image_real[ids], self.image_imag[ids]).reshape(-1, 1, 1).to(dtype=y.dtype)
        dc = torch.complex(self.dc_real[ids], self.dc_imag[ids]).reshape(-1, 1, 1).to(dtype=y.dtype)
        y = y + image * y.conj() + dc
        return y.reshape(ids.numel(), batch_size, num_symbols)

    def residual_power_ratio(self, p: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
        """Return per-Tx/sample residual-to-clean power ratios."""
        x_power = x_clean.abs().pow(2).mean(dim=-1).unsqueeze(0).clamp_min(1e-12)
        p_power = p.abs().pow(2).mean(dim=-1)
        return p_power / x_power

    def forward(
        self,
        x_clean: torch.Tensor,
        tx_ids: torch.Tensor,
        snr_db: float | torch.Tensor | None = None,
        time_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the projected residual induced by fixed pre-compensation."""
        del snr_db, time_indices
        x_pre = self.precompensate(x_clean, tx_ids)
        x_tx = x_clean.unsqueeze(0).expand_as(x_pre)
        p = x_pre - x_tx
        flat_p = p.reshape(-1, p.shape[-1])
        flat_x = x_tx.reshape(-1, x_tx.shape[-1])
        return project_residual_power(flat_p, flat_x, self.max_residual_power_ratio).reshape_as(p)


FixedCompensator = FixedPrecompensator
