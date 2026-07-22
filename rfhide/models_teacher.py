"""Deterministic teacher compensation network for offline RF pre-compensation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ResidualConvBlock(nn.Module):
    """Small residual 1D convolution block used by the teacher network."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual convolution block."""
        return self.activation(x + self.net(x))


class TeacherCompensator(nn.Module):
    """Shared Conv1D teacher that predicts complex residuals for each Tx.

    Args:
        cfg: Full experiment config or a teacher-only config dictionary.

    Forward inputs:
        ``x_clean``: ``[B, N]`` complex clean signal.
        ``tx_ids``: ``[D]`` transmitter ids.
        ``snr_db``: scalar or ``[B]`` SNR values.
        ``time_indices``: ``[D, B]`` or ``[B]`` drift time indices.

    Returns:
        Complex residual ``p`` with shape ``[D, B, N]``. The residual is
        projected so that each Tx/sample obeys
        ``mean(|p|^2) / mean(|x_clean|^2) <= max_residual_power_ratio``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        teacher_cfg = cfg.get("teacher", cfg)
        impair_cfg = cfg.get("impairments", {})

        self.num_tx = int(impair_cfg.get("num_tx", teacher_cfg.get("num_tx", 6)))
        self.max_residual_power_ratio = float(teacher_cfg.get("max_residual_power_ratio", 0.05))
        hidden_channels = int(teacher_cfg.get("hidden_channels", 48))
        condition_dim = int(teacher_cfg.get("condition_dim", 16))
        num_blocks = int(teacher_cfg.get("num_blocks", 3))

        self.tx_embedding = nn.Embedding(self.num_tx, condition_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(2, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.input = nn.Sequential(
            nn.Conv1d(2 + condition_dim, hidden_channels, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[ResidualConvBlock(hidden_channels) for _ in range(num_blocks)])
        self.output = nn.Conv1d(hidden_channels, 2, kernel_size=3, padding=1)
        nn.init.normal_(self.output.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.output.bias)

    def _prepare_snr(self, snr_db: float | torch.Tensor, num_tx: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Broadcast SNR values to ``[D, B]``."""
        snr = torch.as_tensor(snr_db, device=device, dtype=torch.float32)
        if snr.ndim == 0:
            return snr.expand(num_tx, batch_size)
        if snr.shape == (batch_size,):
            return snr.unsqueeze(0).expand(num_tx, -1)
        if snr.shape == (num_tx, batch_size):
            return snr
        raise ValueError(f"Cannot broadcast snr_db shape {tuple(snr.shape)} to [{num_tx}, {batch_size}]")

    def _prepare_time(
        self,
        time_indices: torch.Tensor,
        num_tx: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Broadcast time indices to ``[D, B]``."""
        times = torch.as_tensor(time_indices, device=device, dtype=torch.float32)
        if times.ndim == 0:
            return times.expand(num_tx, batch_size)
        if times.shape == (batch_size,):
            return times.unsqueeze(0).expand(num_tx, -1)
        if times.shape == (num_tx, batch_size):
            return times
        raise ValueError(f"Cannot broadcast time_indices shape {tuple(times.shape)} to [{num_tx}, {batch_size}]")

    def _project_residual(self, p: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
        """Project residual power to the configured ratio bound."""
        x_power = x_clean.abs().pow(2).mean(dim=-1, keepdim=True).unsqueeze(0).clamp_min(1e-12)
        p_power = p.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
        max_power = self.max_residual_power_ratio * x_power * (1.0 - 1e-6)
        scale = torch.sqrt(max_power / p_power).clamp(max=1.0)
        return p * scale

    def residual_power_ratio(self, p: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
        """Return per-Tx/sample residual-to-clean power ratios with shape ``[D, B]``."""
        x_power = x_clean.abs().pow(2).mean(dim=-1).unsqueeze(0).clamp_min(1e-12)
        p_power = p.abs().pow(2).mean(dim=-1)
        return p_power / x_power

    def forward(
        self,
        x_clean: torch.Tensor,
        tx_ids: torch.Tensor,
        snr_db: float | torch.Tensor,
        time_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict projected complex residuals for all requested transmitters."""
        if not torch.is_complex(x_clean):
            raise TypeError("TeacherCompensator expects complex x_clean.")
        if x_clean.ndim != 2:
            raise ValueError(f"Expected x_clean shape [B, N], got {tuple(x_clean.shape)}")

        device = x_clean.device
        tx_ids = torch.as_tensor(tx_ids, device=device, dtype=torch.long)
        if tx_ids.ndim != 1:
            raise ValueError(f"Expected tx_ids shape [D], got {tuple(tx_ids.shape)}")
        if torch.any((tx_ids < 0) | (tx_ids >= self.num_tx)):
            raise ValueError(f"tx_ids must be in [0, {self.num_tx - 1}]")

        num_tx = tx_ids.numel()
        batch_size, num_symbols = x_clean.shape
        snr = self._prepare_snr(snr_db, num_tx, batch_size, device)
        times = self._prepare_time(time_indices, num_tx, batch_size, device)

        x_channels = torch.stack([x_clean.real, x_clean.imag], dim=0)
        x_channels = x_channels.unsqueeze(0).expand(num_tx, -1, -1, -1)
        x_channels = x_channels.permute(0, 2, 1, 3).reshape(num_tx * batch_size, 2, num_symbols)

        tx_cond = self.tx_embedding(tx_ids).unsqueeze(1).expand(-1, batch_size, -1)
        scalar_cond = torch.stack([snr / 30.0, torch.log1p(times.clamp_min(0.0)) / 10.0], dim=-1)
        cond = tx_cond + self.condition_mlp(scalar_cond)
        cond_channels = cond.reshape(num_tx * batch_size, -1).unsqueeze(-1).expand(-1, -1, num_symbols)

        net_input = torch.cat([x_channels, cond_channels], dim=1)
        hidden = self.blocks(self.input(net_input))
        raw = torch.tanh(self.output(hidden))
        raw = raw.reshape(num_tx, batch_size, 2, num_symbols)
        p = torch.complex(raw[:, :, 0], raw[:, :, 1])
        return self._project_residual(p, x_clean)


TeacherModel = TeacherCompensator
