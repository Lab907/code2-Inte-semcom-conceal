"""Hardware impairment models for RF signal-chain simulation.

The impairment bank assigns each transmitter its own fixed RF offsets. Optional
drift coefficients are still accepted for legacy experiments, but their default
value is zero so the active signal chain is time-invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

import torch


@dataclass
class DriftState:
    """Container for sampled transmitter drift parameters."""

    gain_db: torch.Tensor
    amplitude: torch.Tensor
    phase_rad: torch.Tensor
    cfo_hz: torch.Tensor
    iq_gain_mismatch_db: torch.Tensor
    iq_phase_mismatch_rad: torch.Tensor
    dc_offset: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return the drift state as a tensor dictionary."""
        return {
            "gain_db": self.gain_db,
            "amplitude": self.amplitude,
            "phase_rad": self.phase_rad,
            "cfo_hz": self.cfo_hz,
            "iq_gain_mismatch_db": self.iq_gain_mismatch_db,
            "iq_phase_mismatch_rad": self.iq_phase_mismatch_rad,
            "dc_offset": self.dc_offset,
        }


class HardwareImpairmentBank:
    """Bank of transmitter-specific fixed RF impairments.

    Args:
        cfg: Impairment configuration dictionary.
        device: Torch device used for parameters and outputs.
    """

    def __init__(self, cfg: dict[str, Any] | None, device: torch.device | str) -> None:
        self.cfg = cfg or {}
        self.device = torch.device(device)
        self.num_tx = int(self.cfg.get("num_tx", 6))
        self.sample_rate = float(self.cfg.get("sample_rate", 1_000_000.0))

        gain_std = float(self.cfg.get("gain_db_std", 0.8))
        phase_std = float(self.cfg.get("phase_rad_std", 0.25))
        cfo_std = float(self.cfg.get("cfo_hz_std", 80.0))
        iq_gain_std = float(self.cfg.get("iq_gain_mismatch_db_std", 0.35))
        iq_phase_std = float(self.cfg.get("iq_phase_mismatch_rad_std", 0.04))
        dc_std = float(self.cfg.get("dc_offset_std", 0.015))

        self.gain_drift_db_per_step = float(self.cfg.get("gain_drift_db_per_step", 0.0))
        self.phase_drift_rad_per_step = float(self.cfg.get("phase_drift_rad_per_step", 0.0))
        self.cfo_drift_hz_per_step = float(self.cfg.get("cfo_drift_hz_per_step", 0.0))

        self.base_gain_db = torch.randn(self.num_tx, device=self.device) * gain_std
        self.base_phase_rad = torch.randn(self.num_tx, device=self.device) * phase_std
        self.base_cfo_hz = torch.randn(self.num_tx, device=self.device) * cfo_std
        self.base_iq_gain_mismatch_db = torch.randn(self.num_tx, device=self.device) * iq_gain_std
        self.base_iq_phase_mismatch_rad = torch.randn(self.num_tx, device=self.device) * iq_phase_std
        self.base_dc_offset = torch.complex(
            torch.randn(self.num_tx, device=self.device) * dc_std,
            torch.randn(self.num_tx, device=self.device) * dc_std,
        )
        self.last_params: dict[str, torch.Tensor] | None = None

    def _broadcast_value(
        self,
        value: torch.Tensor | int | float | list[int] | list[float] | None,
        leading_shape: torch.Size,
        dtype: torch.dtype,
        default: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a scalar/vector parameter to the input leading shape."""
        if value is None:
            tensor = default.to(device=self.device, dtype=dtype)
        else:
            tensor = torch.as_tensor(value, device=self.device, dtype=dtype)

        if tensor.ndim == 0:
            return tensor.expand(leading_shape)
        if tuple(tensor.shape) == tuple(leading_shape):
            return tensor
        if len(leading_shape) == 2 and tensor.ndim == 1 and tensor.shape[0] == leading_shape[0]:
            return tensor[:, None].expand(leading_shape)
        if len(leading_shape) == 1 and tensor.ndim == 1 and tensor.shape[0] == leading_shape[0]:
            return tensor
        if tensor.numel() == int(torch.tensor(leading_shape).prod().item()):
            return tensor.reshape(leading_shape)
        return torch.broadcast_to(tensor, leading_shape)

    def _prepare_tx_ids(
        self,
        x: torch.Tensor,
        tx_ids: torch.Tensor | int | list[int] | None,
    ) -> torch.Tensor:
        """Prepare transmitter ids for the leading dimensions of ``x``."""
        leading_shape = x.shape[:-1]
        if x.ndim == 3:
            default = torch.arange(x.shape[0], device=self.device, dtype=torch.long)
        else:
            default = torch.zeros((), device=self.device, dtype=torch.long)
        ids = self._broadcast_value(tx_ids, leading_shape, torch.long, default)
        if torch.any((ids < 0) | (ids >= self.num_tx)):
            raise ValueError(f"tx_ids must be in [0, {self.num_tx - 1}]")
        return ids

    def _prepare_time_indices(
        self,
        x: torch.Tensor,
        time_indices: torch.Tensor | int | float | list[int] | list[float] | None,
    ) -> torch.Tensor:
        """Prepare time indices for the leading dimensions of ``x``."""
        leading_shape = x.shape[:-1]
        default = torch.zeros((), device=self.device)
        return self._broadcast_value(time_indices, leading_shape, torch.float32, default)

    def sample_params(
        self,
        tx_ids: torch.Tensor | int | list[int],
        time_indices: torch.Tensor | int | float | list[int] | list[float],
    ) -> dict[str, torch.Tensor]:
        """Return actual impairment parameters for transmitter/time pairs."""
        ids, times = torch.broadcast_tensors(
            torch.as_tensor(tx_ids, device=self.device, dtype=torch.long),
            torch.as_tensor(time_indices, device=self.device, dtype=torch.float32),
        )
        if torch.any((ids < 0) | (ids >= self.num_tx)):
            raise ValueError(f"tx_ids must be in [0, {self.num_tx - 1}]")

        gain_db = self.base_gain_db[ids] + self.gain_drift_db_per_step * times
        phase_rad = self.base_phase_rad[ids] + self.phase_drift_rad_per_step * times
        cfo_hz = self.base_cfo_hz[ids] + self.cfo_drift_hz_per_step * times
        amplitude = torch.pow(torch.tensor(10.0, device=self.device), gain_db / 20.0)

        state = DriftState(
            gain_db=gain_db,
            amplitude=amplitude,
            phase_rad=phase_rad,
            cfo_hz=cfo_hz,
            iq_gain_mismatch_db=self.base_iq_gain_mismatch_db[ids],
            iq_phase_mismatch_rad=self.base_iq_phase_mismatch_rad[ids],
            dc_offset=self.base_dc_offset[ids],
        )
        return state.as_dict()

    def apply(
        self,
        x: torch.Tensor,
        tx_ids: torch.Tensor | int | list[int] | None = None,
        time_indices: torch.Tensor | int | float | list[int] | list[float] | None = None,
    ) -> torch.Tensor:
        """Apply Tx-specific impairments to a complex signal tensor.

        Args:
            x: Complex tensor with shape ``[batch, symbols]`` or
                ``[num_tx, batch, symbols]``.
            tx_ids: Transmitter ids. Scalars, vectors, or full leading-shape
                tensors are accepted.
            time_indices: Time indices used to evaluate deterministic drift.

        Returns:
            Complex tensor with the same shape as ``x``.
        """
        if not torch.is_complex(x):
            raise TypeError("Hardware impairments expect a complex input tensor.")
        if x.ndim not in (2, 3):
            raise ValueError(f"Expected x shape [batch, symbols] or [num_tx, batch, symbols], got {tuple(x.shape)}")

        x = x.to(self.device)
        ids = self._prepare_tx_ids(x, tx_ids)
        times = self._prepare_time_indices(x, time_indices)
        params = self.sample_params(ids, times)
        self.last_params = params

        num_symbols = x.shape[-1]
        symbol_index = torch.arange(num_symbols, device=self.device, dtype=x.real.dtype)
        view_shape = (1,) * (x.ndim - 1) + (num_symbols,)
        symbol_index = symbol_index.reshape(view_shape)

        amplitude = params["amplitude"].unsqueeze(-1).to(dtype=x.real.dtype)
        phase = params["phase_rad"].unsqueeze(-1).to(dtype=x.real.dtype)
        cfo_phase = (2.0 * pi * params["cfo_hz"].unsqueeze(-1).to(dtype=x.real.dtype) / self.sample_rate) * symbol_index
        rotation = torch.exp(1j * (phase + cfo_phase))

        y = amplitude * x * rotation

        iq_gain = torch.pow(
            torch.tensor(10.0, device=self.device, dtype=x.real.dtype),
            params["iq_gain_mismatch_db"].unsqueeze(-1).to(dtype=x.real.dtype) / 20.0,
        )
        iq_phase = params["iq_phase_mismatch_rad"].unsqueeze(-1).to(dtype=x.real.dtype)
        image_coeff = 0.5 * (iq_gain - 1.0) * torch.exp(1j * iq_phase)
        y = y + image_coeff * y.conj()
        y = y + params["dc_offset"].unsqueeze(-1).to(dtype=y.dtype)
        return y
