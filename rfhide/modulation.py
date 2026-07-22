"""16-QAM modulation and hard-decision demodulation utilities.

Bit tensors use shape ``[..., 4]`` for one 16-QAM symbol per four bits. Symbol
tensors use the matching leading shape ``[...]`` and have a complex torch dtype.
The constellation uses Gray mapping on each I/Q axis and is normalized to unit
average power.
"""

from __future__ import annotations

import torch


_QAM16_NORMALIZER = 10.0**0.5


def _axis_bits_to_levels(two_bits: torch.Tensor) -> torch.Tensor:
    """Map Gray-coded bit pairs to unnormalized 16-QAM axis levels."""
    first = two_bits[..., 0].long()
    second = two_bits[..., 1].long()
    index = first * 2 + second
    levels = torch.tensor([-3.0, -1.0, 3.0, 1.0], device=two_bits.device)
    return levels[index]


def _axis_levels_to_bits(values: torch.Tensor) -> torch.Tensor:
    """Hard-map normalized axis values to Gray-coded bit pairs."""
    scaled = values * _QAM16_NORMALIZER
    nearest_level_index = torch.bucketize(
        scaled,
        torch.tensor([-2.0, 0.0, 2.0], device=values.device, dtype=scaled.real.dtype),
    )
    labels = torch.tensor(
        [[0, 0], [0, 1], [1, 1], [1, 0]],
        device=values.device,
        dtype=torch.long,
    )
    return labels[nearest_level_index]


def get_16qam_constellation(device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized 16-QAM constellation points and their bit labels.

    Args:
        device: Device for the returned tensors.

    Returns:
        A tuple ``(points, bit_labels)`` where ``points`` has shape ``[16]`` and
        complex dtype, while ``bit_labels`` has shape ``[16, 4]`` with integer
        Gray labels ordered as ``[I_bit0, I_bit1, Q_bit0, Q_bit1]``.
    """
    labels = torch.tensor(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 1, 0],
        ],
        device=device,
        dtype=torch.long,
    )
    i_values = _axis_bits_to_levels(labels[:, 0:2])
    q_values = _axis_bits_to_levels(labels[:, 2:4])
    points = torch.complex(i_values, q_values) / _QAM16_NORMALIZER
    return points, labels


def bits_to_16qam(bits: torch.Tensor) -> torch.Tensor:
    """Convert Gray-coded bit groups to normalized 16-QAM complex symbols.

    Args:
        bits: Tensor with shape ``[..., 4]`` containing binary values.

    Returns:
        Complex tensor with shape ``[...]``.
    """
    if bits.shape[-1] != 4:
        raise ValueError(f"Expected bits shape [..., 4], got {tuple(bits.shape)}")
    bit_groups = bits.to(dtype=torch.long)
    if not torch.all((bit_groups == 0) | (bit_groups == 1)):
        raise ValueError("16-QAM bits must contain only 0 or 1 values.")

    i_values = _axis_bits_to_levels(bit_groups[..., 0:2])
    q_values = _axis_bits_to_levels(bit_groups[..., 2:4])
    return torch.complex(i_values, q_values) / _QAM16_NORMALIZER


def qam16_to_bits(symbols: torch.Tensor) -> torch.Tensor:
    """Hard-demodulate normalized 16-QAM symbols to Gray-coded bit groups.

    Args:
        symbols: Complex tensor with shape ``[...]``.

    Returns:
        Integer tensor with shape ``[..., 4]``.
    """
    if not torch.is_complex(symbols):
        raise TypeError("16-QAM demodulation expects a complex tensor.")
    i_bits = _axis_levels_to_bits(symbols.real)
    q_bits = _axis_levels_to_bits(symbols.imag)
    return torch.cat([i_bits, q_bits], dim=-1)


def generate_random_bits(
    batch_size: int,
    num_symbols: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate random 16-QAM bit groups with shape ``[batch, symbols, 4]``."""
    return torch.randint(0, 2, (batch_size, num_symbols, 4), device=device, dtype=torch.long)


def modulate_16qam(bits: torch.Tensor) -> torch.Tensor:
    """Alias for ``bits_to_16qam`` used by signal-chain scripts."""
    return bits_to_16qam(bits)


def demodulate_16qam_hard(symbols: torch.Tensor) -> torch.Tensor:
    """Alias for hard-decision 16-QAM demodulation."""
    return qam16_to_bits(symbols)
