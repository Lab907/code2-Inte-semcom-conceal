"""Loss functions for multi-transmitter RF hiding training.

The distribution losses compare transmitter feature distributions pairwise
across devices. Communication losses keep the received signal close to the
intended 16-QAM symbols and bits.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import torch
import torch.nn.functional as F

from rfhide.metrics import evm_linear
from rfhide.modulation import get_16qam_constellation


DEFAULT_MMD_SIGMAS = [0.5, 1.0, 2.0, 4.0, 8.0]


def _tx_pairs(num_tx: int) -> list[tuple[int, int]]:
    """Return all transmitter index pairs."""
    return list(combinations(range(num_tx), 2))


def _as_sigma_tensor(sigmas: list[float] | tuple[float, ...] | torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert kernel bandwidths to a tensor."""
    sigma_tensor = torch.as_tensor(sigmas, device=device, dtype=dtype)
    if torch.any(sigma_tensor <= 0):
        raise ValueError("RBF MMD sigmas must be positive.")
    return sigma_tensor


def _loss_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Return the nested loss config when present."""
    if cfg is None:
        return {}
    return cfg.get("losses", cfg.get("loss", cfg))


def _prepare_bits_for_y(bits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Reshape or broadcast bit targets to ``y.shape + (4,)``."""
    if bits.ndim == y.ndim + 1 and tuple(bits.shape[:-1]) == tuple(y.shape):
        return bits.to(device=y.device, dtype=y.real.dtype)

    if y.ndim == 2:
        batch_size, num_symbols = y.shape
        if bits.shape == (batch_size, num_symbols * 4):
            return bits.reshape(batch_size, num_symbols, 4).to(device=y.device, dtype=y.real.dtype)
        if bits.shape == (batch_size, num_symbols, 4):
            return bits.to(device=y.device, dtype=y.real.dtype)

    if y.ndim == 3:
        num_tx, batch_size, num_symbols = y.shape
        if bits.shape == (batch_size, num_symbols * 4):
            return bits.reshape(batch_size, num_symbols, 4).unsqueeze(0).expand(num_tx, -1, -1, -1).to(
                device=y.device,
                dtype=y.real.dtype,
            )
        if bits.shape == (batch_size, num_symbols, 4):
            return bits.unsqueeze(0).expand(num_tx, -1, -1, -1).to(device=y.device, dtype=y.real.dtype)

    raise ValueError(f"Could not align bits shape {tuple(bits.shape)} with y shape {tuple(y.shape)}")


def mean_alignment_loss(features: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distance between per-Tx feature means."""
    if features.ndim != 3:
        raise ValueError(f"Expected features shape [D, B, F], got {tuple(features.shape)}")
    pairs = _tx_pairs(features.shape[0])
    if not pairs:
        return features.new_zeros(())

    means = features.mean(dim=1)
    losses = [(means[i] - means[j]).pow(2).mean() for i, j in pairs]
    return torch.stack(losses).mean()


def cov_alignment_loss(features: torch.Tensor) -> torch.Tensor:
    """Pairwise covariance alignment loss between transmitter features."""
    if features.ndim != 3:
        raise ValueError(f"Expected features shape [D, B, F], got {tuple(features.shape)}")
    pairs = _tx_pairs(features.shape[0])
    if not pairs:
        return features.new_zeros(())

    centered = features - features.mean(dim=1, keepdim=True)
    denom = max(features.shape[1] - 1, 1)
    covariances = centered.transpose(1, 2).matmul(centered) / denom
    losses = [(covariances[i] - covariances[j]).pow(2).mean() for i, j in pairs]
    return torch.stack(losses).mean()


def rbf_mmd_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: list[float] | tuple[float, ...] | torch.Tensor = DEFAULT_MMD_SIGMAS,
) -> torch.Tensor:
    """Compute biased multi-kernel RBF MMD between two sample sets."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"Expected x and y shapes [B, F], got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[1] != y.shape[1]:
        raise ValueError("MMD inputs must share the feature dimension.")

    sigmas_tensor = _as_sigma_tensor(sigmas, x.device, x.dtype)
    dist_xx = torch.cdist(x, x).pow(2)
    dist_yy = torch.cdist(y, y).pow(2)
    dist_xy = torch.cdist(x, y).pow(2)

    loss = x.new_zeros(())
    for sigma in sigmas_tensor:
        gamma = 1.0 / (2.0 * sigma.pow(2))
        k_xx = torch.exp(-gamma * dist_xx).mean()
        k_yy = torch.exp(-gamma * dist_yy).mean()
        k_xy = torch.exp(-gamma * dist_xy).mean()
        loss = loss + k_xx + k_yy - 2.0 * k_xy
    return (loss / sigmas_tensor.numel()).clamp_min(0.0)


def pairwise_mmd_loss(
    features: torch.Tensor,
    sigmas: list[float] | tuple[float, ...] | torch.Tensor = DEFAULT_MMD_SIGMAS,
) -> torch.Tensor:
    """Average pairwise MMD over all transmitter feature distributions."""
    if features.ndim != 3:
        raise ValueError(f"Expected features shape [D, B, F], got {tuple(features.shape)}")
    pairs = _tx_pairs(features.shape[0])
    if not pairs:
        return features.new_zeros(())
    losses = [rbf_mmd_loss(features[i], features[j], sigmas=sigmas) for i, j in pairs]
    return torch.stack(losses).mean()


def device_distribution_alignment(features: torch.Tensor, cfg: dict[str, Any] | None = None) -> torch.Tensor:
    """Weighted sum of mean, covariance, and MMD alignment losses."""
    loss_cfg = _loss_cfg(cfg)
    sigmas = loss_cfg.get("mmd_sigmas", DEFAULT_MMD_SIGMAS)
    mean_weight = float(loss_cfg.get("mean_weight", 1.0))
    cov_weight = float(loss_cfg.get("cov_weight", 1.0))
    mmd_weight = float(loss_cfg.get("mmd_weight", 1.0))
    return (
        mean_weight * mean_alignment_loss(features)
        + cov_weight * cov_alignment_loss(features)
        + mmd_weight * pairwise_mmd_loss(features, sigmas=sigmas)
    )


def evm_loss(y_eq_or_rx: torch.Tensor, x_clean_tx: torch.Tensor) -> torch.Tensor:
    """Mean linear EVM loss between received/equalized and clean Tx signals."""
    return evm_linear(y_eq_or_rx, x_clean_tx, align_gain=True).mean()


def soft_bit_loss(y_rx: torch.Tensor, bits: torch.Tensor, cfg: dict[str, Any] | None = None) -> torch.Tensor:
    """Differentiable 16-QAM soft-demapper BCE loss.

    Each received symbol is compared against all 16 constellation points. For
    every bit position, logsumexp over bit-1 points minus logsumexp over bit-0
    points produces a soft logit for BCEWithLogitsLoss.
    """
    if not torch.is_complex(y_rx):
        raise TypeError("soft_bit_loss expects complex received symbols.")

    loss_cfg = _loss_cfg(cfg)
    temperature = float(loss_cfg.get("soft_bit_temperature", loss_cfg.get("demapper_temperature", 0.05)))
    if temperature <= 0:
        raise ValueError("soft_bit_temperature must be positive.")

    points, labels = get_16qam_constellation(y_rx.device)
    points = points.to(dtype=y_rx.dtype)
    labels = labels.to(device=y_rx.device)
    distances = (y_rx.unsqueeze(-1) - points).abs().pow(2)
    scores = -distances / temperature

    logits = []
    for bit_idx in range(4):
        one_mask = labels[:, bit_idx] == 1
        zero_mask = ~one_mask
        log_one = torch.logsumexp(scores[..., one_mask], dim=-1)
        log_zero = torch.logsumexp(scores[..., zero_mask], dim=-1)
        logits.append(log_one - log_zero)
    bit_logits = torch.stack(logits, dim=-1)
    bit_targets = _prepare_bits_for_y(bits, y_rx)
    return F.binary_cross_entropy_with_logits(bit_logits, bit_targets)


def power_loss(p: torch.Tensor, max_power: float | torch.Tensor) -> torch.Tensor:
    """Penalty for signal or residual power above ``max_power``.

    ``p`` can be complex residuals/signals or real-valued power-like tensors.
    Larger residuals produce larger penalties, especially once average power
    exceeds the configured limit.
    """
    if torch.is_complex(p):
        sample_power = p.abs().pow(2).mean(dim=-1)
    else:
        sample_power = p.pow(2).mean(dim=-1) if p.ndim > 0 else p.pow(2)
    max_power_tensor = torch.as_tensor(max_power, device=p.device, dtype=sample_power.dtype).clamp_min(1e-12)
    excess = torch.relu(sample_power - max_power_tensor)
    return (excess.pow(2) / max_power_tensor.pow(2)).mean()


def total_teacher_loss(
    y_rx: torch.Tensor,
    x_clean_tx: torch.Tensor,
    bits: torch.Tensor,
    features: torch.Tensor | None = None,
    p: torch.Tensor | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Combine teacher-stage distribution and communication losses.

    Returns a dictionary containing ``total`` plus individual components so
    training scripts can log the pieces without recomputing them.
    """
    loss_cfg = _loss_cfg(cfg)
    features_for_alignment = features
    if features_for_alignment is None:
        from rfhide.features import extract_raw_features

        features_for_alignment = extract_raw_features(y_rx)

    align = device_distribution_alignment(features_for_alignment, cfg=cfg)
    evm = evm_loss(y_rx, x_clean_tx)
    soft_bits = soft_bit_loss(y_rx, bits, cfg=cfg)

    power = y_rx.new_zeros(())
    if p is not None:
        power = power_loss(p, loss_cfg.get("max_power", 1.0))

    total = (
        float(loss_cfg.get("alignment_weight", 1.0)) * align
        + float(loss_cfg.get("evm_weight", 1.0)) * evm
        + float(loss_cfg.get("soft_bit_weight", 1.0)) * soft_bits
        + float(loss_cfg.get("power_weight", 1.0)) * power
    )
    return {
        "total": total,
        "alignment": align,
        "evm": evm,
        "soft_bit": soft_bits,
        "power": power,
    }


def compute_loss(*args, **kwargs):
    """Backward-compatible alias for ``total_teacher_loss``."""
    return total_teacher_loss(*args, **kwargs)
