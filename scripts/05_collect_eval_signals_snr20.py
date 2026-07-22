"""Step 7 collect uncompensated, random, and fixed-precompensated eval signals."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.channel import add_awgn
from rfhide.config import load_config
from rfhide.dataset import MultiTxBatchGenerator
from rfhide.impairments import HardwareImpairmentBank
from rfhide.logging_utils import get_logger
from rfhide.metrics import ber_16qam, estimate_complex_gain, evm_linear
from rfhide.fixed_precomp import FixedPrecompensator, complex_to_channels, project_residual_power
from rfhide.semantic_jscc import semantic_enabled
from rfhide.utils import ensure_dir, get_device, save_json, set_seed

STEP_NAME = "Step 7 collect SNR20 eval signals"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Override fixed precomp checkpoint path.")
    parser.add_argument("--num-batches", type=int, default=None, help="Override number of eval batches.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eval batch size.")
    return parser.parse_args()


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _collection_config(config: dict[str, Any], batch_size: int) -> dict[str, Any]:
    """Copy config and set batch size for eval collection."""
    cfg = deepcopy(config)
    cfg.setdefault("data", {})
    cfg["data"]["batch_size"] = batch_size
    return cfg


def _load_fixed_precomp(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> FixedPrecompensator:
    """Load optimized fixed pre-compensation checkpoint."""
    model = FixedPrecompensator(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _equalize(y_rx: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Least-squares equalize received symbols to the clean reference."""
    gain = estimate_complex_gain(y_rx, x_ref)
    safe_gain = torch.where(gain.abs() < 1e-12, torch.ones_like(gain), gain)
    return y_rx / safe_gain


def _residual_ratio(p: torch.Tensor, x_clean_tx: torch.Tensor) -> torch.Tensor:
    """Compute residual-to-clean average power ratio for [D, B, N] tensors."""
    p_power = p.abs().pow(2).mean(dim=-1)
    x_power = x_clean_tx.abs().pow(2).mean(dim=-1).clamp_min(1e-12)
    return p_power / x_power


def _random_residual_like(p_ref: torch.Tensor, x_clean_tx: torch.Tensor, max_ratio: float) -> torch.Tensor:
    """Generate random complex residuals with per-sample power matching p_ref."""
    noise = torch.complex(torch.randn_like(p_ref.real), torch.randn_like(p_ref.real))
    noise_power = noise.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
    target_power = p_ref.abs().pow(2).mean(dim=-1, keepdim=True)
    p_rand = noise * torch.sqrt(target_power / noise_power)
    flat_rand = p_rand.reshape(-1, p_rand.shape[-1])
    flat_x = x_clean_tx.reshape(-1, x_clean_tx.shape[-1])
    return project_residual_power(flat_rand, flat_x, max_ratio).reshape_as(p_rand)


def _apply_chain(
    bank: HardwareImpairmentBank,
    x_pre: torch.Tensor,
    tx_ids: torch.Tensor,
    time_indices: torch.Tensor,
    snr_db: torch.Tensor,
) -> torch.Tensor:
    """Apply hardware impairment and AWGN."""
    y_imp = bank.apply(x_pre, tx_ids=tx_ids, time_indices=time_indices)
    return add_awgn(y_imp, snr_db=snr_db)


def _flatten_signal_entry(
    y_rx: torch.Tensor,
    x_clean_tx: torch.Tensor,
    bits: torch.Tensor,
    tx_ids: torch.Tensor,
    snr_db: torch.Tensor,
    p_residual: torch.Tensor,
    semantic_image: torch.Tensor | None = None,
    semantic_label: torch.Tensor | None = None,
    is_semantic: bool = False,
) -> dict[str, torch.Tensor]:
    """Flatten [D, B, ...] tensors into the saved eval format."""
    num_tx, batch_size, num_symbols = y_rx.shape
    x_flat = x_clean_tx.reshape(num_tx * batch_size, num_symbols)
    y_flat = y_rx.reshape(num_tx * batch_size, num_symbols)
    bits_flat = bits.unsqueeze(0).expand(num_tx, -1, -1).reshape(num_tx * batch_size, num_symbols * 4)
    labels = tx_ids.unsqueeze(1).expand(-1, batch_size).reshape(num_tx * batch_size)
    snr_flat = snr_db.unsqueeze(0).expand(num_tx, -1).reshape(num_tx * batch_size)
    p_ratio = _residual_ratio(p_residual, x_clean_tx).reshape(num_tx * batch_size)

    y_eq = _equalize(y_rx, x_clean_tx)
    evm = evm_linear(y_eq, x_clean_tx, align_gain=False).reshape(num_tx * batch_size)
    if is_semantic:
        ber = torch.zeros_like(evm)
    else:
        bit_groups = bits.reshape(batch_size, num_symbols, 4).unsqueeze(0).expand(num_tx, -1, -1, -1)
        ber = ber_16qam(y_eq, bit_groups).reshape(num_tx * batch_size)

    entry = {
        "signals": complex_to_channels(y_flat).detach().cpu(),
        "labels": labels.detach().cpu(),
        "bits": bits_flat.detach().cpu(),
        "x_clean": complex_to_channels(x_flat).detach().cpu(),
        "snr_db": snr_flat.detach().cpu(),
        "tx_id": labels.detach().cpu(),
        "evm": evm.detach().cpu(),
        "ber": ber.detach().cpu(),
        "residual_power_ratio": p_ratio.detach().cpu(),
    }
    if semantic_image is not None and semantic_label is not None:
        image_shape = tuple(semantic_image.shape[1:])
        entry["semantic_image"] = (
            semantic_image.unsqueeze(0)
            .expand(num_tx, -1, *image_shape)
            .reshape(num_tx * batch_size, *image_shape)
            .detach()
            .cpu()
        )
        entry["semantic_label"] = semantic_label.unsqueeze(0).expand(num_tx, -1).reshape(num_tx * batch_size).detach().cpu()
    return entry


def _append_columns(columns: dict[str, list[torch.Tensor]], values: dict[str, torch.Tensor]) -> None:
    """Append tensor columns from one batch."""
    for name, value in values.items():
        columns.setdefault(name, []).append(value)


def _cat_columns(columns: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate tensor columns."""
    return {name: torch.cat(parts, dim=0) for name, parts in columns.items()}


def _summarize(name: str, data: dict[str, torch.Tensor], path: Path) -> dict[str, Any]:
    """Summarize one saved eval signal class."""
    tx_counts = Counter(int(tx_id) for tx_id in data["tx_id"].tolist())
    return {
        "name": name,
        "num_samples": int(data["signals"].shape[0]),
        "tx_counts": {str(k): int(v) for k, v in sorted(tx_counts.items())},
        "mean_ber": float(data["ber"].mean().item()),
        "mean_evm": float(data["evm"].mean().item()),
        "mean_residual_power": float(data["residual_power_ratio"].mean().item()),
        "max_residual_power": float(data["residual_power_ratio"].max().item()),
        "path": str(path),
    }


def main() -> None:
    """Collect the three final evaluation signal classes."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    eval_cfg = config.get("eval_signal_collection", {})
    batch_size = int(args.batch_size if args.batch_size is not None else eval_cfg.get("batch_size", 32))
    num_batches = int(args.num_batches if args.num_batches is not None else eval_cfg.get("num_batches", 6))
    default_checkpoint = (
        eval_cfg.get("checkpoint")
        or config.get("fixed_precomp", {}).get("checkpoint")
        or config.get("compensation_dataset", {}).get("checkpoint")
        or "outputs/snr20/checkpoints/teacher_best.pt"
    )
    checkpoint_path = _resolve_project_path(args.checkpoint or default_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fixed precomp checkpoint not found: {checkpoint_path}")

    device = get_device(bool(config.get("device", {}).get("prefer_cuda", True)))
    logger = get_logger("rfhide.eval_signals")
    logger.info("Current step: %s", STEP_NAME)
    logger.info("Config path: %s", args.config)
    logger.info("Device: %s", device)
    logger.info("Batches: %d | batch size: %d | checkpoint: %s", num_batches, batch_size, checkpoint_path)

    run_cfg = _collection_config(config, batch_size=batch_size)
    generator = MultiTxBatchGenerator(run_cfg, split="eval", device=device)
    fixed_precomp = _load_fixed_precomp(run_cfg, checkpoint_path, device)
    max_ratio = float(
        run_cfg.get("fixed_precomp", {}).get(
            "max_residual_power_ratio",
            run_cfg.get("teacher", {}).get("max_residual_power_ratio", 0.05),
        )
    )
    is_semantic = semantic_enabled(run_cfg)

    columns = {
        "uncompensated": {},
        "random_perturb": {},
        "fixed_precomp": {},
    }

    for _ in tqdm(range(num_batches), desc="Collecting eval signal batches"):
        batch = generator.sample_batch()
        p_fixed = fixed_precomp(
            x_clean=batch["x_clean"],
            tx_ids=batch["tx_ids"],
            snr_db=batch["snr_db"],
            time_indices=batch["time_indices"],
        )
        p_zero = torch.zeros_like(p_fixed)
        p_rand = _random_residual_like(p_fixed, batch["x_clean_tx"], max_ratio)

        bank_uncomp = generator.impairment_bank
        y_uncomp = batch["y_rx"]
        y_rand = _apply_chain(bank_uncomp, batch["x_clean_tx"] + p_rand, batch["tx_ids"], batch["time_indices"], batch["snr_db"])
        y_fixed = _apply_chain(bank_uncomp, batch["x_clean_tx"] + p_fixed, batch["tx_ids"], batch["time_indices"], batch["snr_db"])

        _append_columns(
            columns["uncompensated"],
            _flatten_signal_entry(
                y_uncomp,
                batch["x_clean_tx"],
                batch["bits"],
                batch["tx_ids"],
                batch["snr_db"],
                p_zero,
                semantic_image=batch.get("semantic_image"),
                semantic_label=batch.get("semantic_label"),
                is_semantic=is_semantic,
            ),
        )
        _append_columns(
            columns["random_perturb"],
            _flatten_signal_entry(
                y_rand,
                batch["x_clean_tx"],
                batch["bits"],
                batch["tx_ids"],
                batch["snr_db"],
                p_rand,
                semantic_image=batch.get("semantic_image"),
                semantic_label=batch.get("semantic_label"),
                is_semantic=is_semantic,
            ),
        )
        _append_columns(
            columns["fixed_precomp"],
            _flatten_signal_entry(
                y_fixed,
                batch["x_clean_tx"],
                batch["bits"],
                batch["tx_ids"],
                batch["snr_db"],
                p_fixed,
                semantic_image=batch.get("semantic_image"),
                semantic_label=batch.get("semantic_label"),
                is_semantic=is_semantic,
            ),
        )

    output_dir = PROJECT_ROOT / config.get("experiment", {}).get("output_dir", "outputs/snr20")
    data_dir = ensure_dir(output_dir / "data")
    log_dir = ensure_dir(output_dir / "logs")
    paths = {
        "uncompensated": data_dir / "eval_uncompensated.pt",
        "random_perturb": data_dir / "eval_random_perturb.pt",
        "fixed_precomp": data_dir / "eval_fixed_precomp.pt",
    }

    summaries = {}
    saved_data = {}
    for name, path in paths.items():
        data = _cat_columns(columns[name])
        data["meta"] = {
            "class_name": name,
            "semantic_enabled": is_semantic,
            "format": "signals/x_clean are [S, 2, N] real-imag channels.",
        }
        torch.save(data, path)
        saved_data[name] = data
        summary = _summarize(name, data, path)
        summaries[name] = summary
        logger.info(
            "%s | samples %d | tx_counts %s | BER %.6f | EVM %.6f | residual power %.6f | saved %s",
            name,
            summary["num_samples"],
            summary["tx_counts"],
            summary["mean_ber"],
            summary["mean_evm"],
            summary["mean_residual_power"],
            path,
        )

    rand_power = summaries["random_perturb"]["mean_residual_power"]
    fixed_power = summaries["fixed_precomp"]["mean_residual_power"]
    relative_gap = abs(rand_power - fixed_power) / max(fixed_power, 1e-12)
    logger.info(
        "Random vs fixed precomp residual power match | random %.6f | fixed %.6f | relative gap %.4f",
        rand_power,
        fixed_power,
        relative_gap,
    )
    summary_path = log_dir / "eval_signal_collection_summary.json"
    save_json(
        {
            "checkpoint": str(checkpoint_path),
            "num_batches": num_batches,
            "batch_size": batch_size,
            "random_fixed_precomp_power_relative_gap": relative_gap,
            "splits": summaries,
        },
        summary_path,
    )
    logger.info("Saved eval signal collection summary: %s", summary_path)
    logger.info("Step 7 eval signal collection passed")


if __name__ == "__main__":
    main()
