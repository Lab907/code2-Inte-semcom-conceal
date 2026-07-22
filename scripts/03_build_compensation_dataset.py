"""Step 5 build an offline fixed pre-compensation dataset."""

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
from rfhide.features import extract_raw_features
from rfhide.logging_utils import get_logger
from rfhide.losses import mean_alignment_loss
from rfhide.metrics import ber_16qam, estimate_complex_gain, evm_linear
from rfhide.fixed_precomp import FixedPrecompensator
from rfhide.semantic_jscc import semantic_enabled
from rfhide.utils import ensure_dir, get_device, save_json, set_seed

STEP_NAME = "Step 5 build fixed precomp dataset"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", default=None, help="Override teacher checkpoint path.")
    parser.add_argument("--train-batches", type=int, default=None, help="Override number of train batches.")
    parser.add_argument("--val-batches", type=int, default=None, help="Override number of val batches.")
    parser.add_argument("--test-batches", type=int, default=None, help="Override number of test batches.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override compensation dataset batch size.")
    return parser.parse_args()


def _dataset_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Return compensation dataset config with defaults."""
    return config.get("compensation_dataset", {})


def _build_generator_config(config: dict[str, Any], batch_size: int) -> dict[str, Any]:
    """Copy config and set generator batch size."""
    generator_cfg = deepcopy(config)
    generator_cfg.setdefault("data", {})
    generator_cfg["data"]["batch_size"] = batch_size
    return generator_cfg


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve paths relative to the project root when needed."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _load_fixed_precomp(config: dict[str, Any], checkpoint_path: Path, device: torch.device) -> FixedPrecompensator:
    """Load an optimized fixed pre-compensation checkpoint."""
    model = FixedPrecompensator(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _equalize(y_rx: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Least-squares equalize received symbols against clean references."""
    gain = estimate_complex_gain(y_rx, x_ref)
    safe_gain = torch.where(gain.abs() < 1e-12, torch.ones_like(gain), gain)
    return y_rx / safe_gain


def _residual_power_ratio(p: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
    """Return residual-to-clean power ratio with shape [D, B]."""
    x_power = x_clean.abs().pow(2).mean(dim=-1).unsqueeze(0).clamp_min(1e-12)
    p_power = p.abs().pow(2).mean(dim=-1)
    return p_power / x_power


def _flatten_batch(batch: dict[str, torch.Tensor], p_teacher: torch.Tensor, residual_ratio: torch.Tensor) -> dict[str, torch.Tensor]:
    """Flatten a [D, B, ...] teacher batch into sample-major tensors."""
    num_tx, batch_size, num_symbols = p_teacher.shape
    x_clean = batch["x_clean"].unsqueeze(0).expand(num_tx, -1, -1).reshape(num_tx * batch_size, num_symbols)
    bits = batch["bits"].unsqueeze(0).expand(num_tx, -1, -1).reshape(num_tx * batch_size, num_symbols * 4)
    snr_db = batch["snr_db"].unsqueeze(0).expand(num_tx, -1).reshape(num_tx * batch_size)
    tx_ids = batch["tx_ids"].unsqueeze(1).expand(-1, batch_size).reshape(num_tx * batch_size)
    time_index = batch["time_indices"].reshape(num_tx * batch_size)
    columns = {
        "x_clean": x_clean.detach().cpu(),
        "bits": bits.detach().cpu(),
        "tx_id": tx_ids.detach().cpu(),
        "snr_db": snr_db.detach().cpu(),
        "time_index": time_index.detach().cpu(),
        "p_teacher": p_teacher.reshape(num_tx * batch_size, num_symbols).detach().cpu(),
        "residual_power_ratio": residual_ratio.reshape(num_tx * batch_size).detach().cpu(),
    }
    if "semantic_image" in batch:
        image_shape = tuple(batch["semantic_image"].shape[1:])
        columns["semantic_image"] = (
            batch["semantic_image"]
            .unsqueeze(0)
            .expand(num_tx, -1, *image_shape)
            .reshape(num_tx * batch_size, *image_shape)
            .detach()
            .cpu()
        )
        columns["semantic_label"] = (
            batch["semantic_label"].unsqueeze(0).expand(num_tx, -1).reshape(num_tx * batch_size).detach().cpu()
        )
    return columns


def _append_columns(columns: dict[str, list[torch.Tensor]], batch_columns: dict[str, torch.Tensor]) -> None:
    """Append one flattened batch to column buffers."""
    for name, value in batch_columns.items():
        columns.setdefault(name, []).append(value)


def _cat_columns(columns: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate column buffers into one saved tensor dictionary."""
    return {name: torch.cat(parts, dim=0) for name, parts in columns.items()}


@torch.no_grad()
def _build_split(
    split: str,
    num_batches: int,
    config: dict[str, Any],
    fixed_precomp: FixedPrecompensator,
    device: torch.device,
    output_path: Path,
) -> dict[str, Any]:
    """Generate and save one split."""
    generator = MultiTxBatchGenerator(config, split=split, device=device)
    is_semantic = semantic_enabled(config)
    columns: dict[str, list[torch.Tensor]] = {}
    tx_counts: Counter[int] = Counter()
    residual_values: list[torch.Tensor] = []
    comp_evm_values: list[torch.Tensor] = []
    comp_ber_values: list[torch.Tensor] = []
    uncomp_evm_values: list[torch.Tensor] = []
    uncomp_ber_values: list[torch.Tensor] = []
    alignment_values: list[torch.Tensor] = []

    for _ in tqdm(range(num_batches), desc=f"Building {split} compensation data"):
        batch = generator.sample_batch()
        p_teacher = fixed_precomp(
            x_clean=batch["x_clean"],
            tx_ids=batch["tx_ids"],
            snr_db=batch["snr_db"],
            time_indices=batch["time_indices"],
        )
        residual_ratio = _residual_power_ratio(p_teacher, batch["x_clean"])
        x_pre = batch["x_clean_tx"] + p_teacher
        y_imp = generator.impairment_bank.apply(x_pre, tx_ids=batch["tx_ids"], time_indices=batch["time_indices"])
        y_rx_comp = add_awgn(y_imp, snr_db=batch["snr_db"])
        y_eq_comp = _equalize(y_rx_comp, batch["x_clean_tx"])
        y_eq_uncomp = _equalize(batch["y_rx"], batch["x_clean_tx"])

        comp_evm_values.append(evm_linear(y_eq_comp, batch["x_clean_tx"], align_gain=False).detach().flatten().cpu())
        uncomp_evm_values.append(evm_linear(y_eq_uncomp, batch["x_clean_tx"], align_gain=False).detach().flatten().cpu())
        if is_semantic:
            comp_ber_values.append(torch.zeros_like(comp_evm_values[-1]))
            uncomp_ber_values.append(torch.zeros_like(uncomp_evm_values[-1]))
        else:
            bit_groups = batch["bits"].reshape(batch["x_clean"].shape[0], batch["x_clean"].shape[1], 4)
            bit_groups_tx = bit_groups.unsqueeze(0).expand(batch["x_clean_tx"].shape[0], -1, -1, -1)
            comp_ber_values.append(ber_16qam(y_eq_comp, bit_groups_tx).detach().flatten().cpu())
            uncomp_ber_values.append(ber_16qam(y_eq_uncomp, bit_groups_tx).detach().flatten().cpu())
        alignment_values.append(mean_alignment_loss(extract_raw_features(y_rx_comp)).detach().reshape(1).cpu())
        residual_values.append(residual_ratio.detach().flatten().cpu())

        batch_columns = _flatten_batch(batch, p_teacher, residual_ratio)
        _append_columns(columns, batch_columns)
        tx_counts.update(int(tx_id) for tx_id in batch_columns["tx_id"].tolist())

    data = _cat_columns(columns)
    data["meta"] = {
        "split": split,
        "num_batches": num_batches,
        "num_samples": int(data["tx_id"].numel()),
        "num_symbols": int(data["x_clean"].shape[-1]),
        "format": "columnar tensor dict; dimension 0 indexes individual samples.",
    }
    ensure_dir(output_path.parent)
    torch.save(data, output_path)

    residual_all = torch.cat(residual_values)
    summary = {
        "split": split,
        "num_samples": int(data["tx_id"].numel()),
        "tx_counts": {str(k): int(v) for k, v in sorted(tx_counts.items())},
        "residual_power_mean": float(residual_all.mean().item()),
        "residual_power_std": float(residual_all.std(unbiased=False).item()),
        "residual_power_max": float(residual_all.max().item()),
        "fixed_precomp_evm": float(torch.cat(comp_evm_values).mean().item()),
        "fixed_precomp_ber": float(torch.cat(comp_ber_values).mean().item()),
        "fixed_precomp_alignment": float(torch.cat(alignment_values).mean().item()),
        "uncomp_evm": float(torch.cat(uncomp_evm_values).mean().item()),
        "uncomp_ber": float(torch.cat(uncomp_ber_values).mean().item()),
        "path": str(output_path),
    }
    return summary


def main() -> None:
    """Build train/val/test compensation datasets using fixed pre-compensation."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = get_device(bool(config.get("device", {}).get("prefer_cuda", True)))
    logger = get_logger("rfhide.comp_dataset")
    logger.info("Current step: %s", STEP_NAME)
    logger.info("Config path: %s", args.config)
    logger.info("Device: %s", device)

    comp_cfg = _dataset_cfg(config)
    batch_size = int(args.batch_size if args.batch_size is not None else comp_cfg.get("batch_size", 64))
    split_batches = {
        "train": int(args.train_batches if args.train_batches is not None else comp_cfg.get("train_batches", 8)),
        "val": int(args.val_batches if args.val_batches is not None else comp_cfg.get("val_batches", 2)),
        "test": int(args.test_batches if args.test_batches is not None else comp_cfg.get("test_batches", 2)),
    }
    checkpoint_path = _resolve_project_path(args.checkpoint or comp_cfg.get("checkpoint", "outputs/snr20/checkpoints/teacher_best.pt"))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    generator_cfg = _build_generator_config(config, batch_size=batch_size)
    fixed_precomp = _load_fixed_precomp(generator_cfg, checkpoint_path, device)
    output_dir = PROJECT_ROOT / config.get("experiment", {}).get("output_dir", "outputs/snr20")
    data_dir = ensure_dir(output_dir / "data")
    log_dir = ensure_dir(output_dir / "logs")

    summaries = {}
    for split, num_batches in split_batches.items():
        output_path = data_dir / f"comp_{split}.pt"
        summary = _build_split(split, num_batches, generator_cfg, fixed_precomp, device, output_path)
        summaries[split] = summary
        logger.info(
            "%s | samples %d | tx_counts %s | residual mean/std/max %.6f/%.6f/%.6f",
            split,
            summary["num_samples"],
            summary["tx_counts"],
            summary["residual_power_mean"],
            summary["residual_power_std"],
            summary["residual_power_max"],
        )
        logger.info(
            "%s | fixed precomp EVM %.6f BER %.6f alignment %.6f | uncomp EVM %.6f BER %.6f",
            split,
            summary["fixed_precomp_evm"],
            summary["fixed_precomp_ber"],
            summary["fixed_precomp_alignment"],
            summary["uncomp_evm"],
            summary["uncomp_ber"],
        )
        logger.info("%s saved to: %s", split, summary["path"])

    summary_path = log_dir / "compensation_dataset_summary.json"
    save_json(
        {
            "checkpoint": str(checkpoint_path),
            "batch_size": batch_size,
            "splits": summaries,
        },
        summary_path,
    )
    logger.info("Saved compensation dataset summary: %s", summary_path)
    logger.info("Step 5 fixed precomp dataset build passed")


if __name__ == "__main__":
    main()
