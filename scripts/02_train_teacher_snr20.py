"""Step 4 optimize fixed transmitter-side pre-compensation at SNR=20."""

from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from rfhide.losses import (
    cov_alignment_loss,
    mean_alignment_loss,
    pairwise_mmd_loss,
    power_loss,
    soft_bit_loss,
)
from rfhide.metrics import ber_16qam, estimate_complex_gain, evm_linear
from rfhide.fixed_precomp import FixedPrecompensator
from rfhide.semantic_jscc import semantic_enabled
from rfhide.utils import count_parameters, ensure_dir, get_device, save_json, set_seed

STEP_NAME = "Step 4 optimize fixed precomp SNR20"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override teacher.epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Override teacher.num_steps_per_epoch.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override teacher.batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override teacher.lr.")
    return parser.parse_args()


def _teacher_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Return teacher config with defaults."""
    return config.get("teacher", {})


def _training_config(config: dict[str, Any], batch_size: int) -> dict[str, Any]:
    """Copy config and override data batch size for teacher training."""
    train_cfg = deepcopy(config)
    train_cfg.setdefault("data", {})
    train_cfg["data"]["batch_size"] = batch_size
    return train_cfg


def _equalize_per_tx(y_rx: torch.Tensor, x_clean_tx: torch.Tensor) -> torch.Tensor:
    """Least-squares equalize each Tx/sample against its clean reference."""
    gain = estimate_complex_gain(y_rx, x_clean_tx)
    safe_gain = torch.where(gain.abs() < 1e-12, torch.ones_like(gain), gain)
    return y_rx / safe_gain


def _residual_power_ratio(p: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
    """Return per-Tx/sample residual power ratio."""
    x_power = x_clean.abs().pow(2).mean(dim=-1).unsqueeze(0).clamp_min(1e-12)
    p_power = p.abs().pow(2).mean(dim=-1)
    return p_power / x_power


def _forward_signal_chain(
    model: FixedPrecompensator,
    generator: MultiTxBatchGenerator,
    batch: dict[str, torch.Tensor],
    use_teacher: bool,
) -> dict[str, torch.Tensor]:
    """Apply optional fixed pre-compensation, hardware impairments, and AWGN."""
    x_clean = batch["x_clean"]
    x_clean_tx = batch["x_clean_tx"]
    if use_teacher:
        p = model(
            x_clean=batch["x_clean"],
            tx_ids=batch["tx_ids"],
            snr_db=batch["snr_db"],
            time_indices=batch["time_indices"],
        )
    else:
        p = torch.zeros_like(x_clean_tx)

    x_pre = x_clean_tx + p
    y_imp = generator.impairment_bank.apply(
        x_pre,
        tx_ids=batch["tx_ids"],
        time_indices=batch["time_indices"],
    )
    y_rx = add_awgn(y_imp, snr_db=batch["snr_db"])
    y_eq = _equalize_per_tx(y_rx, x_clean_tx)
    return {"p": p, "x_pre": x_pre, "y_imp": y_imp, "y_rx": y_rx, "y_eq": y_eq}


def _compute_breakdown(
    model: FixedPrecompensator,
    generator: MultiTxBatchGenerator,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    use_teacher: bool,
) -> dict[str, torch.Tensor]:
    """Compute loss and metrics for one batch."""
    teacher_cfg = _teacher_cfg(config)
    signal = _forward_signal_chain(model, generator, batch, use_teacher=use_teacher)
    features = extract_raw_features(signal["y_rx"], downsample=int(teacher_cfg.get("feature_downsample", 4)))

    mean_loss = mean_alignment_loss(features)
    cov_loss = cov_alignment_loss(features)
    mmd_loss = pairwise_mmd_loss(features, sigmas=teacher_cfg.get("mmd_sigmas", [0.5, 1.0, 2.0, 4.0, 8.0]))
    align_loss = (
        float(teacher_cfg.get("lambda_mean", 1.0)) * mean_loss
        + float(teacher_cfg.get("lambda_cov", 1.0)) * cov_loss
        + float(teacher_cfg.get("lambda_mmd", 1.0)) * mmd_loss
    )
    evm = evm_linear(signal["y_eq"], batch["x_clean_tx"], align_gain=False).mean()
    is_semantic = semantic_enabled(config)
    if is_semantic:
        soft_bits = evm.new_zeros(())
    else:
        soft_bits = soft_bit_loss(signal["y_eq"], batch["bits"], cfg={"soft_bit_temperature": 0.05})
    x_power = batch["x_clean"].abs().pow(2).mean(dim=-1).unsqueeze(0)
    max_residual_power = float(teacher_cfg.get("max_residual_power_ratio", 0.05)) * x_power
    residual_power = power_loss(signal["p"], max_power=max_residual_power)
    residual_ratio = _residual_power_ratio(signal["p"], batch["x_clean"]).mean()
    if is_semantic:
        ber = evm.new_zeros(())
    else:
        bit_groups = batch["bits"].reshape(batch["x_clean"].shape[0], batch["x_clean"].shape[1], 4)
        bit_groups_tx = bit_groups.unsqueeze(0).expand(batch["x_clean_tx"].shape[0], -1, -1, -1)
        ber = ber_16qam(signal["y_eq"], bit_groups_tx).mean()

    total = (
        float(teacher_cfg.get("lambda_align", 1.0)) * align_loss
        + float(teacher_cfg.get("lambda_evm", 5.0)) * evm
        + float(teacher_cfg.get("lambda_softbit", 2.0)) * soft_bits
        + float(teacher_cfg.get("lambda_power", 10.0)) * residual_power
    )
    return {
        "total": total,
        "align": align_loss,
        "mean": mean_loss,
        "cov": cov_loss,
        "mmd": mmd_loss,
        "evm": evm,
        "ber": ber,
        "softbit": soft_bits,
        "power": residual_power,
        "residual_power_ratio": residual_ratio,
    }


def _detach_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """Convert tensor metrics to plain floats."""
    return {name: float(value.detach().cpu().item()) for name, value in metrics.items()}


@torch.no_grad()
def _evaluate_batches(
    model: FixedPrecompensator,
    generator: MultiTxBatchGenerator,
    batches: list[dict[str, torch.Tensor]],
    config: dict[str, Any],
    use_teacher: bool,
) -> dict[str, float]:
    """Average metrics over fixed batches."""
    totals: dict[str, float] = {}
    for batch in batches:
        metrics = _detach_metrics(_compute_breakdown(model, generator, batch, config, use_teacher=use_teacher))
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
    return {name: value / len(batches) for name, value in totals.items()}


def _write_metrics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    """Write epoch metrics to CSV."""
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(path: Path, rows: list[dict[str, float]]) -> None:
    """Save teacher training curves."""
    ensure_dir(path.parent)
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(8, 5))
    for key in ["total", "align", "evm", "softbit", "residual_power_ratio"]:
        plt.plot(epochs, [row[key] for row in rows], marker="o", label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_checkpoint(path: Path, model: FixedPrecompensator, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float]) -> None:
    """Save a fixed pre-compensation checkpoint."""
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "model_type": "fixed_precompensator",
        },
        path,
    )


def main() -> None:
    """Optimize fixed pre-compensation parameters for SNR=20."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    teacher_cfg = _teacher_cfg(config)
    epochs = int(args.epochs if args.epochs is not None else teacher_cfg.get("epochs", 30))
    steps_per_epoch = int(
        args.steps_per_epoch if args.steps_per_epoch is not None else teacher_cfg.get("num_steps_per_epoch", 100)
    )
    batch_size = int(args.batch_size if args.batch_size is not None else teacher_cfg.get("batch_size", 128))
    lr = float(args.lr if args.lr is not None else teacher_cfg.get("lr", 0.001))

    device = get_device(bool(config.get("device", {}).get("prefer_cuda", True)))
    logger = get_logger("rfhide.fixed_precomp")
    logger.info("Current step: %s", STEP_NAME)
    logger.info("Config path: %s", args.config)
    logger.info("Device: %s", device)
    logger.info("Epochs: %d | steps/epoch: %d | batch size: %d | lr: %.6f", epochs, steps_per_epoch, batch_size, lr)

    train_cfg = _training_config(config, batch_size=batch_size)
    model = FixedPrecompensator(train_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    logger.info("Fixed precomp trainable parameters: %d", count_parameters(model))

    output_dir = PROJECT_ROOT / config.get("experiment", {}).get("output_dir", "outputs/snr20")
    log_dir = ensure_dir(output_dir / "logs")
    fig_dir = ensure_dir(output_dir / "figures")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    metrics_csv = log_dir / "fixed_precomp_train_metrics.csv"
    before_after_path = log_dir / "fixed_precomp_before_after.json"
    curve_path = fig_dir / "fixed_precomp_loss_curves.png"
    best_path = checkpoint_dir / "teacher_best.pt"
    last_path = checkpoint_dir / "teacher_last.pt"

    eval_steps = max(1, min(3, steps_per_epoch))
    train_generator = MultiTxBatchGenerator(train_cfg, split="train", device=device)
    eval_batches = [train_generator.sample_batch() for _ in range(eval_steps)]
    before = _evaluate_batches(model, train_generator, eval_batches, train_cfg, use_teacher=False)
    logger.info(
        "Before fixed precomp | loss %.6f | align %.6f | EVM %.6f | BER %.6f | residual ratio %.6f",
        before["total"],
        before["align"],
        before["evm"],
        before["ber"],
        before["residual_power_ratio"],
    )

    rows: list[dict[str, float]] = []
    best_loss = float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_totals: dict[str, float] = {}
        progress = tqdm(range(steps_per_epoch), desc=f"Fixed precomp epoch {epoch}/{epochs}")
        for _ in progress:
            batch = train_generator.sample_batch()
            metrics = _compute_breakdown(model, train_generator, batch, train_cfg, use_teacher=True)
            optimizer.zero_grad(set_to_none=True)
            metrics["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            values = _detach_metrics(metrics)
            for name, value in values.items():
                epoch_totals[name] = epoch_totals.get(name, 0.0) + value
            progress.set_postfix(
                loss=f"{values['total']:.4f}",
                align=f"{values['align']:.4f}",
                evm=f"{values['evm']:.4f}",
                ber=f"{values['ber']:.4f}",
            )

        train_row = {name: value / steps_per_epoch for name, value in epoch_totals.items()}
        eval_row = _evaluate_batches(model, train_generator, eval_batches, train_cfg, use_teacher=True)
        row = {"epoch": float(epoch)}
        row.update(eval_row)
        row.update({f"train_{name}": value for name, value in train_row.items()})
        rows.append(row)
        _write_metrics_csv(metrics_csv, rows)
        _plot_curves(curve_path, rows)

        logger.info(
            "Epoch %d | loss %.6f | align %.6f | mean %.6f | cov %.6f | mmd %.6f | EVM %.6f | BER %.6f | soft-bit %.6f | residual ratio %.6f",
            epoch,
            row["total"],
            row["align"],
            row["mean"],
            row["cov"],
            row["mmd"],
            row["evm"],
            row["ber"],
            row["softbit"],
            row["residual_power_ratio"],
        )

        if row["total"] < best_loss:
            best_loss = row["total"]
            best_model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            _save_checkpoint(best_path, model, optimizer, epoch, row)
        elif best_model_state is not None:
            model.load_state_dict({name: value.to(device) for name, value in best_model_state.items()})
        _save_checkpoint(last_path, model, optimizer, epoch, row)

    model.eval()
    if best_model_state is not None:
        model.load_state_dict({name: value.to(device) for name, value in best_model_state.items()})
    after = _evaluate_batches(model, train_generator, eval_batches, train_cfg, use_teacher=True)
    before_after = {
        "before": before,
        "after": after,
        "loss_decreased": after["total"] < before["total"],
        "alignment_decreased": after["align"] < before["align"],
        "residual_power_threshold": float(teacher_cfg.get("max_residual_power_ratio", 0.05)),
        "residual_power_ok": after["residual_power_ratio"] <= float(teacher_cfg.get("max_residual_power_ratio", 0.05)) + 1e-5,
        "checkpoints": {
            "best": str(best_path),
            "last": str(last_path),
        },
    }
    save_json(before_after, before_after_path)

    logger.info(
        "After fixed precomp | loss %.6f | align %.6f | EVM %.6f | BER %.6f | residual ratio %.6f",
        after["total"],
        after["align"],
        after["evm"],
        after["ber"],
        after["residual_power_ratio"],
    )
    logger.info("Saved metrics CSV: %s", metrics_csv)
    logger.info("Saved before/after JSON: %s", before_after_path)
    logger.info("Saved loss curves: %s", curve_path)
    logger.info("Saved best checkpoint: %s", best_path)
    logger.info("Saved last checkpoint: %s", last_path)
    logger.info("Step 4 fixed precomp optimization passed")


if __name__ == "__main__":
    main()
