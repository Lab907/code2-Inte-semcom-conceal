"""Train the lightweight semantic JSCC model for the SNR=20 loop."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.channel import add_awgn
from rfhide.config import load_config
from rfhide.logging_utils import get_logger
from rfhide.semantic_jscc import SemanticJSCC, load_semantic_split, resolve_semantic_checkpoint
from rfhide.utils import count_parameters, ensure_dir, get_device, save_json, set_seed

STEP_NAME = "Train semantic JSCC SNR20"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override semantic.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override semantic.batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override semantic.lr.")
    return parser.parse_args()


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve paths relative to the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _semantic_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """Return semantic config with defaults."""
    return config.get("semantic", {})


def _psnr_from_mse(mse: float) -> float:
    """Compute PSNR for images in [0, 1]."""
    return float(10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item())


def _sample_training_snr(base_snr_db: float, jitter_db: float, batch_size: int, device: torch.device) -> torch.Tensor | float:
    """Sample per-image SNR values for semantic JSCC training."""
    if jitter_db <= 0.0:
        return base_snr_db
    offsets = (torch.rand(batch_size, device=device) * 2.0 - 1.0) * jitter_db
    return base_snr_db + offsets


def _reconstruction_loss(recon: torch.Tensor, target: torch.Tensor, l1_weight: float) -> torch.Tensor:
    """Combine MSE and a small L1 term for sharper patch reconstructions."""
    return F.mse_loss(recon, target) + l1_weight * F.l1_loss(recon, target)


def _write_metrics_csv(path: Path, rows: list[dict[str, float]]) -> None:
    """Write training metrics to CSV."""
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _show_image(ax: plt.Axes, image: torch.Tensor) -> None:
    """Show a CHW image tensor as grayscale or RGB."""
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.shape[0] == 1:
        ax.imshow(image[0], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    else:
        ax.imshow(image.permute(1, 2, 0), vmin=0.0, vmax=1.0, interpolation="nearest")


@torch.no_grad()
def _evaluate(model: SemanticJSCC, loader: DataLoader, snr_db: float, device: torch.device) -> dict[str, float]:
    """Evaluate reconstruction quality through an AWGN JSCC channel."""
    model.eval()
    losses: list[torch.Tensor] = []
    for (images,) in loader:
        images = images.to(device)
        x = model.encode(images)
        y = add_awgn(x, snr_db=snr_db)
        recon = model.decode(y)
        losses.append(F.mse_loss(recon, images, reduction="none").flatten(start_dim=1).mean(dim=1).cpu())
    mse = float(torch.cat(losses).mean().item())
    return {"mse": mse, "psnr": _psnr_from_mse(mse)}


@torch.no_grad()
def _save_reconstruction_grid(model: SemanticJSCC, images: torch.Tensor, snr_db: float, path: Path) -> None:
    """Save a compact original/reconstruction comparison grid."""
    model.eval()
    count = min(10, images.shape[0])
    sample = images[:count].to(next(model.parameters()).device)
    recon = model.decode(add_awgn(model.encode(sample), snr_db=snr_db)).cpu()
    original = sample.cpu()

    fig, axes = plt.subplots(2, count, figsize=(1.2 * count, 2.6))
    for idx in range(count):
        _show_image(axes[0, idx], original[idx])
        axes[0, idx].axis("off")
        _show_image(axes[1, idx], recon[idx])
        axes[1, idx].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Recon")
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_checkpoint(path: Path, model: SemanticJSCC, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float]) -> None:
    """Save a semantic JSCC checkpoint."""
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    """Train the semantic JSCC autoencoder."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    semantic_cfg = _semantic_cfg(config)

    epochs = int(args.epochs if args.epochs is not None else semantic_cfg.get("epochs", 20))
    batch_size = int(args.batch_size if args.batch_size is not None else semantic_cfg.get("batch_size", 128))
    lr = float(args.lr if args.lr is not None else semantic_cfg.get("lr", 0.001))
    snr_db = float(config.get("signal", {}).get("snr_db", 20.0))
    clean_loss_weight = float(semantic_cfg.get("clean_loss_weight", 0.0))
    l1_loss_weight = float(semantic_cfg.get("l1_loss_weight", 0.0))
    snr_jitter_db = float(semantic_cfg.get("snr_jitter_db", 0.0))

    device = get_device(bool(config.get("device", {}).get("prefer_cuda", True)))
    logger = get_logger("rfhide.semantic_jscc")
    logger.info("Current step: %s", STEP_NAME)
    logger.info("Config path: %s", args.config)
    logger.info("Device: %s", device)
    logger.info(
        "Epochs: %d | batch size: %d | lr: %.6f | channel SNR %.2f dB | clean weight %.3f | L1 weight %.3f | SNR jitter %.2f dB",
        epochs,
        batch_size,
        lr,
        snr_db,
        clean_loss_weight,
        l1_loss_weight,
        snr_jitter_db,
    )

    train_images, _ = load_semantic_split(config, "train", device="cpu")
    val_images, _ = load_semantic_split(config, "val", device="cpu")
    train_loader = DataLoader(TensorDataset(train_images), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_images), batch_size=batch_size, shuffle=False)

    model = SemanticJSCC(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    logger.info("Semantic JSCC trainable parameters: %d", count_parameters(model))

    output_dir = PROJECT_ROOT / config.get("experiment", {}).get("output_dir", "outputs/snr20")
    log_dir = ensure_dir(output_dir / "logs")
    fig_dir = ensure_dir(output_dir / "figures")
    checkpoint_path = _resolve_project_path(resolve_semantic_checkpoint(config))
    metrics_path = log_dir / "semantic_jscc_train_metrics.csv"
    summary_path = log_dir / "semantic_jscc_summary.json"
    grid_path = fig_dir / "semantic_jscc_reconstruction_snr20.png"

    rows: list[dict[str, float]] = []
    best_val = float("inf")
    best_metrics: dict[str, float] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        channel_losses: list[float] = []
        clean_losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Semantic JSCC {epoch}/{epochs}")
        for (images,) in progress:
            images = images.to(device)
            x = model.encode(images)
            train_snr = _sample_training_snr(snr_db, snr_jitter_db, images.shape[0], device)
            y = add_awgn(x, snr_db=train_snr)
            recon = model.decode(y)
            channel_loss = _reconstruction_loss(recon, images, l1_loss_weight)
            clean_loss = _reconstruction_loss(model.decode(x), images, l1_loss_weight)
            loss = channel_loss + clean_loss_weight * clean_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            value = float(loss.detach().cpu().item())
            losses.append(value)
            channel_losses.append(float(channel_loss.detach().cpu().item()))
            clean_losses.append(float(clean_loss.detach().cpu().item()))
            progress.set_postfix(mse=f"{value:.5f}")

        val_metrics = _evaluate(model, val_loader, snr_db, device)
        row = {
            "epoch": float(epoch),
            "train_loss": float(sum(losses) / max(len(losses), 1)),
            "train_channel_mse": float(sum(channel_losses) / max(len(channel_losses), 1)),
            "train_clean_mse": float(sum(clean_losses) / max(len(clean_losses), 1)),
            "val_mse": val_metrics["mse"],
            "val_psnr": val_metrics["psnr"],
        }
        rows.append(row)
        _write_metrics_csv(metrics_path, rows)
        logger.info(
            "Epoch %d | train MSE %.6f | val MSE %.6f | val PSNR %.2f dB",
            epoch,
            row["train_channel_mse"],
            row["val_mse"],
            row["val_psnr"],
        )
        if row["val_mse"] < best_val:
            best_val = row["val_mse"]
            best_metrics = row
            _save_checkpoint(checkpoint_path, model, optimizer, epoch, row)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _save_reconstruction_grid(model, val_images, snr_db, grid_path)
    save_json(
        {
            "checkpoint": str(checkpoint_path),
            "best": best_metrics,
            "figure": str(grid_path),
            "num_symbols": int(config.get("signal", {}).get("num_symbols", 1024)),
            "clean_loss_weight": clean_loss_weight,
            "l1_loss_weight": l1_loss_weight,
            "snr_jitter_db": snr_jitter_db,
        },
        summary_path,
    )
    logger.info("Saved semantic checkpoint: %s", checkpoint_path)
    logger.info("Saved metrics CSV: %s", metrics_path)
    logger.info("Saved reconstruction figure: %s", grid_path)
    logger.info("Semantic JSCC training passed")


if __name__ == "__main__":
    main()
