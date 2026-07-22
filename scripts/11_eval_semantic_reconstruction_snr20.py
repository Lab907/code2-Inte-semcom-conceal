"""Evaluate semantic reconstruction quality for the SNR=20 semantic loop."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.logging_utils import get_logger
from rfhide.metrics import estimate_complex_gain
from rfhide.semantic_jscc import channels_to_complex, load_semantic_model
from rfhide.utils import ensure_dir, get_device, save_json, set_seed

STEP_NAME = "Evaluate semantic reconstruction SNR20"
METHODS = ["uncompensated", "random_perturb", "fixed_precomp"]
METHOD_LABELS = {
    "uncompensated": "Uncompensated",
    "random_perturb": "Random perturb",
    "fixed_precomp": "Fixed precomp",
}
METHOD_COLORS = {
    "uncompensated": "#4C78A8",
    "random_perturb": "#F58518",
    "fixed_precomp": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for reconstruction metrics.")
    return parser.parse_args()


def _load_eval_dataset(path: Path) -> dict[str, torch.Tensor]:
    """Load one saved evaluation dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation dataset: {path}")
    data = torch.load(path, map_location="cpu")
    required = ["signals", "x_clean", "semantic_image"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} is missing semantic fields: {missing}")
    return data


def _equalize(y: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """Least-squares equalize received symbols to the JSCC codeword reference."""
    gain = estimate_complex_gain(y, x_ref)
    safe_gain = torch.where(gain.abs() < 1e-12, torch.ones_like(gain), gain)
    return y / safe_gain


def _psnr_from_mse(mse: float) -> float:
    """Compute PSNR for [0, 1] images."""
    return float(10.0 * torch.log10(torch.tensor(1.0 / max(mse, 1e-12))).item())


def _psnr_per_sample(mse: torch.Tensor) -> torch.Tensor:
    """Compute per-sample PSNR for [0, 1] images."""
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def _ssim_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute a lightweight local SSIM score per image."""
    if x.shape != y.shape:
        raise ValueError(f"SSIM expects matching shapes, got {tuple(x.shape)} and {tuple(y.shape)}")
    c1 = 0.01**2
    c2 = 0.03**2
    kernel_size = 7 if x.shape[-1] >= 7 and x.shape[-2] >= 7 else 3
    padding = kernel_size // 2
    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=padding)
    mu_x2 = mu_x.pow(2)
    mu_y2 = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x2 = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding) - mu_x2
    sigma_y2 = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_y2
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_xy
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / denominator.clamp_min(1e-12)
    return ssim_map.flatten(start_dim=1).mean(dim=1).clamp(-1.0, 1.0)


def _to_luma(images: torch.Tensor) -> torch.Tensor:
    """Convert image tensors to one-channel luma for structure features."""
    if images.ndim != 4 or images.shape[1] not in {1, 3}:
        raise ValueError(f"Expected image tensor [B, 1|3, H, W], got {tuple(images.shape)}")
    if images.shape[1] == 1:
        return images
    weights = torch.tensor([0.299, 0.587, 0.114], device=images.device, dtype=images.dtype).reshape(1, 3, 1, 1)
    return (images * weights).sum(dim=1, keepdim=True)


def _face_feature_embedding(images: torch.Tensor) -> torch.Tensor:
    """Extract a lightweight face-structure feature embedding.

    This is a dependency-free proxy for face identity/feature preservation. It
    combines low-resolution appearance with pooled Sobel gradients, then L2
    normalizes the result for cosine similarity.
    """
    images = _to_luma(images).clamp(0.0, 1.0)
    low = F.interpolate(images, size=(16, 16), mode="bilinear", align_corners=False)
    centered = images - images.mean(dim=(-2, -1), keepdim=True)
    std = centered.flatten(start_dim=1).std(dim=1, keepdim=True, unbiased=False).reshape(-1, 1, 1, 1).clamp_min(1e-6)
    normalized = centered / std
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=images.device,
        dtype=images.dtype,
    ).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=images.device,
        dtype=images.dtype,
    ).reshape(1, 1, 3, 3)
    grad_x = F.conv2d(normalized, sobel_x, padding=1)
    grad_y = F.conv2d(normalized, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-8)
    grad_low = F.interpolate(grad_mag, size=(16, 16), mode="bilinear", align_corners=False)
    embedding = torch.cat([low.flatten(start_dim=1), grad_low.flatten(start_dim=1)], dim=1)
    return F.normalize(embedding, p=2, dim=1, eps=1e-8)


def _face_feature_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return per-sample cosine similarity for lightweight face features."""
    return (_face_feature_embedding(x) * _face_feature_embedding(y)).sum(dim=1).clamp(-1.0, 1.0)


@torch.no_grad()
def _decode_dataset(
    model,
    data: dict[str, torch.Tensor],
    device: torch.device,
    max_samples: int | None,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Decode one method dataset and return reconstructions plus metrics."""
    count = data["signals"].shape[0] if max_samples is None else min(max_samples, data["signals"].shape[0])
    signals = data["signals"][:count].to(device).float()
    x_clean = data["x_clean"][:count].to(device).float()
    target = data["semantic_image"][:count].to(device).float()
    y = channels_to_complex(signals)
    x_ref = channels_to_complex(x_clean)
    y_eq = _equalize(y, x_ref)
    recon_chunks: list[torch.Tensor] = []
    mse_values: list[torch.Tensor] = []
    mae_values: list[torch.Tensor] = []
    ssim_values: list[torch.Tensor] = []
    face_feature_values: list[torch.Tensor] = []
    for start in range(0, count, 128):
        recon = model.decode(y_eq[start : start + 128])
        truth = target[start : start + 128]
        recon_chunks.append(recon.cpu())
        mse_values.append(F.mse_loss(recon, truth, reduction="none").flatten(start_dim=1).mean(dim=1).cpu())
        mae_values.append(F.l1_loss(recon, truth, reduction="none").flatten(start_dim=1).mean(dim=1).cpu())
        recon_clamped = recon.clamp(0.0, 1.0)
        truth_clamped = truth.clamp(0.0, 1.0)
        ssim_values.append(_ssim_per_sample(recon_clamped, truth_clamped).cpu())
        face_feature_values.append(_face_feature_cosine(recon_clamped, truth_clamped).cpu())
    per_mse = torch.cat(mse_values)
    per_mae = torch.cat(mae_values)
    per_psnr = _psnr_per_sample(per_mse)
    per_ssim = torch.cat(ssim_values)
    per_face_feature = torch.cat(face_feature_values)
    mse = float(per_mse.mean().item())
    metrics = {
        "num_samples": float(count),
        "semantic_mse": mse,
        "semantic_psnr": _psnr_from_mse(mse),
        "semantic_mse_std": float(per_mse.std(unbiased=False).item()),
        "semantic_mae": float(per_mae.mean().item()),
        "semantic_mae_std": float(per_mae.std(unbiased=False).item()),
        "semantic_psnr_mean": float(per_psnr.mean().item()),
        "semantic_psnr_std": float(per_psnr.std(unbiased=False).item()),
        "semantic_ssim": float(per_ssim.mean().item()),
        "semantic_ssim_std": float(per_ssim.std(unbiased=False).item()),
        "face_feature_cosine": float(per_face_feature.mean().item()),
        "face_feature_cosine_std": float(per_face_feature.std(unbiased=False).item()),
    }
    per_sample_metrics = {
        "mse": per_mse,
        "mae": per_mae,
        "psnr": per_psnr,
        "ssim": per_ssim,
        "face_feature_cosine": per_face_feature,
    }
    return torch.cat(recon_chunks, dim=0), metrics, per_sample_metrics


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write semantic reconstruction metrics to CSV."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_per_sample_csv(path: Path, per_sample: dict[str, dict[str, torch.Tensor]]) -> None:
    """Write per-sample semantic metrics for all methods."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["method", "sample_index", "mse", "mae", "psnr", "ssim", "face_feature_cosine"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method in METHODS:
            metrics = per_sample[method]
            count = int(metrics["mse"].numel())
            for index in range(count):
                writer.writerow(
                    {
                        "method": method,
                        "sample_index": index,
                        "mse": float(metrics["mse"][index].item()),
                        "mae": float(metrics["mae"][index].item()),
                        "psnr": float(metrics["psnr"][index].item()),
                        "ssim": float(metrics["ssim"][index].item()),
                        "face_feature_cosine": float(metrics["face_feature_cosine"][index].item()),
                    }
                )


def _write_grid_psnr_csv(
    path: Path,
    indices: torch.Tensor,
    labels: torch.Tensor,
    per_sample: dict[str, dict[str, torch.Tensor]],
) -> list[dict[str, Any]]:
    """Write PSNR values for the exact images shown in the reconstruction grid."""
    ensure_dir(path.parent)
    rows: list[dict[str, Any]] = []
    for col, sample_idx in enumerate(indices.tolist(), start=1):
        row: dict[str, Any] = {
            "grid_column": col,
            "sample_index": int(sample_idx),
            "semantic_label": int(labels[sample_idx].item()),
        }
        for method in METHODS:
            row[f"{method}_psnr_db"] = float(per_sample[method]["psnr"][sample_idx].item())
        rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "grid_column",
            "sample_index",
            "semantic_label",
            *[f"{method}_psnr_db" for method in METHODS],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _print_grid_psnr(rows: list[dict[str, Any]]) -> None:
    """Print PSNR values for the reconstruction grid images."""
    print("PSNR values for images shown in semantic_reconstruction_methods_snr20.png:")
    for row in rows:
        values = " | ".join(
            f"{METHOD_LABELS[method]}: {float(row[f'{method}_psnr_db']):.2f} dB"
            for method in METHODS
        )
        print(
            f"Column {int(row['grid_column']):02d} "
            f"(sample {int(row['sample_index'])}, label {int(row['semantic_label'])}) | {values}"
        )


def _plot_quality_bars(path: Path, rows: list[dict[str, Any]]) -> None:
    """Save bar charts for mean semantic reconstruction quality."""
    metrics = [
        ("semantic_mse", "MSE lower is better"),
        ("semantic_mae", "MAE lower is better"),
        ("semantic_psnr_mean", "PSNR dB higher is better"),
        ("semantic_ssim", "SSIM higher is better"),
        ("face_feature_cosine", "Face feature cosine higher is better"),
    ]
    labels = [METHOD_LABELS[row["method"]] for row in rows]
    colors = [METHOD_COLORS[row["method"]] for row in rows]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.0))
    for ax, (metric, title) in zip(axes, metrics):
        values = [float(row[metric]) for row in rows]
        bars = ax.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.7)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            text = f"{value:.4f}" if metric != "semantic_psnr_mean" else f"{value:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), text, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_quality_distributions(path: Path, per_sample: dict[str, dict[str, torch.Tensor]]) -> None:
    """Save box plots for per-sample semantic reconstruction quality."""
    metrics = [
        ("mse", "Per-sample MSE"),
        ("psnr", "Per-sample PSNR dB"),
        ("ssim", "Per-sample SSIM"),
        ("face_feature_cosine", "Per-sample face feature cosine"),
    ]
    labels = [METHOD_LABELS[method] for method in METHODS]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.0))
    for ax, (metric, title) in zip(axes, metrics):
        values = [per_sample[method][metric].numpy() for method in METHODS]
        box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, method in zip(box["boxes"], METHODS):
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.72)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_psnr_cdf(path: Path, per_sample: dict[str, dict[str, torch.Tensor]]) -> None:
    """Save empirical CDF curves for per-sample PSNR."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for method in METHODS:
        values = np.sort(per_sample[method]["psnr"].numpy())
        cdf = np.arange(1, values.size + 1) / max(values.size, 1)
        ax.plot(values, cdf, label=METHOD_LABELS[method], color=METHOD_COLORS[method], linewidth=2.0)
    ax.set_xlabel("PSNR (dB)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Semantic reconstruction PSNR distribution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _choose_grid_indices(labels: torch.Tensor, max_count: int = 10) -> torch.Tensor:
    """Choose a compact set of examples, preferring one per digit label."""
    chosen: list[int] = []
    seen: set[int] = set()
    for index, label in enumerate(labels.tolist()):
        digit = int(label)
        if digit not in seen:
            chosen.append(index)
            seen.add(digit)
        if len(chosen) >= max_count:
            break
    if len(chosen) < max_count:
        chosen.extend(index for index in range(labels.numel()) if index not in chosen)
    return torch.as_tensor(chosen[:max_count], dtype=torch.long)


def _show_image(ax: plt.Axes, image: torch.Tensor) -> None:
    """Show a CHW image tensor as grayscale or RGB."""
    image = image.detach().cpu().clamp(0.0, 1.0)
    if image.shape[0] == 1:
        ax.imshow(image[0], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    else:
        ax.imshow(image.permute(1, 2, 0), vmin=0.0, vmax=1.0, interpolation="nearest")


def _save_method_grid(
    path: Path,
    target: torch.Tensor,
    labels: torch.Tensor,
    reconstructions: dict[str, torch.Tensor],
    methods: list[str] | None = None,
    indices: torch.Tensor | None = None,
) -> None:
    """Save original images and method reconstructions in one grid."""
    methods = METHODS if methods is None else methods
    indices = _choose_grid_indices(labels, max_count=10) if indices is None else indices
    rows = ["original"] + methods
    fig, axes = plt.subplots(len(rows), indices.numel(), figsize=(1.25 * indices.numel(), 1.35 * len(rows)))
    for col, sample_idx in enumerate(indices.tolist()):
        _show_image(axes[0, col], target[sample_idx])
        axes[0, col].set_title(str(int(labels[sample_idx].item())), fontsize=8)
        axes[0, col].axis("off")
        for row_idx, method in enumerate(methods, start=1):
            _show_image(axes[row_idx, col], reconstructions[method][sample_idx])
            axes[row_idx, col].axis("off")
    axes[0, 0].set_ylabel("Original")
    for row_idx, method in enumerate(methods, start=1):
        axes[row_idx, 0].set_ylabel(METHOD_LABELS[method])
    fig.tight_layout()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    """Evaluate semantic reconstruction from collected RF signals."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = get_device(bool(config.get("device", {}).get("prefer_cuda", True)))
    logger = get_logger("rfhide.semantic_eval")
    logger.info("Current step: %s", STEP_NAME)
    logger.info("Config path: %s", args.config)
    logger.info("Device: %s", device)

    output_dir = PROJECT_ROOT / config.get("experiment", {}).get("output_dir", "outputs/snr20")
    data_dir = output_dir / "data"
    log_dir = ensure_dir(output_dir / "logs")
    fig_dir = ensure_dir(output_dir / "figures")

    model = load_semantic_model(config, device, require_checkpoint=True)
    datasets = {method: _load_eval_dataset(data_dir / f"eval_{method}.pt") for method in METHODS}

    rows: list[dict[str, Any]] = []
    reconstructions: dict[str, torch.Tensor] = {}
    per_sample_metrics: dict[str, dict[str, torch.Tensor]] = {}
    for method in METHODS:
        recon, metrics, sample_metrics = _decode_dataset(model, datasets[method], device, args.max_samples)
        row = {"method": method}
        row.update(metrics)
        rows.append(row)
        reconstructions[method] = recon
        per_sample_metrics[method] = sample_metrics
        logger.info(
            "%s | samples %d | MSE %.6f | MAE %.6f | PSNR %.2f dB | SSIM %.4f | face cosine %.4f",
            method,
            int(metrics["num_samples"]),
            metrics["semantic_mse"],
            metrics["semantic_mae"],
            metrics["semantic_psnr_mean"],
            metrics["semantic_ssim"],
            metrics["face_feature_cosine"],
        )

    summary_json = log_dir / "semantic_reconstruction_summary.json"
    summary_csv = log_dir / "semantic_reconstruction_summary.csv"
    per_sample_csv = log_dir / "semantic_reconstruction_per_sample.csv"
    grid_psnr_csv = log_dir / "semantic_reconstruction_grid_psnr.csv"
    grid_path = fig_dir / "semantic_reconstruction_methods_snr20.png"
    grid_uncomp_path = fig_dir / "semantic_reconstruction_original_uncompensated_snr20.png"
    grid_uncomp_fixed_path = fig_dir / "semantic_reconstruction_original_uncompensated_fixed_precomp_snr20.png"
    bar_path = fig_dir / "semantic_quality_bars_snr20.png"
    box_path = fig_dir / "semantic_quality_boxplots_snr20.png"
    cdf_path = fig_dir / "semantic_quality_psnr_cdf_snr20.png"
    first = datasets[METHODS[0]]
    target_count = min(reconstructions[METHODS[0]].shape[0], first["semantic_image"].shape[0])
    labels = first.get("semantic_label", torch.arange(target_count))
    grid_indices = _choose_grid_indices(labels[:target_count], max_count=10)
    _save_method_grid(grid_path, first["semantic_image"][:target_count], labels[:target_count], reconstructions, indices=grid_indices)
    _save_method_grid(
        grid_uncomp_path,
        first["semantic_image"][:target_count],
        labels[:target_count],
        reconstructions,
        methods=["uncompensated"],
        indices=grid_indices,
    )
    _save_method_grid(
        grid_uncomp_fixed_path,
        first["semantic_image"][:target_count],
        labels[:target_count],
        reconstructions,
        methods=["uncompensated", "fixed_precomp"],
        indices=grid_indices,
    )
    grid_psnr_rows = _write_grid_psnr_csv(grid_psnr_csv, grid_indices, labels[:target_count], per_sample_metrics)
    _print_grid_psnr(grid_psnr_rows)
    _plot_quality_bars(bar_path, rows)
    _plot_quality_distributions(box_path, per_sample_metrics)
    _plot_psnr_cdf(cdf_path, per_sample_metrics)
    save_json(
        {
            "summary": rows,
            "figures": {
                "reconstruction_grid": str(grid_path),
                "original_uncompensated_grid": str(grid_uncomp_path),
                "original_uncompensated_fixed_precomp_grid": str(grid_uncomp_fixed_path),
                "quality_bars": str(bar_path),
                "quality_boxplots": str(box_path),
                "psnr_cdf": str(cdf_path),
            },
            "grid_psnr_csv": str(grid_psnr_csv),
        },
        summary_json,
    )
    _write_summary_csv(summary_csv, rows)
    _write_per_sample_csv(per_sample_csv, per_sample_metrics)
    logger.info("Saved semantic summary JSON: %s", summary_json)
    logger.info("Saved semantic summary CSV: %s", summary_csv)
    logger.info("Saved per-sample semantic metrics CSV: %s", per_sample_csv)
    logger.info("Saved grid PSNR CSV: %s", grid_psnr_csv)
    logger.info("Saved semantic reconstruction figures: %s | %s | %s", grid_path, grid_uncomp_path, grid_uncomp_fixed_path)
    logger.info("Saved semantic quality figures: %s | %s | %s", bar_path, box_path, cdf_path)
    logger.info("Semantic reconstruction evaluation passed")


if __name__ == "__main__":
    main()
