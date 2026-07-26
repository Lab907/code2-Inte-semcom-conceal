"""Train Eve with 300/100/100 per Tx at SNR=20 and plot PSNR/accuracy."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.logging_utils import get_logger
from rfhide.utils import ensure_dir, save_json, set_seed, get_device

METHODS = ["uncompensated", "random_perturb", "fixed_precomp"]
DISPLAY_LABELS = ["Original", "Random Perturb", "Pre-Concealment"]
DATASETS = {
    "uncompensated": "auth/outputs/eve_700_300_confusion/data/eval_uncompensated_700_300.pt",
    "random_perturb": (
        "outputs/multisnr_semantic_faces/snr20/figures/"
        "confusion_snr20_random_perturb_700_300/data/eval_random_perturb_700_300.pt"
    ),
    "fixed_precomp": (
        "outputs/multisnr_semantic_faces/snr20/figures/"
        "confusion_snr20_fixed_precomp_700_300/data/eval_fixed_precomp_700_300.pt"
    ),
}

BAR_COLORS = ["#ff6b21", "#c9c9c9", "#08b894"]
LINE_COLOR = "#1db7e8"
FONT_FAMILY = "Times New Roman"
AXIS_LABEL_SIZE = 28
TICK_LABEL_SIZE = 22
XTICK_LABEL_SIZE = 20
CONF_AXIS_LABEL_SIZE = 30
CONF_TICK_LABEL_SIZE = 30
CONF_CELL_FONT_SIZE = 33


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_train_helpers():
    path = PROJECT_ROOT / "scripts" / "06_train_eve_eval_snr20.py"
    spec = importlib.util.spec_from_file_location("eve_train_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Eve helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_dataset(path: Path) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input dataset: {path}")
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "signals" not in data or "labels" not in data:
        raise ValueError(f"Dataset must contain 'signals' and 'labels': {path}")
    return data


def _tx_counts(labels: torch.Tensor) -> dict[int, int]:
    return {int(label): int((labels == int(label)).sum().item()) for label in sorted(labels.unique().tolist())}


def _derive_split_counts(train_ratio: float, val_ratio: float, test_ratio: float, test_samples_per_tx: int) -> tuple[int, int, int, int]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum:g}.")
    if test_ratio <= 0.0:
        raise ValueError("test_ratio must be positive.")
    total_per_tx = int(round(test_samples_per_tx / test_ratio))
    train_per_tx = int(round(total_per_tx * train_ratio))
    val_per_tx = total_per_tx - train_per_tx - test_samples_per_tx
    if train_per_tx <= 0 or val_per_tx <= 0 or test_samples_per_tx <= 0:
        raise ValueError("Derived split counts must all be positive.")
    return total_per_tx, train_per_tx, val_per_tx, test_samples_per_tx


def _validate_counts(data: dict[str, torch.Tensor], needed_per_tx: int, num_classes: int, name: str) -> None:
    counts = _tx_counts(data["labels"].long())
    missing = [tx_id + 1 for tx_id in range(num_classes) if counts.get(tx_id, 0) < needed_per_tx]
    if missing:
        raise ValueError(
            f"{name} needs {needed_per_tx} samples per Tx, but these transmitters are short: {missing}. "
            f"Available counts: {counts}"
        )


def _load_psnr(output_root: Path, snr_db: float) -> dict[str, float]:
    snr_tag = str(int(snr_db)) if float(snr_db).is_integer() else str(snr_db).replace(".", "p")
    path = output_root / f"snr{snr_tag}" / "logs" / "semantic_reconstruction_summary.json"
    data = _load_json(path)
    rows = data.get("summary", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected a summary list in {path}")
    return {
        str(row["method"]): float(row.get("semantic_psnr_mean", row["semantic_psnr"]))
        for row in rows
        if isinstance(row, dict)
    }


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion(confusion: list[list[int]], path: Path) -> None:
    ensure_dir(path.parent)
    plt.rcParams["font.family"] = FONT_FAMILY
    matrix = np.asarray(confusion, dtype=np.int64)
    labels = [f"T{i + 1}" for i in range(matrix.shape[0])]

    fig, ax = plt.subplots(figsize=(16.0, 9.0))
    image = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Predicted label", fontsize=CONF_AXIS_LABEL_SIZE, labelpad=10)
    ax.set_ylabel("True label", fontsize=CONF_AXIS_LABEL_SIZE, labelpad=10)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=CONF_TICK_LABEL_SIZE)
    ax.set_yticklabels(labels, fontsize=CONF_TICK_LABEL_SIZE)
    ax.tick_params(length=0)

    threshold = float(matrix.max()) * 0.55 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "#15415e"
            ax.text(
                col,
                row,
                f"{matrix[row, col]:d}",
                ha="center",
                va="center",
                color=color,
                fontsize=CONF_CELL_FONT_SIZE,
                fontweight="bold",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.03)
    cbar.ax.tick_params(labelsize=CONF_TICK_LABEL_SIZE)
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.16, top=0.96)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(path.with_suffix(".pdf"), dpi=220, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(path.with_suffix(".svg"), dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _plot_psnr_accuracy(rows: list[dict[str, float | str]], path: Path) -> None:
    ensure_dir(path.parent)
    plt.rcParams["font.family"] = FONT_FAMILY
    x = np.arange(len(rows))
    accuracy = np.asarray([float(row["identity_accuracy"]) for row in rows], dtype=np.float32)
    psnr = np.asarray([float(row["psnr_db"]) for row in rows], dtype=np.float32)
    labels = [str(row["label"]) for row in rows]

    fig, ax_left = plt.subplots(figsize=(13.0, 9.0))
    ax_right = ax_left.twinx()

    ax_left.bar(x, accuracy, color=BAR_COLORS, edgecolor="white", linewidth=1.4, width=0.72, zorder=2)
    ax_right.plot(x, psnr, color=LINE_COLOR, marker="o", markersize=11.0, linewidth=3.2, zorder=3)

    ax_left.set_ylabel("Identity recognition accuracy", fontsize=AXIS_LABEL_SIZE, labelpad=22)
    ax_right.set_ylabel("Bob PSNR (dB)", color=LINE_COLOR, fontsize=AXIS_LABEL_SIZE, labelpad=22)
    ax_left.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax_right.tick_params(axis="y", colors=LINE_COLOR, labelsize=TICK_LABEL_SIZE)
    ax_right.spines["right"].set_color(LINE_COLOR)

    ax_left.set_xticks(x)
    ax_left.set_xticklabels(labels, fontsize=XTICK_LABEL_SIZE)
    ax_left.set_ylim(0.0, 1.05)
    ax_left.set_yticks(np.linspace(0.0, 1.0, 6))

    psnr_min = float(psnr.min())
    psnr_max = float(psnr.max())
    psnr_pad = max(0.25, (psnr_max - psnr_min) * 3.0)
    ax_right.set_ylim(psnr_min - psnr_pad, psnr_max + psnr_pad)

    ax_left.grid(True, axis="y", alpha=0.25, linewidth=0.8)
    ax_left.set_axisbelow(True)
    fig.subplots_adjust(left=0.24, right=0.78, bottom=0.22, top=0.96)
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(path.with_suffix(".pdf"), dpi=220, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(path.with_suffix(".svg"), dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SNR=20 Eve training with train/val/test=0.6/0.2/0.2 and 100 test samples per transmitter."
    )
    parser.add_argument(
        "--config",
        default="outputs/multisnr_semantic_faces/configs/snr20.yaml",
        help="SNR=20 config used for Eve hyperparameters.",
    )
    parser.add_argument("--multisnr-config", default="configs/multisinr.yaml", help="Config used to locate PSNR summaries.")
    parser.add_argument(
        "--output-dir",
        default="outputs/multisnr_semantic_faces/snr20/figures/eve_ratio_300_100_100",
        help="New directory for checkpoints, logs, and figures.",
    )
    parser.add_argument("--snr", type=float, default=20.0, help="SNR value used for PSNR loading.")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Per-Tx training ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Per-Tx validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Per-Tx testing ratio.")
    parser.add_argument("--test-samples-per-tx", type=int, default=100, help="Final test samples per transmitter.")
    parser.add_argument("--epochs", type=int, default=None, help="Override eve.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eve.batch_size.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit train batches per epoch for a smoke run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    multisnr_config = load_config(args.multisnr_config)
    set_seed(int(config.get("seed", 42)))
    device = get_device(bool(config.get("device", {}).get("prefer_cuda", True)))
    logger = get_logger("rfhide.eve_ratio_300_100_100")

    output_dir = ensure_dir(_resolve_project_path(args.output_dir))
    log_dir = ensure_dir(output_dir / "logs")
    figure_dir = ensure_dir(output_dir / "figures")
    ensure_dir(output_dir / "checkpoints")

    total_per_tx, train_per_tx, val_per_tx, test_per_tx = _derive_split_counts(
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.test_samples_per_tx,
    )
    num_classes = int(config.get("eve", {}).get("num_classes", config.get("impairments", {}).get("num_tx", 6)))
    logger.info(
        "Using per-Tx split total=%d, train=%d, val=%d, test=%d.",
        total_per_tx,
        train_per_tx,
        val_per_tx,
        test_per_tx,
    )

    helpers = _load_train_helpers()
    helper_args = argparse.Namespace(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        train_samples_per_tx=train_per_tx,
        val_samples_per_tx=val_per_tx,
        test_samples_per_tx=test_per_tx,
        label_shuffle_check=False,
    )

    results: dict[str, Any] = {}
    all_metric_rows: list[dict[str, float | str]] = []
    for idx, method in enumerate(METHODS):
        dataset_path = _resolve_project_path(DATASETS[method])
        data = _load_dataset(dataset_path)
        _validate_counts(data, total_per_tx, num_classes, method)
        result_name = f"{method}_300_100_100"
        result, metric_rows = helpers._train_one_class(
            result_name,
            data,
            config,
            device,
            helper_args,
            output_dir,
            seed_offset=100 * idx,
        )
        result["method"] = method
        result["source_dataset"] = str(dataset_path)
        result["train_ratio"] = float(args.train_ratio)
        result["val_ratio"] = float(args.val_ratio)
        result["test_ratio"] = float(args.test_ratio)
        result["total_samples_per_tx"] = int(total_per_tx)
        results[method] = result
        all_metric_rows.extend(metric_rows)

        confusion_path = figure_dir / f"confusion_matrix_snr20_{method}_300_100_100.png"
        _plot_confusion(result["confusion_matrix"], confusion_path)
        logger.info("%s test accuracy: %.4f", method, result["test_accuracy"])

    helpers._write_metrics_csv(log_dir / "eve_300_100_100_train_metrics.csv", all_metric_rows)
    save_json(
        {
            "snr_db": float(args.snr),
            "split": {
                "train_ratio": float(args.train_ratio),
                "val_ratio": float(args.val_ratio),
                "test_ratio": float(args.test_ratio),
                "total_samples_per_tx": int(total_per_tx),
                "train_samples_per_tx": int(train_per_tx),
                "val_samples_per_tx": int(val_per_tx),
                "test_samples_per_tx": int(test_per_tx),
            },
            "results": results,
        },
        log_dir / "eve_300_100_100_results.json",
    )

    multisnr_output_root = _resolve_project_path(multisnr_config.get("experiment", {}).get("output_dir", "outputs/multisnr_semantic_faces"))
    psnr_by_method = _load_psnr(multisnr_output_root, args.snr)
    plot_rows: list[dict[str, float | str]] = []
    for method, label in zip(METHODS, DISPLAY_LABELS, strict=True):
        plot_rows.append(
            {
                "method": method,
                "label": label,
                "identity_accuracy": float(results[method]["test_accuracy"]),
                "psnr_db": float(psnr_by_method[method]),
            }
        )
    _write_csv(plot_rows, log_dir / "snr20_psnr_accuracy_300_100_100.csv")
    figure_path = figure_dir / "snr20_psnr_accuracy_300_100_100.png"
    _plot_psnr_accuracy(plot_rows, figure_path)

    print(f"Per-Tx split: total={total_per_tx}, train={train_per_tx}, val={val_per_tx}, test={test_per_tx}")
    for row in plot_rows:
        print(
            f"{row['label']}: identity_accuracy={float(row['identity_accuracy']):.6f}, "
            f"psnr_db={float(row['psnr_db']):.6f}"
        )
    print(f"Saved results: {log_dir / 'eve_300_100_100_results.json'}")
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    main()
