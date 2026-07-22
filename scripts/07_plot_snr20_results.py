"""Step 9 generate final SNR=20 figures and summary tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.logging_utils import get_logger
from rfhide.models_eve import EveCNN
from rfhide.utils import ensure_dir, get_device, save_json, set_seed

STEP_NAME = "Step 9 plot final SNR20 results"
METHODS = ["uncompensated", "random_perturb", "fixed_precomp"]
METHOD_LABELS = {
    "uncompensated": "Uncompensated",
    "random_perturb": "Random perturb",
    "fixed_precomp": "Fixed precomp",
}
COLORS = {
    "uncompensated": "#4C78A8",
    "random_perturb": "#F58518",
    "fixed_precomp": "#54A24B",
}
TX_COLORS = ["deeppink", "blueviolet", "cyan", "blue", "lime", "yellow"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--max-tsne-samples", type=int, default=900, help="Maximum points used by all-method t-SNE.")
    return parser.parse_args()


def _require_file(path: Path, description: str) -> Path:
    """Return an existing path or raise a clear file error."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required {description}: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON from a required file."""
    _require_file(path, "JSON result file")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_eval_dataset(path: Path) -> dict[str, torch.Tensor]:
    """Load one evaluation signal dataset and validate required fields."""
    _require_file(path, "evaluation signal file")
    data = torch.load(path, map_location="cpu")
    required = ["signals", "labels", "ber", "evm", "residual_power_ratio"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"{path} is missing required fields: {missing}")
    return data


def _build_summary_rows(eve_results: dict[str, Any], datasets: dict[str, dict[str, torch.Tensor]]) -> list[dict[str, Any]]:
    """Build final summary rows for all methods."""
    rows: list[dict[str, Any]] = []
    results = eve_results.get("results", {})
    for method in METHODS:
        if method not in results:
            raise KeyError(f"Eve results JSON is missing method: {method}")
        data = datasets[method]
        row = {
            "method": method,
            "eve_test_acc": float(results[method]["test_accuracy"]),
            "mean_ber": float(data["ber"].float().mean().item()),
            "std_ber": float(data["ber"].float().std(unbiased=False).item()),
            "mean_evm": float(data["evm"].float().mean().item()),
            "std_evm": float(data["evm"].float().std(unbiased=False).item()),
            "mean_residual_power": float(data["residual_power_ratio"].float().mean().item()),
            "std_residual_power": float(data["residual_power_ratio"].float().std(unbiased=False).item()),
        }
        rows.append(row)
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write final summary rows to CSV."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric_bar(rows: list[dict[str, Any]], metric: str, ylabel: str, title: str, output_path: Path) -> None:
    """Save a single metric comparison bar chart."""
    methods = [row["method"] for row in rows]
    values = [float(row[metric]) for row in rows]
    labels = [METHOD_LABELS[method] for method in methods]
    colors = [COLORS[method] for method in methods]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def _extract_features_with_eve(
    method: str,
    signals: torch.Tensor,
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int,
) -> tuple[np.ndarray, str]:
    """Extract Eve embeddings when a checkpoint exists, otherwise flattened IQ."""
    if not checkpoint_path.exists():
        warnings.warn(f"Eve checkpoint missing for {method}; falling back to flattened IQ: {checkpoint_path}", stacklevel=2)
        return signals.flatten(start_dim=1).float().numpy(), "flattened_iq"

    model = EveCNN(num_classes=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    chunks: list[torch.Tensor] = []
    for start in range(0, signals.shape[0], 128):
        batch = signals[start : start + 128].float().to(device)
        chunks.append(model.extract_embedding(batch).cpu())
    return torch.cat(chunks, dim=0).numpy(), "eve_embedding"


def _fit_tsne(features: np.ndarray, seed: int) -> np.ndarray:
    """Run t-SNE with parameters suitable for the current sample count."""
    if features.shape[0] < 3:
        raise ValueError("t-SNE needs at least three samples.")
    perplexity = min(30.0, max(2.0, (features.shape[0] - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
    return tsne.fit_transform(features)


def _write_tsne_csv(path: Path, coords: np.ndarray, labels: np.ndarray, methods: list[str]) -> None:
    """Write t-SNE coordinates to CSV."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "tx_id", "tsne_x", "tsne_y"])
        writer.writeheader()
        for method, tx_id, coord in zip(methods, labels, coords):
            writer.writerow({"method": method, "tx_id": int(tx_id), "tsne_x": float(coord[0]), "tsne_y": float(coord[1])})


def _plot_tsne(
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
    methods: list[str] | None = None,
) -> None:
    """Save a t-SNE scatter plot colored by Tx ID."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    tx_color_map = {tx_id: TX_COLORS[idx % len(TX_COLORS)] for idx, tx_id in enumerate(sorted(np.unique(labels).astype(int).tolist()))}
    for tx_id in sorted(np.unique(labels).astype(int).tolist()):
        mask = labels == tx_id
        marker = "o"
        if methods is not None:
            method_subset = np.array(methods, dtype=object)[mask]
            for method in METHODS:
                method_mask = method_subset == method
                if method_mask.any():
                    indices = np.where(mask)[0][method_mask]
                    marker = {"uncompensated": "o", "random_perturb": "s", "fixed_precomp": "^"}[method]
                    ax.scatter(coords[indices, 0], coords[indices, 1], s=18, alpha=0.72, c=tx_color_map[tx_id], marker=marker)
        else:
            ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.78, c=tx_color_map[tx_id], marker=marker, label=f"Tx {tx_id}")
    if methods is None:
        ax.legend(frameon=True)
    else:
        tx_handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=f"Tx {tx_id}", markerfacecolor=tx_color_map[tx_id], markersize=8)
            for tx_id in sorted(tx_color_map)
        ]
        method_handles = [
            plt.Line2D([0], [0], marker={"uncompensated": "o", "random_perturb": "s", "fixed_precomp": "^"}[method], color="#444444", label=METHOD_LABELS[method], linestyle="None", markersize=7)
            for method in METHODS
        ]
        first_legend = ax.legend(handles=tx_handles, loc="upper right", frameon=True)
        ax.add_artist(first_legend)
        ax.legend(handles=method_handles, loc="lower right", frameon=True)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.18)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _select_even_subset(features: np.ndarray, labels: np.ndarray, methods: list[str], max_samples: int, seed: int) -> np.ndarray:
    """Select a reproducible class-balanced subset for combined t-SNE."""
    count = features.shape[0]
    if count <= max_samples:
        return np.arange(count)
    rng = np.random.default_rng(seed)
    unique_labels = sorted(np.unique(labels).astype(int).tolist())
    per_group = max(1, max_samples // (len(METHODS) * len(unique_labels)))
    chosen: list[np.ndarray] = []
    method_array = np.array(methods, dtype=object)
    for method in METHODS:
        for tx_id in unique_labels:
            group = np.where((method_array == method) & (labels == tx_id))[0]
            take = min(per_group, group.shape[0])
            if take > 0:
                chosen.append(rng.choice(group, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


def main() -> None:
    """Generate final SNR=20 plots and summary files."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = get_device(bool(config.get("device", {}).get("prefer_cuda", True)))
    logger = get_logger("rfhide.plot_snr20")
    logger.info("Current step: %s", STEP_NAME)
    logger.info("Config path: %s", args.config)
    logger.info("Device: %s", device)

    output_dir = PROJECT_ROOT / config.get("experiment", {}).get("output_dir", "outputs/snr20")
    data_dir = output_dir / "data"
    figure_dir = ensure_dir(output_dir / "figures")
    log_dir = ensure_dir(output_dir / "logs")
    checkpoint_dir = output_dir / "checkpoints"
    snr_db = float(config.get("signal", {}).get("snr_db", 20))
    num_classes = int(config.get("eve", {}).get("num_classes", config.get("impairments", {}).get("num_tx", 6)))

    eve_results = _load_json(log_dir / "eve_results_snr20.json")
    datasets = {method: _load_eval_dataset(data_dir / f"eval_{method}.pt") for method in METHODS}

    rows = _build_summary_rows(eve_results, datasets)
    summary_json_path = log_dir / "final_summary_snr20.json"
    summary_csv_path = log_dir / "final_summary_snr20.csv"
    save_json({"snr_db": snr_db, "summary": rows}, summary_json_path)
    _write_summary_csv(summary_csv_path, rows)

    _plot_metric_bar(rows, "eve_test_acc", "Eve test accuracy", "Eve identity accuracy at SNR=20", figure_dir / "accuracy_comparison_snr20.png")
    _plot_metric_bar(rows, "mean_ber", "Mean BER", "BER comparison at SNR=20", figure_dir / "ber_comparison_snr20.png")
    _plot_metric_bar(rows, "mean_evm", "Mean EVM", "EVM comparison at SNR=20", figure_dir / "evm_comparison_snr20.png")

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_methods: list[str] = []
    for method in tqdm(METHODS, desc="Building t-SNE figures"):
        data = datasets[method]
        signals = data["signals"].float()
        labels = data["labels"].long().numpy()
        checkpoint_path = checkpoint_dir / f"eve_{method}_best.pt"
        features, feature_source = _extract_features_with_eve(method, signals, checkpoint_path, device, num_classes)
        coords = _fit_tsne(features, int(config.get("seed", 42)))
        method_names = [method] * labels.shape[0]
        _write_tsne_csv(figure_dir / f"tsne_{method}_snr20.csv", coords, labels, method_names)
        _plot_tsne(coords, labels, f"{METHOD_LABELS[method]} t-SNE at SNR=20 ({feature_source})", figure_dir / f"tsne_{method}_snr20.png")
        all_features.append(features)
        all_labels.append(labels)
        all_methods.extend(method_names)
        logger.info("Generated %s t-SNE using %s", method, feature_source)

    combined_features = np.concatenate(all_features, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    chosen = _select_even_subset(combined_features, combined_labels, all_methods, args.max_tsne_samples, int(config.get("seed", 42)))
    combined_methods = [all_methods[int(index)] for index in chosen]
    combined_coords = _fit_tsne(combined_features[chosen], int(config.get("seed", 42)))
    _write_tsne_csv(figure_dir / "tsne_all_methods_snr20.csv", combined_coords, combined_labels[chosen], combined_methods)
    _plot_tsne(
        combined_coords,
        combined_labels[chosen],
        "All methods t-SNE at SNR=20",
        figure_dir / "tsne_all_methods_snr20.png",
        combined_methods,
    )

    logger.info("Saved summary JSON: %s", summary_json_path)
    logger.info("Saved summary CSV: %s", summary_csv_path)
    logger.info("Saved figures to: %s", figure_dir)
    logger.info("Step 9 plotting passed")


if __name__ == "__main__":
    main()
