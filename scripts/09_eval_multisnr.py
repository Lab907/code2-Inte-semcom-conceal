"""Step 11 evaluate repeated SNR20-style runs and summarize across SNRs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.logging_utils import get_logger
from rfhide.models_eve import EveCNN
from rfhide.utils import ensure_dir, get_device, save_json, set_seed

STEP_NAME = "Step 11 evaluate repeated single-SNR pipelines"
EVAL_SCRIPTS = [
    "05_collect_eval_signals_snr20.py",
    "06_train_eve_eval_snr20.py",
]
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
TX_COLORS = ["deeppink", "blueviolet", "cyan", "blue", "lime", "yellow"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--config", required=True, help="Path to the multi-SNR YAML config.")
    parser.add_argument(
        "--base-config",
        default=None,
        help="Single-SNR template config. Defaults to base_config in the multi-SNR config, then configs/snr20.yaml.",
    )
    parser.add_argument("--quick", action="store_true", help="Use quick multi-SNR eval overrides.")
    parser.add_argument("--eval-batches", type=int, default=None, help="Override eval batches per SNR.")
    parser.add_argument("--eve-epochs", type=int, default=None, help="Override Eve epochs per SNR/method.")
    parser.add_argument("--max-tsne-samples", type=int, default=900, help="Maximum t-SNE points per method/SNR.")
    return parser.parse_args()


def _snr_tag(snr_db: float) -> str:
    """Return a filesystem-safe SNR tag."""
    return str(int(snr_db)) if float(snr_db).is_integer() else str(snr_db).replace(".", "p")


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _snr_list(config: dict[str, Any]) -> list[float]:
    """Read SNR levels from the multi-SNR config."""
    values = config.get("signal", {}).get("snr_list")
    if not values:
        raise ValueError("Multi-SNR config must define signal.snr_list.")
    return [float(value) for value in values]


def _semantic_enabled(config: dict[str, Any]) -> bool:
    """Return whether a config uses semantic JSCC."""
    semantic_cfg = config.get("semantic", {})
    signal_cfg = config.get("signal", {})
    return bool(semantic_cfg.get("enabled", False)) or signal_cfg.get("modulation") == "semantic_jscc"


def _single_snr_config(template: dict[str, Any], snr_db: float, output_root: Path) -> dict[str, Any]:
    """Create the same single-SNR config used by the multi-SNR trainer."""
    cfg = deepcopy(template)
    tag = _snr_tag(snr_db)
    output_dir = output_root / f"snr{tag}"

    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = f"{cfg['experiment'].get('name', 'single_snr')}_snr{tag}"
    cfg["experiment"]["output_dir"] = str(output_dir.relative_to(PROJECT_ROOT))

    cfg.setdefault("signal", {})
    cfg["signal"]["snr_db"] = float(snr_db)
    cfg["signal"].pop("snr_list", None)

    if _semantic_enabled(cfg):
        cfg.setdefault("semantic", {})
        cfg["semantic"]["checkpoint"] = str((output_dir / "checkpoints" / "semantic_jscc.pt").relative_to(PROJECT_ROOT))

    cfg.setdefault("compensation_dataset", {})
    cfg["compensation_dataset"]["checkpoint"] = str((output_dir / "checkpoints" / "teacher_best.pt").relative_to(PROJECT_ROOT))

    cfg.setdefault("eval_signal_collection", {})
    cfg["eval_signal_collection"]["checkpoint"] = str((output_dir / "checkpoints" / "teacher_best.pt").relative_to(PROJECT_ROOT))
    return cfg


def _write_single_snr_configs(config: dict[str, Any], template: dict[str, Any]) -> list[dict[str, Any]]:
    """Write or refresh derived per-SNR configs."""
    output_root = _resolve_project_path(config.get("experiment", {}).get("output_dir", "outputs/multisnr"))
    config_dir = ensure_dir(output_root / "configs")
    entries: list[dict[str, Any]] = []
    for snr_db in _snr_list(config):
        tag = _snr_tag(snr_db)
        snr_cfg = _single_snr_config(template, snr_db, output_root)
        eval_cfg = config.get("multisnr_eval", {})
        if eval_cfg:
            snr_cfg.setdefault("eval_signal_collection", {})
            snr_cfg["eval_signal_collection"]["batch_size"] = int(eval_cfg.get("batch_size", snr_cfg["eval_signal_collection"].get("batch_size", 32)))
            snr_cfg["eval_signal_collection"]["num_batches"] = int(eval_cfg.get("num_batches_per_snr", snr_cfg["eval_signal_collection"].get("num_batches", 4)))
            snr_cfg.setdefault("eve", {})
            for source_key, target_key in [
                ("train_samples_per_tx", "train_samples_per_tx"),
                ("val_samples_per_tx", "val_samples_per_tx"),
                ("test_samples_per_tx", "test_samples_per_tx"),
            ]:
                if source_key in eval_cfg:
                    snr_cfg["eve"][target_key] = int(eval_cfg[source_key])
        path = config_dir / f"snr{tag}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(snr_cfg, handle, sort_keys=False, allow_unicode=True)
        entries.append(
            {
                "snr_db": snr_db,
                "tag": tag,
                "config_path": path,
                "config": snr_cfg,
                "output_dir": _resolve_project_path(snr_cfg["experiment"]["output_dir"]),
            }
        )
    return entries


def _run_command(command: list[str], logger: Any) -> None:
    """Run one child Python step and fail fast if it fails."""
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _run_eval_pipeline(entry: dict[str, Any], multi_config: dict[str, Any], args: argparse.Namespace, logger: Any) -> None:
    """Run eval-signal collection and Eve training for one SNR."""
    eval_cfg = multi_config.get("multisnr_eval", {})
    config_arg = str(entry["config_path"])
    for script in EVAL_SCRIPTS:
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / script), "--config", config_arg]
        if script == "05_collect_eval_signals_snr20.py":
            batches = args.eval_batches
            if batches is None and args.quick:
                batches = int(eval_cfg.get("quick_batches_per_snr", 1))
            if batches is None:
                batches = int(eval_cfg.get("num_batches_per_snr", 4))
            if batches is not None:
                command += ["--num-batches", str(batches)]
            if "batch_size" in eval_cfg:
                command += ["--batch-size", str(int(eval_cfg["batch_size"]))]
        if script == "06_train_eve_eval_snr20.py":
            epochs = args.eve_epochs
            if epochs is None and args.quick:
                epochs = int(eval_cfg.get("quick_eve_epochs", 2))
            if epochs is None:
                epochs = int(eval_cfg.get("eve_epochs", 8))
            if epochs is not None:
                command += ["--epochs", str(epochs)]
            if "eve_batch_size" in eval_cfg:
                command += ["--batch-size", str(int(eval_cfg["eve_batch_size"]))]
        _run_command(command, logger)


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_eval_dataset(path: Path) -> dict[str, torch.Tensor]:
    """Load one SNR20-format evaluation dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation dataset: {path}")
    return torch.load(path, map_location="cpu")


def _summary_rows_for_snr(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Read SNR20-format outputs and return multi-SNR summary rows."""
    output_dir = entry["output_dir"]
    eve_results = _load_json(output_dir / "logs" / "eve_results_snr20.json").get("results", {})
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        data = _load_eval_dataset(output_dir / "data" / f"eval_{method}.pt")
        result = eve_results[method]
        rows.append(
            {
                "snr_db": float(entry["snr_db"]),
                "method": method,
                "identity_accuracy": float(result["test_accuracy"]),
                "best_val_accuracy": float(result["best_val_accuracy"]),
                "mean_ber": float(data["ber"].float().mean().item()),
                "std_ber": float(data["ber"].float().std(unbiased=False).item()),
                "mean_evm": float(data["evm"].float().mean().item()),
                "std_evm": float(data["evm"].float().std(unbiased=False).item()),
                "mean_residual_power": float(data["residual_power_ratio"].float().mean().item()),
                "std_residual_power": float(data["residual_power_ratio"].float().std(unbiased=False).item()),
                "num_samples": int(data["signals"].shape[0]),
                "eve_checkpoint": result["checkpoint"],
                "confusion_matrix": result["confusion_matrix"],
                "train_samples_per_tx": result.get("train_samples_per_tx"),
                "val_samples_per_tx": result.get("val_samples_per_tx"),
                "test_samples_per_tx": result.get("test_samples_per_tx"),
            }
        )
    return rows


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flattened summary rows."""
    ensure_dir(path.parent)
    fieldnames = [
        "snr_db",
        "method",
        "identity_accuracy",
        "best_val_accuracy",
        "mean_ber",
        "std_ber",
        "mean_evm",
        "std_evm",
        "mean_residual_power",
        "std_residual_power",
        "num_samples",
        "eve_checkpoint",
        "train_samples_per_tx",
        "val_samples_per_tx",
        "test_samples_per_tx",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _plot_metric(rows: list[dict[str, Any]], metric: str, ylabel: str, title: str, path: Path) -> None:
    """Plot one metric versus SNR with one curve per method."""
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for method in METHODS:
        method_rows = sorted([row for row in rows if row["method"] == method], key=lambda item: item["snr_db"])
        ax.plot(
            [row["snr_db"] for row in method_rows],
            [row[metric] for row in method_rows],
            marker="o",
            linewidth=2.0,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    ax.set_title(title)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_confusion_panel(
    rows: list[dict[str, Any]],
    title: str,
    path: Path,
    num_classes: int,
) -> None:
    """Plot confusion matrices for a set of SNR/method rows."""
    ensure_dir(path.parent)
    snr_values = sorted({float(row["snr_db"]) for row in rows})
    fig, axes = plt.subplots(
        len(snr_values),
        len(METHODS),
        figsize=(4.2 * len(METHODS), 3.8 * len(snr_values)),
        squeeze=False,
    )
    max_count = max(
        float(np.asarray(row["confusion_matrix"], dtype=np.float32).max(initial=0.0))
        for row in rows
    )
    vmax = max(max_count, 1.0)
    row_lookup = {(float(row["snr_db"]), row["method"]): row for row in rows}

    for row_idx, snr_db in enumerate(snr_values):
        for method_idx, method in enumerate(METHODS):
            ax = axes[row_idx, method_idx]
            item = row_lookup.get((snr_db, method))
            if item is None:
                ax.axis("off")
                continue
            matrix = np.asarray(item["confusion_matrix"], dtype=np.float32)
            im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=vmax)
            ax.set_title(f"{METHOD_LABELS[method]} | SNR={snr_db:g} dB")
            ax.set_xlabel("Predicted Tx")
            ax.set_ylabel("True Tx")
            ax.set_xticks(range(num_classes))
            ax.set_yticks(range(num_classes))
            ax.set_xticklabels([str(idx) for idx in range(num_classes)])
            ax.set_yticklabels([str(idx) for idx in range(num_classes)])
            threshold = vmax * 0.55
            for true_idx in range(matrix.shape[0]):
                for pred_idx in range(matrix.shape[1]):
                    value = int(matrix[true_idx, pred_idx])
                    ax.text(
                        pred_idx,
                        true_idx,
                        str(value),
                        ha="center",
                        va="center",
                        color="white" if value > threshold else "#222222",
                        fontsize=8,
                    )
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.02, 0.96, 0.98))
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _select_balanced_subset(labels: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    """Select a class-balanced subset for t-SNE."""
    if labels.shape[0] <= max_samples:
        return np.arange(labels.shape[0])
    unique_labels = sorted(np.unique(labels).astype(int).tolist())
    per_label = max(1, max_samples // len(unique_labels))
    rng = np.random.default_rng(seed)
    chosen: list[np.ndarray] = []
    for label in unique_labels:
        group = np.where(labels == label)[0]
        take = min(per_label, group.shape[0])
        if take > 0:
            chosen.append(rng.choice(group, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


@torch.no_grad()
def _extract_features(
    signals: torch.Tensor,
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int,
) -> tuple[np.ndarray, str]:
    """Extract Eve embeddings, falling back to flattened IQ if needed."""
    if not checkpoint_path.exists():
        warnings.warn(f"Eve checkpoint missing; falling back to flattened IQ: {checkpoint_path}", stacklevel=2)
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
    """Fit a 2D t-SNE embedding."""
    if features.shape[0] < 3:
        raise ValueError("t-SNE needs at least three samples.")
    perplexity = min(30.0, max(2.0, (features.shape[0] - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=seed)
    return tsne.fit_transform(features)


def _plot_snr_tsne(entry: dict[str, Any], device: torch.device, max_samples: int, seed: int) -> Path:
    """Save one horizontal three-panel t-SNE figure for a single SNR."""
    output_dir = entry["output_dir"]
    figure_dir = ensure_dir(output_dir / "figures")
    csv_path = figure_dir / f"tsne_methods_snr{entry['tag']}.csv"
    png_path = figure_dir / f"tsne_methods_snr{entry['tag']}.png"
    num_classes = int(entry["config"].get("eve", {}).get("num_classes", entry["config"].get("impairments", {}).get("num_tx", 6)))

    fig, axes = plt.subplots(1, len(METHODS), figsize=(15.5, 4.8), squeeze=False)
    axes_flat = axes[0]
    rows: list[dict[str, Any]] = []
    feature_sources: list[str] = []

    for method_idx, method in enumerate(METHODS):
        data = _load_eval_dataset(output_dir / "data" / f"eval_{method}.pt")
        signals = data["signals"].float()
        labels = data["labels"].long().numpy()
        selected = _select_balanced_subset(labels, max_samples, seed + method_idx)
        checkpoint_path = output_dir / "checkpoints" / f"eve_{method}_best.pt"
        features, feature_source = _extract_features(signals[selected], checkpoint_path, device, num_classes)
        feature_sources.append(feature_source)
        coords = _fit_tsne(features, seed + method_idx)
        selected_labels = labels[selected]

        ax = axes_flat[method_idx]
        tx_color_map = {
            tx_id: TX_COLORS[idx % len(TX_COLORS)]
            for idx, tx_id in enumerate(sorted(np.unique(selected_labels).astype(int).tolist()))
        }
        for tx_id in sorted(tx_color_map):
            mask = selected_labels == tx_id
            ax.scatter(coords[mask, 0], coords[mask, 1], s=16, alpha=0.78, c=tx_color_map[tx_id], label=f"Tx {tx_id}")
        ax.set_title(METHOD_LABELS[method])
        ax.set_xlabel("t-SNE 1")
        if method_idx == 0:
            ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.18)

        for tx_id, coord in zip(selected_labels, coords):
            rows.append(
                {
                    "method": method,
                    "tx_id": int(tx_id),
                    "tsne_x": float(coord[0]),
                    "tsne_y": float(coord[1]),
                    "feature_source": feature_source,
                }
            )

    handles, labels = axes_flat[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(num_classes, 8), frameon=True)
    fig.suptitle(f"t-SNE by method at SNR={entry['snr_db']:g} dB ({', '.join(sorted(set(feature_sources)))})")
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "tx_id", "tsne_x", "tsne_y", "feature_source"])
        writer.writeheader()
        writer.writerows(rows)
    return png_path


def main() -> None:
    """Evaluate every independent single-SNR run and save aggregate figures."""
    args = parse_args()
    multi_config = load_config(args.config)
    base_config = args.base_config or multi_config.get("base_config", "configs/snr20.yaml")
    template = load_config(base_config)
    set_seed(int(template.get("seed", multi_config.get("seed", 42))))
    device = get_device(bool(template.get("device", {}).get("prefer_cuda", True)))
    logger = get_logger("rfhide.multisnr_eval")
    logger.info("Current step: %s", STEP_NAME)
    logger.info("Multi-SNR config: %s", args.config)
    logger.info("Base single-SNR template: %s", base_config)
    logger.info("Device: %s", device)

    output_root = _resolve_project_path(multi_config.get("experiment", {}).get("output_dir", "outputs/multisnr"))
    ensure_dir(output_root / "logs")
    ensure_dir(output_root / "figures")
    entries = _write_single_snr_configs(multi_config, template)

    rows: list[dict[str, Any]] = []
    tsne_paths: list[str] = []
    num_classes = int(template.get("eve", {}).get("num_classes", template.get("impairments", {}).get("num_tx", 6)))
    for snr_idx, entry in enumerate(entries):
        logger.info("Starting SNR=%s dB eval | output: %s", entry["snr_db"], entry["output_dir"])
        _run_eval_pipeline(entry, multi_config, args, logger)
        snr_rows = _summary_rows_for_snr(entry)
        rows.extend(snr_rows)
        tsne_path = _plot_snr_tsne(
            entry,
            device,
            max_samples=args.max_tsne_samples,
            seed=int(template.get("seed", 42)) + 1000 * snr_idx,
        )
        tsne_paths.append(str(tsne_path))
        logger.info("Saved SNR=%s dB three-panel t-SNE: %s", entry["snr_db"], tsne_path)

    results_csv = output_root / "logs" / "multisnr_results.csv"
    results_json = output_root / "logs" / "multisnr_results.json"
    _write_results_csv(results_csv, rows)
    save_json(
        {
            "mode": "repeat_snr20_pipeline_per_snr",
            "base_config": str(_resolve_project_path(base_config)),
            "snr_list": [entry["snr_db"] for entry in entries],
            "quick": bool(args.quick),
            "results": rows,
            "tsne_figures": tsne_paths,
        },
        results_json,
    )

    _plot_metric(rows, "identity_accuracy", "Eve identity accuracy", "Identity accuracy vs SNR", output_root / "figures" / "accuracy_vs_snr.png")
    _plot_metric(rows, "mean_ber", "Mean BER", "BER vs SNR", output_root / "figures" / "ber_vs_snr.png")
    _plot_metric(rows, "mean_evm", "Mean EVM", "EVM vs SNR", output_root / "figures" / "evm_vs_snr.png")
    _plot_metric(rows, "mean_residual_power", "Mean residual power ratio", "Residual power vs SNR", output_root / "figures" / "residual_power_vs_snr.png")
    _plot_confusion_panel(
        rows,
        "Eve confusion matrices across SNRs",
        output_root / "figures" / "confusion_matrices_by_snr.png",
        num_classes,
    )
    for snr_db in sorted({float(row["snr_db"]) for row in rows}):
        snr_rows = [row for row in rows if float(row["snr_db"]) == snr_db]
        _plot_confusion_panel(
            snr_rows,
            f"Eve confusion matrices at SNR={snr_db:g} dB",
            output_root / "figures" / f"confusion_matrices_snr{_snr_tag(snr_db)}.png",
            num_classes,
        )
    logger.info("Saved results CSV: %s", results_csv)
    logger.info("Saved results JSON: %s", results_json)
    logger.info("Saved aggregate figures to: %s", output_root / "figures")
    logger.info("Step 11 repeated single-SNR evaluation passed")


if __name__ == "__main__":
    main()
