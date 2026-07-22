"""Plot separate SNR=30 confusion matrices from existing multi-SNR results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.utils import ensure_dir

METHODS = ["uncompensated", "random_perturb", "fixed_precomp"]


def _resolve_project_path(path_like: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_snr_rows(results_path: Path, snr_db: float) -> list[dict[str, Any]]:
    """Load rows for one SNR from the multi-SNR result summary."""
    data = _load_json(results_path)
    rows = data.get("results", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected a results list in {results_path}")
    selected = [row for row in rows if isinstance(row, dict) and float(row.get("snr_db", -9999.0)) == float(snr_db)]
    if not selected:
        raise FileNotFoundError(f"No SNR={snr_db:g} rows found in {results_path}")
    return selected


def _plot_matrix(matrix: np.ndarray, path: Path, vmax: float) -> None:
    """Plot one confusion matrix without a title."""
    ensure_dir(path.parent)
    num_classes = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=vmax)
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
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Plot separate SNR=30 confusion matrices without titles.")
    parser.add_argument("--config", default="configs/multisinr.yaml", help="Multi-SNR YAML config.")
    parser.add_argument("--snr", type=float, default=30.0, help="SNR value to plot.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = _resolve_project_path(config.get("experiment", {}).get("output_dir", "outputs/multisnr"))
    results_path = output_root / "logs" / "multisnr_results.json"
    figure_dir = _resolve_project_path(args.output_dir) if args.output_dir else output_root / "figures" / f"confusion_snr{int(args.snr)}"

    rows = _load_snr_rows(results_path, args.snr)
    row_lookup = {row["method"]: row for row in rows}
    matrices = [
        np.asarray(row_lookup[method]["confusion_matrix"], dtype=np.float32)
        for method in METHODS
        if method in row_lookup
    ]
    if not matrices:
        raise FileNotFoundError(f"No configured method rows found for SNR={args.snr:g}")
    vmax = max(max(float(matrix.max(initial=0.0)) for matrix in matrices), 1.0)

    for method in METHODS:
        if method not in row_lookup:
            continue
        matrix = np.asarray(row_lookup[method]["confusion_matrix"], dtype=np.float32)
        output_path = figure_dir / f"confusion_matrix_snr{int(args.snr)}_{method}.png"
        _plot_matrix(matrix, output_path, vmax)
        print(f"Saved {method}: {output_path}")


if __name__ == "__main__":
    main()
