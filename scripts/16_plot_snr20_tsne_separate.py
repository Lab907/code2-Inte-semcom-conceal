"""Plot separate SNR=20 t-SNE figures from existing coordinates."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

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
OUTPUT_NAMES = {
    "uncompensated": "origin",
    "random_perturb": "random_perturb",
    "fixed_precomp": "pre_compensation",
}
TX_COLORS = ["deeppink", "blueviolet", "cyan", "blue", "lime", "yellow"]
FONT_FAMILY = "Times New Roman"
AXIS_LABEL_SIZE = 48
TICK_LABEL_SIZE = 36
POINT_SIZE = 80
POINT_ALPHA = 0.9


def _resolve_project_path(path_like: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _snr_tag(snr_db: float) -> str:
    """Return the SNR directory tag used by the pipeline."""
    return str(int(snr_db)) if float(snr_db).is_integer() else str(snr_db).replace(".", "p")


def _load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load existing t-SNE coordinates."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _plot_method(rows: list[dict[str, str]], method: str, path: Path) -> None:
    """Plot one method's t-SNE coordinates without title or legend."""
    ensure_dir(path.parent)
    plt.rcParams["font.family"] = FONT_FAMILY
    method_rows = [row for row in rows if row["method"] == method]
    if not method_rows:
        raise FileNotFoundError(f"No t-SNE rows found for method: {method}")

    labels = np.asarray([int(row["tx_id"]) for row in method_rows], dtype=np.int64)
    x_values = np.asarray([float(row["tsne_x"]) for row in method_rows], dtype=np.float32)
    y_values = np.asarray([float(row["tsne_y"]) for row in method_rows], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    tx_ids = sorted(np.unique(labels).astype(int).tolist())
    color_map = {tx_id: TX_COLORS[idx % len(TX_COLORS)] for idx, tx_id in enumerate(tx_ids)}
    for tx_id in tx_ids:
        mask = labels == tx_id
        ax.scatter(
            x_values[mask],
            y_values[mask],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            c=color_map[tx_id],
            edgecolors="none",
        )

    ax.set_xlabel("t-SNE 1", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("t-SNE 2", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(alpha=0.18, linewidth=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"), dpi=220)
    fig.savefig(path.with_suffix(".svg"), dpi=220)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Plot separate SNR=20 t-SNE figures without title or legend.")
    parser.add_argument("--config", default="configs/multisinr.yaml", help="Multi-SNR YAML config.")
    parser.add_argument("--snr", type=float, default=20.0, help="SNR value to plot.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = _resolve_project_path(config.get("experiment", {}).get("output_dir", "outputs/multisnr"))
    snr_tag = _snr_tag(args.snr)
    figure_root = output_root / f"snr{snr_tag}" / "figures"
    csv_path = figure_root / f"tsne_methods_snr{snr_tag}.csv"
    output_dir = _resolve_project_path(args.output_dir) if args.output_dir else figure_root / f"tsne_snr{snr_tag}_separate"

    rows = _load_rows(csv_path)
    for method in METHODS:
        output_path = output_dir / f"tsne_snr{snr_tag}_{OUTPUT_NAMES[method]}.png"
        _plot_method(rows, method, output_path)
        print(f"Saved {method}: {output_path}")


if __name__ == "__main__":
    main()
