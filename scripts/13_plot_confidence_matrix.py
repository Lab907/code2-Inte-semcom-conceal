"""Plot a trust confidence matrix from existing multi-SNR results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.utils import ensure_dir, save_json

FONT_FAMILY = "Times New Roman"
TITLE_FONT_SIZE = 44
AXIS_LABEL_SIZE = 48
TICK_LABEL_SIZE = 36
REGION_FONT_SIZE = 27
COLORBAR_LABEL_SIZE = 40
COLORBAR_TICK_SIZE = 32
SNR_CMAP = "YlGnBu_r"


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


def _snr_tag(snr_db: float) -> str:
    """Convert an SNR value to the directory tag used by the pipeline."""
    return str(int(snr_db)) if float(snr_db).is_integer() else str(snr_db).replace(".", "p")


def _load_identity_accuracy(output_root: Path, method: str) -> dict[float, float]:
    """Read Eve identity accuracy for one method from the multi-SNR result summary."""
    results_path = output_root / "logs" / "multisnr_results.json"
    data = _load_json(results_path)
    rows = data.get("results", [])
    if not isinstance(rows, list):
        raise ValueError(f"Expected a results list in {results_path}")

    values: dict[float, float] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("method") != method:
            continue
        values[float(row["snr_db"])] = float(row["identity_accuracy"])
    return values


def _load_semantic_confidence(output_root: Path, method: str) -> dict[float, float]:
    """Read face-feature cosine values for one method from existing semantic summaries."""
    values: dict[float, float] = {}
    for log_path in sorted(output_root.glob("snr*/logs/semantic_reconstruction_summary.json")):
        snr_name = log_path.parents[1].name
        if not snr_name.startswith("snr"):
            continue
        snr_db = float(snr_name[3:].replace("p", "."))
        data = _load_json(log_path)
        summary = data.get("summary", [])
        if not isinstance(summary, list):
            raise ValueError(f"Expected a summary list in {log_path}")
        for row in summary:
            if isinstance(row, dict) and row.get("method") == method:
                values[snr_db] = float(row["face_feature_cosine"])
                break
    return values


def _collect_rows(output_root: Path, snr_list: list[float], method: str) -> list[dict[str, float]]:
    """Join identity and semantic metrics by SNR."""
    identity_by_snr = _load_identity_accuracy(output_root, method)
    semantic_by_snr = _load_semantic_confidence(output_root, method)

    rows: list[dict[str, float]] = []
    missing: list[str] = []
    for snr_db in snr_list:
        if snr_db not in identity_by_snr:
            missing.append(f"SNR {snr_db:g} identity accuracy")
            continue
        if snr_db not in semantic_by_snr:
            missing.append(f"SNR {snr_db:g} semantic confidence")
            continue
        rows.append(
            {
                "snr_db": float(snr_db),
                "semantic_confidence": float(semantic_by_snr[snr_db]),
                "identity_confidence": float(identity_by_snr[snr_db]),
            }
        )
    if missing:
        raise FileNotFoundError("Missing existing metrics: " + ", ".join(missing))
    return rows


def _write_csv(rows: list[dict[str, float]], path: Path) -> None:
    """Save plotted coordinates as CSV."""
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "snr_db",
                "semantic_confidence",
                "identity_confidence",
                "face_feature_cosine",
                "identity_accuracy",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "snr_db": row["snr_db"],
                    "semantic_confidence": row["semantic_confidence"],
                    "identity_confidence": row["identity_confidence"],
                    "face_feature_cosine": row["semantic_confidence"],
                    "identity_accuracy": row["identity_confidence"],
                }
            )


def _plot_confidence_matrix(rows: list[dict[str, float]], threshold: float, path: Path) -> None:
    """Draw the four-region confidence matrix."""
    ensure_dir(path.parent)
    plt.rcParams["font.family"] = FONT_FAMILY
    fig, ax = plt.subplots(figsize=(13.0, 9.0))

    ax.add_patch(Rectangle((0.0, threshold), threshold, 1.0 - threshold, facecolor="#d8e3fb", alpha=0.88, zorder=0))
    ax.add_patch(Rectangle((threshold, threshold), 1.0 - threshold, 1.0 - threshold, facecolor="#cdeed7", alpha=0.88, zorder=0))
    ax.add_patch(Rectangle((0.0, 0.0), threshold, threshold, facecolor="#eeeeee", alpha=0.9, zorder=0))
    ax.add_patch(Rectangle((threshold, 0.0), 1.0 - threshold, threshold, facecolor="#f7ddd2", alpha=0.9, zorder=0))

    ax.axhline(threshold, color="#888888", linewidth=1.8)
    ax.axvline(threshold, color="#888888", linewidth=1.8)

    x_values = [row["semantic_confidence"] for row in rows]
    y_values = [row["identity_confidence"] for row in rows]
    snr_values = [row["snr_db"] for row in rows]
    scatter = ax.scatter(
        x_values,
        y_values,
        c=snr_values,
        cmap=SNR_CMAP,
        s=118,
        edgecolors="#1f2933",
        linewidths=1.0,
        zorder=3,
    )

    ax.text(
        threshold * 0.5,
        threshold + (1.0 - threshold) * 0.5,
        "Re-transmit",
        ha="center",
        va="center",
        color="#4774b8",
        fontsize=REGION_FONT_SIZE,
        fontweight="bold",
    )
    ax.text(
        threshold + (1.0 - threshold) * 0.5,
        threshold + (1.0 - threshold) * 0.18,
        "Accept",
        ha="center",
        va="center",
        color="#178a45",
        fontsize=REGION_FONT_SIZE,
        fontweight="bold",
    )
    ax.text(
        threshold * 0.5,
        threshold * 0.42,
        "Reject",
        ha="center",
        va="center",
        color="#777777",
        fontsize=REGION_FONT_SIZE,
        fontweight="bold",
    )
    ax.text(
        threshold + (1.0 - threshold) * 0.5,
        threshold * 0.5,
        "Re-authenticate",
        ha="center",
        va="center",
        color="#c34b37",
        fontsize=REGION_FONT_SIZE,
        fontweight="bold",
        rotation=90,
    )

    ax.set_title("Trust decision matrix", fontsize=TITLE_FONT_SIZE, fontweight="bold", pad=12)
    ax.set_xlabel("Semantic confidence", fontsize=AXIS_LABEL_SIZE, labelpad=22)
    ax.set_ylabel("Identity confidence", fontsize=AXIS_LABEL_SIZE, labelpad=22)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.035)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, threshold, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, threshold, 1.0])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.grid(True, alpha=0.18, linewidth=0.8)

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("SNR (dB)", fontsize=COLORBAR_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"), dpi=220)
    fig.savefig(path.with_suffix(".svg"), dpi=220)
    plt.close(fig)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Plot a confidence matrix from existing multi-SNR results.")
    parser.add_argument("--config", default="configs/multisinr.yaml", help="Multi-SNR YAML config.")
    parser.add_argument("--method", default="uncompensated", help="Method to plot; defaults to uncompensated.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Decision threshold for both axes.")
    parser.add_argument("--output", default=None, help="Optional PNG output path.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = _resolve_project_path(config.get("experiment", {}).get("output_dir", "outputs/multisnr"))
    snr_list = [float(value) for value in config.get("signal", {}).get("snr_list", [0, 5, 10, 15, 20, 25, 30])]

    rows = _collect_rows(output_root, snr_list, args.method)
    figure_path = _resolve_project_path(args.output) if args.output else output_root / "figures" / f"confidence_matrix_{args.method}.png"
    csv_path = output_root / "logs" / f"confidence_matrix_{args.method}.csv"
    json_path = output_root / "logs" / f"confidence_matrix_{args.method}.json"

    _plot_confidence_matrix(rows, args.threshold, figure_path)
    _write_csv(rows, csv_path)
    save_json(
        {
            "method": args.method,
            "threshold": float(args.threshold),
            "x_axis": "semantic_confidence",
            "y_axis": "identity_confidence",
            "semantic_confidence_source": "face_feature_cosine",
            "identity_confidence_source": "identity_accuracy",
            "rows": rows,
            "figure": str(figure_path),
            "csv": str(csv_path),
        },
        json_path,
    )
    print(f"Saved confidence matrix: {figure_path}")
    print(f"Saved plotted data: {csv_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
