"""Plot the multi-SNR fixed-precomp trust decision matrix from existing results."""

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

from rfhide.utils import ensure_dir, save_json

FONT_FAMILY = "Times New Roman"
TITLE_FONT_SIZE = 30
AXIS_LABEL_SIZE = 32
TICK_LABEL_SIZE = 23
REGION_FONT_SIZE = 28
COLORBAR_LABEL_SIZE = 26
COLORBAR_TICK_SIZE = 21
SNR_CMAP = "YlGnBu_r"


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_semantic_similarity(summary_path: Path, method: str) -> float:
    data = _load_json(summary_path)
    summary = data.get("summary", [])
    if not isinstance(summary, list):
        raise ValueError(f"Expected a summary list in {summary_path}")

    for row in summary:
        if isinstance(row, dict) and row.get("method") == method:
            return float(row["face_feature_cosine"])
    raise KeyError(f"Method {method!r} was not found in {summary_path}")


def _load_semantic_similarity_by_snr(output_root: Path, snr_list: list[float], method: str) -> dict[float, float]:
    values: dict[float, float] = {}
    for snr_db in snr_list:
        snr_tag = _snr_tag(snr_db)
        summary_path = output_root / snr_tag / "logs" / "semantic_reconstruction_summary.json"
        values[float(snr_db)] = _load_semantic_similarity(summary_path, method)
    return values


def _load_authentication_accuracy_by_snr(results_path: Path, method: str) -> tuple[list[float], dict[float, float]]:
    data = _load_json(results_path)
    results = data.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Expected a results list in {results_path}")

    values: dict[float, float] = {}
    for row in results:
        if isinstance(row, dict) and row.get("method") == method:
            values[float(row["snr_db"])] = float(row["identity_accuracy"])

    snr_list = data.get("snr_list")
    if not isinstance(snr_list, list):
        snr_list = sorted(values)
    return [float(snr_db) for snr_db in snr_list], values


def _snr_tag(snr_db: float) -> str:
    value = float(snr_db)
    return f"snr{int(value)}" if value.is_integer() else f"snr{str(value).replace('.', 'p')}"


def _collect_rows(output_root: Path, results_path: Path, method: str) -> list[dict[str, float]]:
    snr_list, auth_by_snr = _load_authentication_accuracy_by_snr(results_path, method)
    semantic_by_snr = _load_semantic_similarity_by_snr(output_root, snr_list, method)

    rows: list[dict[str, float]] = []
    missing: list[str] = []
    for snr_db in snr_list:
        if snr_db < 0.0 or snr_db > 30.0:
            continue
        if snr_db not in auth_by_snr:
            missing.append(f"SNR {snr_db:g} authentication accuracy")
            continue
        rows.append(
            {
                "snr_db": float(snr_db),
                "semantic_similarity": float(semantic_by_snr[snr_db]),
                "authentication_accuracy": float(auth_by_snr[snr_db]),
                "method": method,
            }
        )
    if missing:
        raise FileNotFoundError("Missing existing metrics: " + ", ".join(missing))
    return rows


def _write_csv(rows: list[dict[str, float]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "snr_db",
                "semantic_similarity",
                "authentication_accuracy",
                "method",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_trust_matrix(rows: list[dict[str, float]], threshold: float, path: Path) -> None:
    ensure_dir(path.parent)
    plt.rcParams["font.family"] = FONT_FAMILY

    fig, ax = plt.subplots(figsize=(12.0, 9.0))

    ax.add_patch(Rectangle((0.0, threshold), threshold, 1.0 - threshold, facecolor="#d8e3fb", alpha=0.88, zorder=0))
    ax.add_patch(Rectangle((threshold, threshold), 1.0 - threshold, 1.0 - threshold, facecolor="#cdeed7", alpha=0.88, zorder=0))
    ax.add_patch(Rectangle((0.0, 0.0), threshold, threshold, facecolor="#eeeeee", alpha=0.9, zorder=0))
    ax.add_patch(Rectangle((threshold, 0.0), 1.0 - threshold, threshold, facecolor="#f7ddd2", alpha=0.9, zorder=0))

    ax.axhline(threshold, color="#888888", linewidth=1.8)
    ax.axvline(threshold, color="#888888", linewidth=1.8)

    scatter = ax.scatter(
        [row["semantic_similarity"] for row in rows],
        [row["authentication_accuracy"] for row in rows],
        c=[row["snr_db"] for row in rows],
        cmap=SNR_CMAP,
        vmin=0.0,
        vmax=30.0,
        s=170,
        edgecolors="#1f2933",
        linewidths=1.2,
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
        threshold + (1.0 - threshold) * 0.5,
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
        threshold + (1.0 - threshold) * 0.84,
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
    ax.set_xlabel("Semantic confidence", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Identity confidence", fontsize=AXIS_LABEL_SIZE)
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
    parser = argparse.ArgumentParser(
        description="Plot the multi-SNR fixed-precomp trust decision matrix from existing experiment outputs."
    )
    parser.add_argument("--method", default="fixed_precomp", help="Semantic reconstruction method to read.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Decision threshold for both axes.")
    parser.add_argument(
        "--output-root",
        default="outputs/multisnr_semantic_faces",
        help="Existing multi-SNR output root.",
    )
    parser.add_argument(
        "--multisnr-results",
        default="outputs/multisnr_semantic_faces/logs/multisnr_results.json",
        help="Existing multi-SNR result summary JSON.",
    )
    parser.add_argument(
        "--output",
        default="outputs/multisnr_semantic_faces/figures/trust_matrix_fixed_precomp_multisnr.png",
        help="PNG output path.",
    )
    args = parser.parse_args()

    output_root = _resolve_project_path(args.output_root)
    multisnr_results_path = _resolve_project_path(args.multisnr_results)
    figure_path = _resolve_project_path(args.output)
    csv_path = figure_path.with_suffix(".csv")
    json_path = figure_path.with_suffix(".json")

    rows = _collect_rows(output_root, multisnr_results_path, args.method)

    _plot_trust_matrix(rows, float(args.threshold), figure_path)
    _write_csv(rows, csv_path)
    save_json(
        {
            "threshold": float(args.threshold),
            "x_axis": "semantic_similarity",
            "y_axis": "authentication_accuracy",
            "semantic_similarity_source": str(output_root / "snr*/logs/semantic_reconstruction_summary.json"),
            "authentication_accuracy_source": str(multisnr_results_path),
            "points": rows,
            "figure": str(figure_path),
            "csv": str(csv_path),
        },
        json_path,
    )

    print("SNR | Semantic similarity | Authentication accuracy")
    for row in rows:
        print(f"{row['snr_db']:>3.0f} | {row['semantic_similarity']:.6f} | {row['authentication_accuracy']:.6f}")
    print(f"Saved trust matrix: {figure_path}")
    print(f"Saved plotted data: {csv_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
