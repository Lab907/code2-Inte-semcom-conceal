"""Plot SNR=20 PSNR line and identity-accuracy bars from existing results."""

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
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.utils import ensure_dir

METHODS = ["uncompensated", "random_perturb", "fixed_precomp"]
DISPLAY_LABELS = ["Original", "Random Perturb", "Pre-Concealment"]
EVE_700_300_RESULT_PATHS = {
    "uncompensated": "auth/outputs/eve_700_300_confusion/logs/eve_700_300_results.json",
    "random_perturb": (
        "outputs/multisnr_semantic_faces/snr20/figures/"
        "confusion_snr20_random_perturb_700_300/logs/eve_random_perturb_snr20_700_300_results.json"
    ),
    "fixed_precomp": (
        "outputs/multisnr_semantic_faces/snr20/figures/"
        "confusion_snr20_fixed_precomp_700_300/logs/eve_fixed_precomp_snr20_700_300_results.json"
    ),
}
BAR_COLORS = ["#ff6b21", "#c9c9c9", "#08b894"]
LINE_COLOR = "#1db7e8"
FONT_FAMILY = "Times New Roman"
AXIS_LABEL_SIZE = 34
TICK_LABEL_SIZE = 27
XTICK_LABEL_SIZE = 22


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


def _load_identity_accuracy(output_root: Path, snr_db: float) -> dict[str, float]:
    """Load identity recognition accuracy for one SNR."""
    data = _load_json(output_root / "logs" / "multisnr_results.json")
    rows = data.get("results", [])
    if not isinstance(rows, list):
        raise ValueError("Expected a results list in multisnr_results.json")
    return {
        str(row["method"]): float(row["identity_accuracy"])
        for row in rows
        if isinstance(row, dict) and float(row.get("snr_db", -9999.0)) == float(snr_db)
    }


def _load_eve_700_300_identity_accuracy() -> dict[str, float]:
    """Load SNR=20 Eve accuracy from the fixed 700/300 per-Tx experiments."""
    values: dict[str, float] = {}
    for method, path in EVE_700_300_RESULT_PATHS.items():
        data = _load_json(_resolve_project_path(path))
        train_per_tx = int(data.get("train_samples_per_tx", -1))
        test_per_tx = int(data.get("test_samples_per_tx", -1))
        if train_per_tx != 700 or test_per_tx != 300:
            raise ValueError(f"{path} is not a 700/300 per-Tx result.")
        values[method] = float(data["test_accuracy"])
    return values


def _load_psnr(output_root: Path, snr_db: float) -> dict[str, float]:
    """Load Bob PSNR for one SNR."""
    snr_tag = str(int(snr_db)) if float(snr_db).is_integer() else str(snr_db).replace(".", "p")
    data = _load_json(output_root / f"snr{snr_tag}" / "logs" / "semantic_reconstruction_summary.json")
    rows = data.get("summary", [])
    if not isinstance(rows, list):
        raise ValueError("Expected a summary list in semantic_reconstruction_summary.json")
    return {
        str(row["method"]): float(row.get("semantic_psnr_mean", row["semantic_psnr"]))
        for row in rows
        if isinstance(row, dict)
    }


def _collect_rows(output_root: Path, snr_db: float, accuracy_source: str) -> list[dict[str, float | str]]:
    """Join SNR=20 identity accuracy and PSNR values by method."""
    if accuracy_source == "eve-700-300":
        if float(snr_db) != 20.0:
            raise ValueError("--accuracy-source eve-700-300 is only available for SNR=20.")
        accuracy_by_method = _load_eve_700_300_identity_accuracy()
    else:
        accuracy_by_method = _load_identity_accuracy(output_root, snr_db)
    psnr_by_method = _load_psnr(output_root, snr_db)
    rows: list[dict[str, float | str]] = []
    for method, label in zip(METHODS, DISPLAY_LABELS, strict=True):
        if method not in accuracy_by_method:
            raise FileNotFoundError(f"Missing identity accuracy for {method} at SNR={snr_db:g}")
        if method not in psnr_by_method:
            raise FileNotFoundError(f"Missing PSNR for {method} at SNR={snr_db:g}")
        rows.append(
            {
                "method": method,
                "label": label,
                "identity_accuracy": accuracy_by_method[method],
                "psnr_db": psnr_by_method[method],
            }
        )
    return rows


def _write_csv(rows: list[dict[str, float | str]], path: Path) -> None:
    """Save plotted values."""
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "label", "identity_accuracy", "psnr_db"])
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, float | str]], path: Path) -> None:
    """Draw identity-accuracy bars and PSNR line in one figure."""
    ensure_dir(path.parent)
    plt.rcParams["font.family"] = FONT_FAMILY
    x = np.arange(len(rows))
    accuracy = np.asarray([float(row["identity_accuracy"]) for row in rows], dtype=np.float32)
    psnr = np.asarray([float(row["psnr_db"]) for row in rows], dtype=np.float32)
    labels = [str(row["label"]) for row in rows]

    fig, ax_left = plt.subplots(figsize=(13.0, 9.0))
    ax_right = ax_left.twinx()

    ax_left.bar(x, accuracy, color=BAR_COLORS, edgecolor="white", linewidth=1.4, width=0.72, zorder=2)
    ax_right.plot(
        x,
        psnr,
        color=LINE_COLOR,
        marker="o",
        markersize=11.0,
        linewidth=3.2,
        zorder=3,
    )

    ax_left.set_ylabel("Identity recognition accuracy", fontsize=AXIS_LABEL_SIZE, labelpad=18)
    ax_right.set_ylabel("Bob PSNR (dB)", color=LINE_COLOR, fontsize=AXIS_LABEL_SIZE, labelpad=18)
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


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Plot SNR=20 PSNR line with identity-accuracy bars.")
    parser.add_argument("--config", default="configs/multisinr.yaml", help="Multi-SNR YAML config.")
    parser.add_argument("--snr", type=float, default=20.0, help="SNR value to plot.")
    parser.add_argument("--output", default=None, help="Optional PNG output path.")
    parser.add_argument(
        "--accuracy-source",
        choices=["multisnr", "eve-700-300"],
        default="multisnr",
        help="Use the original multi-SNR accuracy or SNR=20 Eve results trained with 700/300 per Tx.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_root = _resolve_project_path(config.get("experiment", {}).get("output_dir", "outputs/multisnr"))
    suffix = "_700_300" if args.accuracy_source == "eve-700-300" else ""
    figure_path = (
        _resolve_project_path(args.output)
        if args.output
        else output_root / "figures" / f"snr{int(args.snr)}_psnr_accuracy{suffix}.png"
    )
    csv_path = output_root / "logs" / f"snr{int(args.snr)}_psnr_accuracy{suffix}.csv"

    rows = _collect_rows(output_root, args.snr, args.accuracy_source)
    _plot(rows, figure_path)
    _write_csv(rows, csv_path)
    print(f"Saved figure: {figure_path}")
    print(f"Saved plotted data: {csv_path}")


if __name__ == "__main__":
    main()
