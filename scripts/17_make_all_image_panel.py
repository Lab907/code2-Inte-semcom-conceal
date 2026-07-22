"""Compose selected result images into a two-row paper panel."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageChops

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.utils import ensure_dir


TOP_ROW_IMAGES = [
    ("semantic_reconstruction_original_uncompensated_snr20.png", (0, 0)),
    ("tsne_snr20_origin.png", (0, 2)),
    ("confusion_matrix_snr30_uncompensated.png", (0, 4)),
    ("confidence_matrix_uncompensated.png", (0, 5)),
]

BOTTOM_ROW_IMAGES = [
    ("semantic_reconstruction_original_uncompensated_fixed_precomp_snr20.png", (1, 0)),
    ("tsne_snr20_origin.png", (1, 1)),
    ("tsne_snr20_random_perturb.png", (1, 2)),
    ("tsne_snr20_pre_compensation.png", (1, 3)),
    ("confusion_matrix_snr30_fixed_precomp.png", (1, 4)),
    ("snr20_psnr_accuracy.png", (1, 5)),
]


def _resolve_project_path(path_like: str | Path) -> Path:
    """Resolve a project-relative path."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _trim_white_margin(image: Image.Image, padding: int = 12) -> Image.Image:
    """Trim white margins while keeping a small padding."""
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, (255, 255, 255))
    diff = ImageChops.difference(rgb, background)
    bbox = diff.getbbox()
    if bbox is None:
        return rgb
    left = max(0, bbox[0] - padding)
    upper = max(0, bbox[1] - padding)
    right = min(rgb.size[0], bbox[2] + padding)
    lower = min(rgb.size[1], bbox[3] + padding)
    return rgb.crop((left, upper, right, lower))


def _load_image(path: Path) -> Image.Image:
    """Load and lightly crop one panel image."""
    if not path.exists():
        raise FileNotFoundError(path)
    return _trim_white_margin(Image.open(path))


def _compose(input_dir: Path, output_base: Path) -> list[Path]:
    """Compose and save the paper panel."""
    ensure_dir(output_base.parent)
    fig_width = 16.2
    fig_height = 5.9
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=False)

    def add_by_width(filename: str, left: float, center_y: float, width: float) -> None:
        image = _load_image(input_dir / filename)
        aspect = image.size[0] / image.size[1]
        height = width / aspect * fig_width / fig_height
        bottom = center_y - height / 2.0
        ax = fig.add_axes([left, bottom, width, height])
        image = _load_image(input_dir / filename)
        ax.imshow(image)
        ax.axis("off")

    def add_by_height(filename: str, center_x: float, center_y: float, height: float) -> None:
        image = _load_image(input_dir / filename)
        aspect = image.size[0] / image.size[1]
        width = aspect * height * fig_height / fig_width
        left = center_x - width / 2.0
        bottom = center_y - height / 2.0
        ax = fig.add_axes([left, bottom, width, height])
        ax.imshow(image)
        ax.axis("off")

    def width_for_height(filename: str, height: float) -> float:
        image = _load_image(input_dir / filename)
        aspect = image.size[0] / image.size[1]
        return aspect * height * fig_height / fig_width

    top_center_y = 0.735
    bottom_center_y = 0.275
    aligned_plot_height = 0.33
    bottom_tsne_height = 0.255
    semantic_width = 0.30
    tsne_centers = [0.38, 0.50, 0.62]
    confusion_center = 0.755
    final_column_center = 0.918
    final_column_width = width_for_height("confidence_matrix_uncompensated.png", aligned_plot_height)

    add_by_width("semantic_reconstruction_original_uncompensated_snr20.png", 0.012, top_center_y, semantic_width)
    add_by_height("tsne_snr20_origin.png", tsne_centers[1], top_center_y, aligned_plot_height)
    add_by_height("confusion_matrix_snr30_uncompensated.png", confusion_center, top_center_y, aligned_plot_height)
    add_by_width(
        "confidence_matrix_uncompensated.png",
        final_column_center - final_column_width / 2.0,
        top_center_y,
        final_column_width,
    )

    add_by_width("semantic_reconstruction_original_uncompensated_fixed_precomp_snr20.png", 0.012, bottom_center_y, semantic_width)
    add_by_height("tsne_snr20_origin.png", tsne_centers[0], bottom_center_y, bottom_tsne_height)
    add_by_height("tsne_snr20_random_perturb.png", tsne_centers[1], bottom_center_y, bottom_tsne_height)
    add_by_height("tsne_snr20_pre_compensation.png", tsne_centers[2], bottom_center_y, bottom_tsne_height)
    add_by_height("confusion_matrix_snr30_fixed_precomp.png", confusion_center, bottom_center_y, aligned_plot_height)
    add_by_width(
        "snr20_psnr_accuracy.png",
        final_column_center - final_column_width / 2.0,
        bottom_center_y,
        final_column_width,
    )

    output_paths = [
        output_base.with_suffix(".pdf"),
        output_base.with_suffix(".svg"),
        output_base.with_suffix(".png"),
    ]
    for path in output_paths:
        fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return output_paths


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Compose all-folder result images into a two-row paper panel.")
    parser.add_argument(
        "--input-dir",
        default="outputs/multisnr_semantic_faces/all",
        help="Directory containing the selected result PNG files.",
    )
    parser.add_argument(
        "--output",
        default="outputs/multisnr_semantic_faces/all/uncompensated_precomp_panel",
        help="Output base path without extension.",
    )
    args = parser.parse_args()

    input_dir = _resolve_project_path(args.input_dir)
    output_base = _resolve_project_path(args.output)
    for path in _compose(input_dir, output_base):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
