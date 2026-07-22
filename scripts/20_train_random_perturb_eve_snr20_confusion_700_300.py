"""Train Eve on SNR=20 random-perturbed signals and plot one confusion matrix."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.dataset import MultiTxBatchGenerator
from rfhide.semantic_jscc import semantic_enabled
from rfhide.utils import ensure_dir

METHOD = "random_perturb"
DEFAULT_OUTPUT_DIR = "outputs/multisnr_semantic_faces/snr20/figures/confusion_snr20_random_perturb_700_300"


def _load_fixed_split_module():
    path = PROJECT_ROOT / "scripts" / "18_train_fixed_precomp_eve_snr20_confusion_700_300.py"
    spec = importlib.util.spec_from_file_location("fixed_split_eve", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load fixed-split Eve helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Eve with 700/300 per Tx on SNR=20 random-perturbed signals."
    )
    parser.add_argument(
        "--config",
        default="outputs/multisnr_semantic_faces/configs/snr20.yaml",
        help="SNR=20 config file.",
    )
    parser.add_argument(
        "--data",
        default=f"{DEFAULT_OUTPUT_DIR}/data/eval_random_perturb_700_300.pt",
        help="Existing random-perturbed dataset. Fresh data is collected if it is insufficient.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the new matrix and training artifacts.",
    )
    parser.add_argument("--checkpoint", default=None, help="Override fixed-precompensator checkpoint.")
    parser.add_argument("--train-samples-per-tx", type=int, default=700, help="Training samples per transmitter.")
    parser.add_argument("--test-samples-per-tx", type=int, default=300, help="Testing samples per transmitter.")
    parser.add_argument("--epochs", type=int, default=None, help="Override eve.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eve.batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override eve.lr.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit train batches per epoch for debugging.")
    parser.add_argument("--collect-batch-size", type=int, default=32, help="Batch size for fresh data collection.")
    parser.add_argument("--force-collect", action="store_true", help="Always collect a fresh random-perturbed dataset.")
    parser.add_argument("--use-existing-only", action="store_true", help="Fail instead of collecting if --data is insufficient.")
    parser.add_argument(
        "--eval-batchnorm-mode",
        choices=["batch", "running"],
        default="batch",
        help="Use batch statistics or running statistics for BatchNorm during Eve testing.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    return parser.parse_args()


def _collect_random_perturb_data(
    config: dict[str, Any],
    output_dir: Path,
    checkpoint_path: Path,
    needed_per_tx: int,
    train_per_tx: int,
    test_per_tx: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], Path]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Fixed-precompensator checkpoint not found: {checkpoint_path}")

    fixed_module = _load_fixed_split_module()
    helpers = fixed_module._load_collect_helpers()
    num_classes = int(config.get("eve", {}).get("num_classes", config.get("impairments", {}).get("num_tx", 6)))
    num_batches = math.ceil(needed_per_tx / batch_size)
    run_cfg = helpers._collection_config(config, batch_size=batch_size)
    generator = MultiTxBatchGenerator(run_cfg, split="eval", device=device)
    fixed_precomp = helpers._load_fixed_precomp(run_cfg, checkpoint_path, device)
    max_ratio = float(
        run_cfg.get("fixed_precomp", {}).get(
            "max_residual_power_ratio",
            run_cfg.get("teacher", {}).get("max_residual_power_ratio", 0.05),
        )
    )
    is_semantic = semantic_enabled(run_cfg)
    columns: dict[str, list[torch.Tensor]] = {}

    for _ in tqdm(range(num_batches), desc="Collecting random-perturb Eve data"):
        batch = generator.sample_batch()
        p_fixed = fixed_precomp(
            x_clean=batch["x_clean"],
            tx_ids=batch["tx_ids"],
            snr_db=batch["snr_db"],
            time_indices=batch["time_indices"],
        )
        p_rand = helpers._random_residual_like(p_fixed, batch["x_clean_tx"], max_ratio)
        y_rand = helpers._apply_chain(
            generator.impairment_bank,
            batch["x_clean_tx"] + p_rand,
            batch["tx_ids"],
            batch["time_indices"],
            batch["snr_db"],
        )
        entry = helpers._flatten_signal_entry(
            y_rand,
            batch["x_clean_tx"],
            batch["bits"],
            batch["tx_ids"],
            batch["snr_db"],
            p_rand,
            semantic_image=batch.get("semantic_image"),
            semantic_label=batch.get("semantic_label"),
            is_semantic=is_semantic,
        )
        helpers._append_columns(columns, entry)

    data = helpers._cat_columns(columns)
    data["meta"] = {
        "class_name": METHOD,
        "semantic_enabled": is_semantic,
        "format": "signals/x_clean are [S, 2, N] real-imag channels.",
        "checkpoint": str(checkpoint_path),
        "train_samples_per_tx": train_per_tx,
        "test_samples_per_tx": test_per_tx,
        "num_tx": num_classes,
        "fixed_split_eve": True,
    }
    data_dir = ensure_dir(output_dir / "data")
    path = data_dir / f"eval_{METHOD}_{train_per_tx}_{test_per_tx}.pt"
    torch.save(data, path)
    return data, path


def main() -> None:
    fixed_module = _load_fixed_split_module()
    fixed_module.METHOD = METHOD
    fixed_module.parse_args = parse_args
    fixed_module._collect_fixed_precomp_data = _collect_random_perturb_data
    fixed_module.main()


if __name__ == "__main__":
    main()
