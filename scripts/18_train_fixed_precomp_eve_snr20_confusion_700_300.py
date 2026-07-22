"""Train Eve on SNR=20 fixed-precompensated signals and plot one confusion matrix."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.dataset import MultiTxBatchGenerator
from rfhide.logging_utils import get_logger
from rfhide.models_eve import EveCNN
from rfhide.semantic_jscc import semantic_enabled
from rfhide.utils import count_parameters, ensure_dir, get_device, save_json, set_seed

METHOD = "fixed_precomp"
FONT_FAMILY = "Times New Roman"
AXIS_LABEL_SIZE = 30
TICK_LABEL_SIZE = 30
CELL_FONT_SIZE = 33


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Eve with 700/300 per Tx on SNR=20 fixed-precompensated signals."
    )
    parser.add_argument(
        "--config",
        default="outputs/multisnr_semantic_faces/configs/snr20.yaml",
        help="SNR=20 config file.",
    )
    parser.add_argument(
        "--data",
        default="outputs/multisnr_semantic_faces/snr20/data/eval_fixed_precomp.pt",
        help="Existing fixed-precompensated dataset. Fresh data is collected if it is insufficient.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/multisnr_semantic_faces/snr20/figures/confusion_snr20_fixed_precomp_700_300",
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
    parser.add_argument("--force-collect", action="store_true", help="Always collect a fresh fixed-precompensated dataset.")
    parser.add_argument("--use-existing-only", action="store_true", help="Fail instead of collecting if --data is insufficient.")
    parser.add_argument(
        "--eval-batchnorm-mode",
        choices=["batch", "running"],
        default="batch",
        help="Use batch statistics or running statistics for BatchNorm during Eve testing.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override config seed.")
    return parser.parse_args()


def _resolve_project_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_collect_helpers():
    path = PROJECT_ROOT / "scripts" / "05_collect_eval_signals_snr20.py"
    spec = importlib.util.spec_from_file_location("collect_eval_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load collect helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tx_counts(labels: torch.Tensor) -> dict[int, int]:
    return {int(label): int((labels == int(label)).sum().item()) for label in sorted(labels.unique().tolist())}


def _has_enough_samples(data: dict[str, torch.Tensor], needed_per_tx: int, num_classes: int) -> bool:
    labels = data["labels"].long()
    counts = _tx_counts(labels)
    return len(counts) >= num_classes and all(counts.get(tx_id, 0) >= needed_per_tx for tx_id in range(num_classes))


def _load_existing_data(path: Path) -> dict[str, torch.Tensor] | None:
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "signals" not in data or "labels" not in data:
        raise ValueError(f"Dataset must contain 'signals' and 'labels': {path}")
    return data


def _default_checkpoint(config: dict[str, Any]) -> Path:
    eval_cfg = config.get("eval_signal_collection", {})
    checkpoint = (
        eval_cfg.get("checkpoint")
        or config.get("fixed_precomp", {}).get("checkpoint")
        or config.get("compensation_dataset", {}).get("checkpoint")
    )
    if checkpoint is None:
        output_dir = Path(config.get("experiment", {}).get("output_dir", "outputs/snr20"))
        checkpoint = output_dir / "checkpoints" / "teacher_best.pt"
    return _resolve_project_path(checkpoint)


def _collect_fixed_precomp_data(
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

    helpers = _load_collect_helpers()
    num_classes = int(config.get("eve", {}).get("num_classes", config.get("impairments", {}).get("num_tx", 6)))
    num_batches = math.ceil(needed_per_tx / batch_size)
    run_cfg = helpers._collection_config(config, batch_size=batch_size)
    generator = MultiTxBatchGenerator(run_cfg, split="eval", device=device)
    fixed_precomp = helpers._load_fixed_precomp(run_cfg, checkpoint_path, device)
    is_semantic = semantic_enabled(run_cfg)
    columns: dict[str, list[torch.Tensor]] = {}

    for _ in tqdm(range(num_batches), desc="Collecting fixed-precomp Eve data"):
        batch = generator.sample_batch()
        p_fixed = fixed_precomp(
            x_clean=batch["x_clean"],
            tx_ids=batch["tx_ids"],
            snr_db=batch["snr_db"],
            time_indices=batch["time_indices"],
        )
        y_fixed = helpers._apply_chain(
            generator.impairment_bank,
            batch["x_clean_tx"] + p_fixed,
            batch["tx_ids"],
            batch["time_indices"],
            batch["snr_db"],
        )
        entry = helpers._flatten_signal_entry(
            y_fixed,
            batch["x_clean_tx"],
            batch["bits"],
            batch["tx_ids"],
            batch["snr_db"],
            p_fixed,
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


def _fixed_train_test_indices(
    labels: torch.Tensor,
    train_per_tx: int,
    test_per_tx: int,
    num_classes: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    train_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    needed = train_per_tx + test_per_tx
    for tx_id in range(num_classes):
        indices = torch.where(labels == tx_id)[0]
        if indices.numel() < needed:
            raise ValueError(
                f"Tx {tx_id + 1} needs {needed} samples ({train_per_tx} train + {test_per_tx} test), "
                f"but only found {int(indices.numel())}."
            )
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        train_parts.append(indices[:train_per_tx])
        test_parts.append(indices[train_per_tx : train_per_tx + test_per_tx])
    train_idx = torch.cat(train_parts)
    test_idx = torch.cat(test_parts)
    train_idx = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
    test_idx = test_idx[torch.randperm(test_idx.numel(), generator=generator)]
    return train_idx, test_idx


def _make_loader(signals: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(signals[indices].float(), labels[indices].long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _accuracy_and_confusion(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> tuple[float, torch.Tensor]:
    preds = logits.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for target, pred in zip(labels.cpu(), preds.cpu()):
        confusion[int(target), int(pred)] += 1
    return acc, confusion


def _train_epoch(
    model: EveCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None,
    desc: str,
) -> tuple[float, float]:
    model.train()
    losses: list[float] = []
    correct = 0
    total = 0
    progress = tqdm(loader, desc=desc)
    for batch_idx, (signals, labels) in enumerate(progress, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        signals = signals.to(device)
        labels = labels.to(device)
        logits = model(signals)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
        value = float(loss.detach().cpu().item())
        losses.append(value)
        progress.set_postfix(loss=f"{value:.4f}", acc=f"{correct / max(total, 1):.4f}")
    return sum(losses) / max(len(losses), 1), correct / max(total, 1)


@torch.no_grad()
def _evaluate(
    model: EveCNN,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    batchnorm_mode: str,
) -> tuple[float, float, torch.Tensor]:
    if batchnorm_mode == "batch":
        model.train()
    else:
        model.eval()
    losses: list[float] = []
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for signals, labels in loader:
        signals = signals.to(device)
        labels = labels.to(device)
        logits = model(signals)
        losses.append(float(F.cross_entropy(logits, labels).detach().cpu().item()))
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    acc, confusion = _accuracy_and_confusion(logits_cat, labels_cat, num_classes)
    return sum(losses) / max(len(losses), 1), acc, confusion


def _write_metrics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_confusion(matrix: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    plt.rcParams["font.family"] = FONT_FAMILY
    num_classes = matrix.shape[0]
    vmax = max(float(matrix.max(initial=0.0)), 1.0)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=vmax)
    ax.set_xlabel("Predicted Tx", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("True Tx", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels([f"T{idx + 1}" for idx in range(num_classes)])
    ax.set_yticklabels([f"T{idx + 1}" for idx in range(num_classes)])
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)

    threshold = vmax * 0.55
    for true_idx in range(num_classes):
        for pred_idx in range(num_classes):
            value = int(matrix[true_idx, pred_idx])
            ax.text(
                pred_idx,
                true_idx,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "#222222",
                fontsize=CELL_FONT_SIZE,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"), dpi=220)
    fig.savefig(path.with_suffix(".svg"), dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config_path = _resolve_project_path(args.config)
    config = load_config(config_path)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    set_seed(seed)
    device = get_device(bool(config.get("device", {}).get("prefer_cuda", True)))
    logger = get_logger("rfhide.fixed_precomp_eve_700_300")

    output_dir = ensure_dir(_resolve_project_path(args.output_dir))
    data_path = _resolve_project_path(args.data)
    checkpoint_path = _resolve_project_path(args.checkpoint) if args.checkpoint else _default_checkpoint(config)
    num_classes = int(config.get("eve", {}).get("num_classes", config.get("impairments", {}).get("num_tx", 6)))
    needed_per_tx = int(args.train_samples_per_tx + args.test_samples_per_tx)

    data = None if args.force_collect else _load_existing_data(data_path)
    if data is not None and _has_enough_samples(data, needed_per_tx, num_classes):
        used_data_path = data_path
        logger.info("Using existing fixed-precompensated dataset: %s", used_data_path)
    else:
        if args.use_existing_only:
            counts = {} if data is None else _tx_counts(data["labels"].long())
            raise ValueError(
                f"Existing dataset is insufficient for {needed_per_tx} samples per Tx. "
                f"Found counts: {counts}. Remove --use-existing-only to collect fresh data."
            )
        data, used_data_path = _collect_fixed_precomp_data(
            config,
            output_dir,
            checkpoint_path,
            needed_per_tx,
            int(args.train_samples_per_tx),
            int(args.test_samples_per_tx),
            args.collect_batch_size,
            device,
        )
        logger.info("Collected fresh fixed-precompensated dataset: %s", used_data_path)

    labels = data["labels"].long()
    signals = data["signals"].float()
    train_idx, test_idx = _fixed_train_test_indices(
        labels,
        int(args.train_samples_per_tx),
        int(args.test_samples_per_tx),
        num_classes,
        seed,
    )

    eve_cfg = config.get("eve", {})
    epochs = int(args.epochs if args.epochs is not None else eve_cfg.get("epochs", 20))
    batch_size = int(args.batch_size if args.batch_size is not None else eve_cfg.get("batch_size", 64))
    lr = float(args.lr if args.lr is not None else eve_cfg.get("lr", 0.001))
    train_loader = _make_loader(signals, labels, train_idx, batch_size, shuffle=True)
    test_loader = _make_loader(signals, labels, test_idx, batch_size, shuffle=False)

    model = EveCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rows: list[dict[str, float | int]] = []
    best_train_loss = float("inf")
    split_tag = f"{int(args.train_samples_per_tx)}_{int(args.test_samples_per_tx)}"
    checkpoint_out = output_dir / "checkpoints" / f"eve_{METHOD}_snr20_{split_tag}_best.pt"
    ensure_dir(checkpoint_out.parent)

    logger.info(
        "Training Eve on %s | SNR=20 | train/Tx=%d | test/Tx=%d | epochs=%d | batch=%d | device=%s",
        METHOD,
        args.train_samples_per_tx,
        args.test_samples_per_tx,
        epochs,
        batch_size,
        device,
    )
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.max_batches,
            f"Eve {METHOD} {epoch}/{epochs}",
        )
        rows.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc})
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": {"train_loss": train_loss, "train_acc": train_acc},
                },
                checkpoint_out,
            )

    checkpoint = torch.load(checkpoint_out, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, confusion = _evaluate(model, test_loader, device, num_classes, args.eval_batchnorm_mode)

    log_dir = ensure_dir(output_dir / "logs")
    fig_dir = ensure_dir(output_dir / "figures")
    metrics_path = log_dir / f"eve_{METHOD}_snr20_{split_tag}_train_metrics.csv"
    results_path = log_dir / f"eve_{METHOD}_snr20_{split_tag}_results.json"
    figure_path = fig_dir / f"confusion_matrix_snr20_{METHOD}_{split_tag}.png"
    _write_metrics_csv(metrics_path, rows)
    _plot_confusion(confusion.numpy().astype(np.float32), figure_path)

    result = {
        "method": METHOD,
        "snr_db": 20.0,
        "config": str(config_path),
        "data": str(used_data_path),
        "fixed_precomp_checkpoint": str(checkpoint_path),
        "eve_checkpoint": str(checkpoint_out),
        "figure": str(figure_path),
        "vector_pdf": str(figure_path.with_suffix(".pdf")),
        "vector_svg": str(figure_path.with_suffix(".svg")),
        "train_samples_per_tx": int(args.train_samples_per_tx),
        "test_samples_per_tx": int(args.test_samples_per_tx),
        "num_train": int(train_idx.numel()),
        "num_test": int(test_idx.numel()),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "confusion_matrix": confusion.tolist(),
        "tx_counts": _tx_counts(labels),
        "num_parameters": int(count_parameters(model)),
        "font_family": FONT_FAMILY,
        "figure_aspect": "16:9",
        "eval_batchnorm_mode": args.eval_batchnorm_mode,
    }
    save_json(result, results_path)
    print(json.dumps({"test_accuracy": float(test_acc), "figure": str(figure_path), "results": str(results_path)}, indent=2))


if __name__ == "__main__":
    main()
