"""Step 8 retrain CNN Eve classifiers for SNR=20 identity evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.logging_utils import get_logger
from rfhide.models_eve import EveCNN
from rfhide.utils import count_parameters, ensure_dir, get_device, save_json, set_seed

STEP_NAME = "Step 8 train retrained Eve CNN SNR20"
SIGNAL_CLASSES = ["uncompensated", "random_perturb", "fixed_precomp"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Override eve.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override eve.batch_size.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit train batches per epoch.")
    parser.add_argument("--train-samples-per-tx", type=int, default=None, help="Fixed Eve train samples per transmitter.")
    parser.add_argument("--val-samples-per-tx", type=int, default=None, help="Fixed Eve validation samples per transmitter.")
    parser.add_argument("--test-samples-per-tx", type=int, default=None, help="Fixed Eve test samples per transmitter.")
    parser.add_argument("--label-shuffle-check", action="store_true", help="Train with shuffled labels as a sanity check.")
    return parser.parse_args()


def _load_eval_dataset(name: str, data_dir: Path) -> dict[str, torch.Tensor]:
    """Load one collected eval signal class."""
    return torch.load(data_dir / f"eval_{name}.pt", map_location="cpu")


def _balanced_split_indices(
    labels: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create balanced train/val/test indices for each transmitter label."""
    generator = torch.Generator().manual_seed(seed)
    train_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    for label in sorted(labels.unique().tolist()):
        indices = torch.where(labels == int(label))[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        n_train = int(indices.numel() * train_ratio)
        n_val = int(indices.numel() * val_ratio)
        train_parts.append(indices[:n_train])
        val_parts.append(indices[n_train : n_train + n_val])
        test_parts.append(indices[n_train + n_val :])
    return torch.cat(train_parts), torch.cat(val_parts), torch.cat(test_parts)


def _fixed_count_split_indices(
    labels: torch.Tensor,
    train_per_tx: int,
    val_per_tx: int,
    test_per_tx: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create balanced fixed-count train/val/test indices per transmitter."""
    generator = torch.Generator().manual_seed(seed)
    train_parts: list[torch.Tensor] = []
    val_parts: list[torch.Tensor] = []
    test_parts: list[torch.Tensor] = []
    needed = train_per_tx + val_per_tx + test_per_tx
    for label in sorted(labels.unique().tolist()):
        indices = torch.where(labels == int(label))[0]
        if indices.numel() < needed:
            raise ValueError(
                f"Eve split needs {needed} samples for Tx {int(label)} "
                f"({train_per_tx} train, {val_per_tx} val, {test_per_tx} test), "
                f"but only found {int(indices.numel())}. Increase eval collection batches."
            )
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        train_parts.append(indices[:train_per_tx])
        val_start = train_per_tx
        val_end = val_start + val_per_tx
        val_parts.append(indices[val_start:val_end])
        test_parts.append(indices[val_end : val_end + test_per_tx])
    return torch.cat(train_parts), torch.cat(val_parts), torch.cat(test_parts)


def _make_loader(signals: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    """Create a DataLoader for selected indices."""
    dataset = TensorDataset(signals[indices].float(), labels[indices].long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _accuracy_and_confusion(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> tuple[float, torch.Tensor]:
    """Compute accuracy and confusion matrix."""
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
) -> float:
    """Train one Eve epoch."""
    model.train()
    losses: list[float] = []
    progress = tqdm(loader, desc=desc)
    for batch_idx, (signals, labels) in enumerate(progress, start=1):
        if max_batches is not None and batch_idx > max_batches:
            break
        signals = signals.to(device)
        labels = labels.to(device)
        loss = F.cross_entropy(model(signals), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu().item())
        losses.append(value)
        progress.set_postfix(loss=f"{value:.4f}")
    return sum(losses) / max(len(losses), 1)


@torch.no_grad()
def _evaluate(model: EveCNN, loader: DataLoader, device: torch.device, num_classes: int) -> tuple[float, float, torch.Tensor]:
    """Evaluate loss, accuracy, and confusion matrix."""
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


def _write_metrics_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    """Write per-epoch Eve metrics."""
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(path: Path, model: EveCNN, epoch: int, metrics: dict[str, Any]) -> None:
    """Save best Eve checkpoint."""
    ensure_dir(path.parent)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics}, path)


def _train_one_class(
    name: str,
    data: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
    seed_offset: int,
) -> tuple[dict[str, Any], list[dict[str, float | str]]]:
    """Train one fresh Eve CNN for a single signal class."""
    eve_cfg = config.get("eve", {})
    epochs = int(args.epochs if args.epochs is not None else eve_cfg.get("epochs", 30))
    batch_size = int(args.batch_size if args.batch_size is not None else eve_cfg.get("batch_size", 64))
    lr = float(eve_cfg.get("lr", 0.001))
    train_ratio = float(eve_cfg.get("train_ratio", 0.6))
    val_ratio = float(eve_cfg.get("val_ratio", 0.2))
    num_classes = int(eve_cfg.get("num_classes", config.get("impairments", {}).get("num_tx", 6)))
    train_samples_per_tx = args.train_samples_per_tx
    if train_samples_per_tx is None:
        train_samples_per_tx = eve_cfg.get("train_samples_per_tx")
    val_samples_per_tx = args.val_samples_per_tx
    if val_samples_per_tx is None:
        val_samples_per_tx = eve_cfg.get("val_samples_per_tx")
    test_samples_per_tx = args.test_samples_per_tx
    if test_samples_per_tx is None:
        test_samples_per_tx = eve_cfg.get("test_samples_per_tx")

    signals = data["signals"].float()
    labels = data["labels"].long().clone()
    if args.label_shuffle_check:
        labels = labels[torch.randperm(labels.numel(), generator=torch.Generator().manual_seed(int(config.get("seed", 42)) + seed_offset))]

    if train_samples_per_tx is not None or test_samples_per_tx is not None:
        train_count = int(train_samples_per_tx if train_samples_per_tx is not None else 500)
        val_count = int(val_samples_per_tx if val_samples_per_tx is not None else max(1, train_count // 5))
        test_count = int(test_samples_per_tx if test_samples_per_tx is not None else 500)
        train_idx, val_idx, test_idx = _fixed_count_split_indices(
            labels,
            train_count,
            val_count,
            test_count,
            int(config.get("seed", 42)) + seed_offset,
        )
        train_samples_per_tx = train_count
        val_samples_per_tx = val_count
        test_samples_per_tx = test_count
    else:
        train_idx, val_idx, test_idx = _balanced_split_indices(labels, train_ratio, val_ratio, int(config.get("seed", 42)) + seed_offset)
    train_loader = _make_loader(signals, labels, train_idx, batch_size, shuffle=True)
    val_loader = _make_loader(signals, labels, val_idx, batch_size, shuffle=False)
    test_loader = _make_loader(signals, labels, test_idx, batch_size, shuffle=False)

    model = EveCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    checkpoint_name = f"eve_{name}_best.pt" if not args.label_shuffle_check else f"eve_{name}_label_shuffle_best.pt"
    checkpoint_path = output_dir / "checkpoints" / checkpoint_name
    best_val_acc = -1.0
    best_metrics: dict[str, Any] = {}
    rows: list[dict[str, float | str]] = []

    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device, args.max_batches, f"Eve {name} {epoch}/{epochs}")
        val_loss, val_acc, _ = _evaluate(model, val_loader, device, num_classes)
        row = {
            "class_name": name,
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        rows.append(row)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = row.copy()
            _save_checkpoint(checkpoint_path, model, epoch, best_metrics)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, confusion = _evaluate(model, test_loader, device, num_classes)

    result = {
        "class_name": name,
        "best_val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "confusion_matrix": confusion.tolist(),
        "checkpoint": str(checkpoint_path),
        "num_train": int(train_idx.numel()),
        "num_val": int(val_idx.numel()),
        "num_test": int(test_idx.numel()),
        "train_samples_per_tx": None if train_samples_per_tx is None else int(train_samples_per_tx),
        "val_samples_per_tx": None if val_samples_per_tx is None else int(val_samples_per_tx),
        "test_samples_per_tx": None if test_samples_per_tx is None else int(test_samples_per_tx),
        "mean_ber": float(data["ber"].mean().item()),
        "mean_evm": float(data["evm"].mean().item()),
        "label_shuffle_check": bool(args.label_shuffle_check),
        "num_parameters": int(count_parameters(model)),
    }
    return result, rows


def main() -> None:
    """Train fresh CNN Eve classifiers for all three signal classes."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = get_device(bool(config.get("device", {}).get("prefer_cuda", True)))
    logger = get_logger("rfhide.eve")
    logger.info("Current step: %s", STEP_NAME)
    logger.info("Config path: %s", args.config)
    logger.info("Device: %s", device)
    logger.info("Label shuffle check: %s", args.label_shuffle_check)

    output_dir = PROJECT_ROOT / config.get("experiment", {}).get("output_dir", "outputs/snr20")
    data_dir = output_dir / "data"
    log_dir = ensure_dir(output_dir / "logs")
    ensure_dir(output_dir / "checkpoints")

    results: dict[str, Any] = {}
    all_rows: list[dict[str, float | str]] = []
    for idx, name in enumerate(SIGNAL_CLASSES):
        data = _load_eval_dataset(name, data_dir)
        result, rows = _train_one_class(name, data, config, device, args, output_dir, seed_offset=100 * idx)
        results[name] = result
        all_rows.extend(rows)
        logger.info(
            "%s | best val acc %.4f | test acc %.4f | BER %.6f | EVM %.6f | confusion %s",
            name,
            result["best_val_accuracy"],
            result["test_accuracy"],
            result["mean_ber"],
            result["mean_evm"],
            result["confusion_matrix"],
        )

    metrics_path = log_dir / ("eve_train_metrics_shuffle.csv" if args.label_shuffle_check else "eve_train_metrics.csv")
    results_path = log_dir / ("eve_results_snr20_label_shuffle.json" if args.label_shuffle_check else "eve_results_snr20.json")
    _write_metrics_csv(metrics_path, all_rows)
    save_json(
        {
            "strong_eve_setting": "fresh EveCNN retrained separately per signal class",
            "results": results,
        },
        results_path,
    )
    logger.info("Saved Eve metrics CSV: %s", metrics_path)
    logger.info("Saved Eve results JSON: %s", results_path)
    logger.info("Step 8 Eve evaluation passed")


if __name__ == "__main__":
    main()
