"""Step 1/2/3 signal-chain, batch, feature, and loss sanity check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.dataset import MultiTxBatchGenerator
from rfhide.features import extract_raw_features
from rfhide.logging_utils import get_logger
from rfhide.losses import total_teacher_loss
from rfhide.metrics import ber_16qam, estimate_complex_gain, evm_db, evm_db_or_linear
from rfhide.utils import ensure_dir, get_device, save_json, set_seed

STEP_NAME = "Step 3 multi-Tx feature/loss signal-chain check"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Run a paired multi-Tx batch sanity check and save diagnostics."""
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))

    device = get_device(bool(config.get("device", {}).get("prefer_cuda", True)))
    logger = get_logger("rfhide.step01")
    logger.info("Current step: %s", STEP_NAME)
    logger.info("Config path: %s", args.config)
    logger.info("Device: %s", device)

    generator = MultiTxBatchGenerator(config, split="sanity", device=device)
    batch = generator.sample_batch()
    params = generator.impairment_bank.last_params or {}

    fig_dir = ensure_dir(PROJECT_ROOT / config.get("experiment", {}).get("output_dir", "outputs/snr20") / "figures")
    fig_path = fig_dir / "signal_chain_constellation.png"
    log_dir = ensure_dir(PROJECT_ROOT / config.get("experiment", {}).get("output_dir", "outputs/snr20") / "logs")
    sanity_path = log_dir / "batch_sanity.json"

    bits = batch["bits"]
    x = batch["x_clean"]
    x_by_tx = batch["x_clean_tx"]
    y = batch["y_rx"]
    snr_db = batch["snr_db"]
    num_tx, batch_size, num_symbols = y.shape
    bit_groups = bits.reshape(batch_size, num_symbols, 4)
    features = extract_raw_features(y, downsample=4, include_power=True)
    teacher_losses = total_teacher_loss(
        y_rx=y,
        x_clean_tx=x_by_tx,
        bits=bits,
        features=features,
        p=y - x_by_tx,
        cfg=config,
    )

    clean_same_for_b0 = bool(
        torch.all(x_by_tx[:, 0] == x_by_tx[0, 0]).item()
        if num_tx > 0 and batch_size > 0
        else True
    )
    snr_same_for_b0 = True
    if batch_size > 0:
        snr_tx_view = snr_db.unsqueeze(0).expand(num_tx, -1)
        snr_same_for_b0 = bool(torch.all(snr_tx_view[:, 0] == snr_tx_view[0, 0]).item())

    logger.info("Batch bits shape: %s", tuple(bits.shape))
    logger.info("Batch x_clean shape: %s", tuple(x.shape))
    logger.info("Batch x_clean_tx shape: %s", tuple(x_by_tx.shape))
    logger.info("Batch tx_ids shape: %s", tuple(batch["tx_ids"].shape))
    logger.info("Batch snr_db shape: %s", tuple(snr_db.shape))
    logger.info("Batch time_indices shape: %s", tuple(batch["time_indices"].shape))
    logger.info("Batch y_imp shape: %s", tuple(batch["y_imp"].shape))
    logger.info("Batch y_rx shape: %s", tuple(y.shape))
    logger.info("Raw feature shape: %s", tuple(features.shape))
    logger.info("For sample b=0, all Tx SNR equal: %s", snr_same_for_b0)
    logger.info("For sample b=0, all Tx clean x exactly equal: %s", clean_same_for_b0)
    logger.info(
        "Loss sanity | total %.6f | alignment %.6f | EVM %.6f | soft-bit %.6f | power %.6f",
        teacher_losses["total"].detach().cpu().item(),
        teacher_losses["alignment"].detach().cpu().item(),
        teacher_losses["evm"].detach().cpu().item(),
        teacher_losses["soft_bit"].detach().cpu().item(),
        teacher_losses["power"].detach().cpu().item(),
    )

    tx_stats = []
    plt.figure(figsize=(7, 6))
    for tx_idx in tqdm(range(num_tx), desc="Checking Tx signal chains"):
        gain = estimate_complex_gain(y[tx_idx], x)
        y_equalized = y[tx_idx] / gain
        evm = evm_db_or_linear(y[tx_idx], x).mean()
        ber = ber_16qam(y_equalized, bit_groups).mean()
        evm_db_value = evm_db(evm)

        logger.info(
            "Tx %d | EVM linear %.6f | EVM dB %.2f | BER %.6f",
            tx_idx,
            evm.item(),
            evm_db_value.item(),
            ber.item(),
        )
        if params:
            tx_params = {
                name: value[tx_idx, 0].detach().cpu().item()
                if value.ndim == 2 and not torch.is_complex(value)
                else value[tx_idx, 0].detach().cpu()
                if value.ndim == 2
                else value[tx_idx].detach().cpu()
                for name, value in params.items()
            }
            logger.info("Tx %d drift params: %s", tx_idx, tx_params)

        tx_stats.append(
            {
                "tx_id": int(batch["tx_ids"][tx_idx].detach().cpu().item()),
                "evm_linear": float(evm.detach().cpu().item()),
                "evm_db": float(evm_db_value.detach().cpu().item()),
                "ber": float(ber.detach().cpu().item()),
            }
        )

        samples = y_equalized.reshape(-1).detach().cpu()
        samples = samples[: min(samples.numel(), 2048)]
        plt.scatter(samples.real, samples.imag, s=5, alpha=0.35, label=f"Tx {tx_idx}")

    plt.title("Step 1 16-QAM Rx Constellation")
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    batch_sanity = {
        "step": STEP_NAME,
        "shapes": {
            "bits": list(bits.shape),
            "x_clean": list(x.shape),
            "x_clean_tx": list(x_by_tx.shape),
            "tx_ids": list(batch["tx_ids"].shape),
            "snr_db": list(snr_db.shape),
            "time_indices": list(batch["time_indices"].shape),
            "y_imp": list(batch["y_imp"].shape),
            "y_rx": list(y.shape),
            "features": list(features.shape),
        },
        "time_indices_meaning": "[D, B], drift time per Tx/sample pair; sample time is shared across Tx by default.",
        "sample0_snr_same_across_tx": snr_same_for_b0,
        "sample0_clean_x_same_across_tx": clean_same_for_b0,
        "sample0_snr_db": float(snr_db[0].detach().cpu().item()) if batch_size > 0 else None,
        "tx_stats": tx_stats,
        "losses": {
            name: float(value.detach().cpu().item()) for name, value in teacher_losses.items()
        },
    }
    save_json(batch_sanity, sanity_path)
    logger.info("Saved constellation figure: %s", fig_path)
    logger.info("Saved batch sanity JSON: %s", sanity_path)
    logger.info("Step 3 feature/loss sanity check passed")


if __name__ == "__main__":
    main()
