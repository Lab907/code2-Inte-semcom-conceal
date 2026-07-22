"""Tests for the Step 5 teacher compensation dataset artifacts."""

from __future__ import annotations

from pathlib import Path

import torch

from rfhide.channel import add_awgn
from rfhide.config import load_config
from rfhide.impairments import HardwareImpairmentBank
from rfhide.metrics import ber_16qam, estimate_complex_gain, evm_linear


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "outputs" / "snr20" / "data"
SPLIT_PATHS = {
    "train": DATA_DIR / "comp_train.pt",
    "val": DATA_DIR / "comp_val.pt",
    "test": DATA_DIR / "comp_test.pt",
}


def _load_split(split: str) -> dict:
    """Load one saved compensation dataset split."""
    return torch.load(SPLIT_PATHS[split], map_location="cpu")


def test_compensation_dataset_files_exist() -> None:
    """The train, val, and test dataset files should exist."""
    for path in SPLIT_PATHS.values():
        assert path.exists(), f"Missing dataset file: {path}"


def test_train_val_test_are_nonempty() -> None:
    """Every split should contain at least one flattened Tx sample."""
    for split in SPLIT_PATHS:
        data = _load_split(split)
        assert data["x_clean"].shape[0] > 0
        assert data["p_teacher"].shape[0] == data["x_clean"].shape[0]


def test_three_tx_sample_counts_are_balanced() -> None:
    """Each split should have equal sample counts for the three transmitters."""
    for split in SPLIT_PATHS:
        data = _load_split(split)
        counts = torch.bincount(data["tx_id"].long(), minlength=3)
        assert counts.tolist()[0] == counts.tolist()[1] == counts.tolist()[2]


def test_p_teacher_shape_matches_x_clean() -> None:
    """Saved teacher residuals should be complex tensors with [S, N] shape."""
    data = _load_split("train")

    assert data["p_teacher"].shape == data["x_clean"].shape
    assert torch.is_complex(data["p_teacher"])
    assert torch.is_complex(data["x_clean"])
    assert data["bits"].shape == (data["x_clean"].shape[0], data["x_clean"].shape[1] * 4)


def test_residual_power_does_not_exceed_threshold() -> None:
    """Residual power ratio should obey the teacher projection threshold."""
    config = load_config(PROJECT_ROOT / "configs" / "snr20.yaml")
    threshold = float(config["teacher"]["max_residual_power_ratio"])

    for split in SPLIT_PATHS:
        data = _load_split(split)
        assert torch.all(data["residual_power_ratio"] <= threshold + 1e-5)


def test_saved_p_teacher_can_recompute_evm_and_ber() -> None:
    """Saved residuals should be usable in the hardware chain for metrics."""
    config = load_config(PROJECT_ROOT / "configs" / "snr20.yaml")
    impairment_cfg = dict(config["impairments"])
    impairment_cfg["sample_rate"] = float(config["signal"]["sample_rate"])
    bank = HardwareImpairmentBank(impairment_cfg, device="cpu")
    data = _load_split("train")

    sample_count = min(12, data["x_clean"].shape[0])
    x_clean = data["x_clean"][:sample_count]
    p_teacher = data["p_teacher"][:sample_count]
    tx_ids = data["tx_id"][:sample_count]
    time_indices = data["time_index"][:sample_count]
    snr_db = data["snr_db"][:sample_count]
    bits = data["bits"][:sample_count].reshape(sample_count, x_clean.shape[-1], 4)

    y_imp = bank.apply(x_clean + p_teacher, tx_ids=tx_ids, time_indices=time_indices)
    y_rx = add_awgn(y_imp, snr_db=snr_db)
    gain = estimate_complex_gain(y_rx, x_clean)
    y_eq = y_rx / torch.where(gain.abs() < 1e-12, torch.ones_like(gain), gain)
    evm = evm_linear(y_eq, x_clean, align_gain=False)
    ber = ber_16qam(y_eq, bits)

    assert torch.isfinite(evm).all()
    assert torch.isfinite(ber).all()
    assert torch.all((ber >= 0.0) & (ber <= 1.0))
