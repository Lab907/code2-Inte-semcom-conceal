"""Tests for Step 7 collected evaluation signal datasets."""

from __future__ import annotations

from pathlib import Path

import torch

from rfhide.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "outputs" / "snr20" / "data"
EVAL_PATHS = {
    "uncompensated": DATA_DIR / "eval_uncompensated.pt",
    "random_perturb": DATA_DIR / "eval_random_perturb.pt",
    "fixed_precomp": DATA_DIR / "eval_fixed_precomp.pt",
}


def _load(name: str) -> dict:
    """Load one eval signal dataset."""
    return torch.load(EVAL_PATHS[name], map_location="cpu")


def test_eval_signal_files_exist() -> None:
    """All three evaluation signal files should exist."""
    for path in EVAL_PATHS.values():
        assert path.exists(), f"Missing eval signal file: {path}"


def test_eval_signal_shapes_are_consistent() -> None:
    """All three classes should share the same tensor shapes."""
    datasets = {name: _load(name) for name in EVAL_PATHS}
    reference = datasets["uncompensated"]
    for data in datasets.values():
        assert data["signals"].shape == reference["signals"].shape
        assert data["labels"].shape == reference["labels"].shape
        assert data["bits"].shape == reference["bits"].shape
        assert data["x_clean"].shape == reference["x_clean"].shape
        assert data["signals"].ndim == 3
        assert data["signals"].shape[1] == 2
        assert data["x_clean"].shape[1] == 2
        assert data["tx_id"].shape == data["labels"].shape
        assert data["evm"].shape == data["labels"].shape
        assert data["ber"].shape == data["labels"].shape


def test_each_eval_class_has_balanced_tx_labels() -> None:
    """Every eval class should contain balanced samples for Tx 0, 1, and 2."""
    for name in EVAL_PATHS:
        data = _load(name)
        counts = torch.bincount(data["tx_id"].long(), minlength=3)
        assert counts.tolist()[0] == counts.tolist()[1] == counts.tolist()[2]


def test_random_and_fixed_residual_power_match_within_ten_percent() -> None:
    """Random perturbation power should match fixed precomp power within 10%."""
    random_data = _load("random_perturb")
    fixed_data = _load("fixed_precomp")
    random_power = random_data["residual_power_ratio"].mean()
    fixed_power = fixed_data["residual_power_ratio"].mean()
    relative_gap = (random_power - fixed_power).abs() / fixed_power.clamp_min(1e-12)

    assert relative_gap <= 0.10


def test_fixed_precomp_residual_power_is_below_threshold() -> None:
    """Fixed precomp residuals should obey the configured projection limit."""
    config = load_config(PROJECT_ROOT / "configs" / "snr20.yaml")
    threshold = float(config["fixed_precomp"]["max_residual_power_ratio"])
    fixed_data = _load("fixed_precomp")

    assert torch.all(fixed_data["residual_power_ratio"] <= threshold + 1e-5)


def test_eval_evm_and_ber_are_present_and_finite() -> None:
    """All saved EVM and BER fields should be finite and valid."""
    for name in EVAL_PATHS:
        data = _load(name)
        assert torch.isfinite(data["evm"]).all()
        assert torch.isfinite(data["ber"]).all()
        assert torch.all(data["ber"] >= 0.0)
        assert torch.all(data["ber"] <= 1.0)
