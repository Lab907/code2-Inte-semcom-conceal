"""Tests for the Step 2 multi-transmitter batch generator."""

from __future__ import annotations

import torch

from rfhide.dataset import MultiTxBatchGenerator
from rfhide.modulation import demodulate_16qam_hard
from rfhide.utils import set_seed


def _test_cfg() -> dict:
    """Return a compact deterministic config for generator tests."""
    return {
        "seed": 123,
        "signal": {
            "snr_db": 20,
            "sample_rate": 1_000_000,
            "num_symbols": 128,
            "modulation": "16qam",
        },
        "data": {
            "batch_size": 5,
            "num_workers": 0,
        },
        "impairments": {
            "num_tx": 3,
            "gain_db_std": 0.8,
            "gain_drift_db_per_step": 0.0,
            "phase_rad_std": 0.25,
            "phase_drift_rad_per_step": 0.0,
            "cfo_hz_std": 80.0,
            "cfo_drift_hz_per_step": 0.0,
            "iq_gain_mismatch_db_std": 0.35,
            "iq_phase_mismatch_rad_std": 0.04,
            "dc_offset_std": 0.015,
        },
    }


def test_clean_signal_is_paired_across_three_tx() -> None:
    """All transmitters should receive the exact same clean batch."""
    set_seed(101)
    batch = MultiTxBatchGenerator(_test_cfg(), split="test", device="cpu").sample_batch()
    x_clean_tx = batch["x_clean_tx"]

    assert torch.equal(x_clean_tx[0], batch["x_clean"])
    assert torch.equal(x_clean_tx[0], x_clean_tx[1])
    assert torch.equal(x_clean_tx[1], x_clean_tx[2])


def test_three_tx_rx_outputs_are_not_identical() -> None:
    """Different transmitter impairments should make Rx outputs differ."""
    set_seed(102)
    batch = MultiTxBatchGenerator(_test_cfg(), split="test", device="cpu").sample_batch()
    y_rx = batch["y_rx"]

    assert not torch.equal(y_rx[0], y_rx[1])
    assert not torch.equal(y_rx[1], y_rx[2])
    assert not torch.equal(y_rx[0], y_rx[2])


def test_snr_condition_is_shared_across_tx_dimension() -> None:
    """Each sample's SNR should be identical for all transmitters."""
    set_seed(103)
    batch = MultiTxBatchGenerator(_test_cfg(), split="test", device="cpu").sample_batch()
    snr_by_tx = batch["snr_db"].unsqueeze(0).expand(batch["x_clean_tx"].shape[0], -1)

    assert torch.equal(snr_by_tx[0], snr_by_tx[1])
    assert torch.equal(snr_by_tx[1], snr_by_tx[2])
    assert torch.all(snr_by_tx == 20)


def test_batch_shapes_are_correct() -> None:
    """The generator should return the documented tensor shapes."""
    set_seed(104)
    batch = MultiTxBatchGenerator(_test_cfg(), split="test", device="cpu").sample_batch()

    assert batch["bits"].shape == (5, 128 * 4)
    assert batch["x_clean"].shape == (5, 128)
    assert batch["x_clean_tx"].shape == (3, 5, 128)
    assert batch["tx_ids"].shape == (3,)
    assert batch["snr_db"].shape == (5,)
    assert batch["time_indices"].shape == (3, 5)
    assert batch["y_imp"].shape == (3, 5, 128)
    assert batch["y_rx"].shape == (3, 5, 128)
    assert torch.is_complex(batch["x_clean"])
    assert torch.is_complex(batch["y_imp"])
    assert torch.is_complex(batch["y_rx"])


def test_bits_and_x_clean_correspond() -> None:
    """Demodulating clean symbols should recover the flattened batch bits."""
    set_seed(105)
    batch = MultiTxBatchGenerator(_test_cfg(), split="test", device="cpu").sample_batch()
    recovered_bits = demodulate_16qam_hard(batch["x_clean"]).reshape(batch["bits"].shape)

    assert torch.equal(recovered_bits, batch["bits"])
