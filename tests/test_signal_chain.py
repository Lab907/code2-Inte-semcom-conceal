"""Tests for the Step 1 16-QAM signal chain."""

from __future__ import annotations

import torch

from rfhide.channel import add_awgn
from rfhide.impairments import HardwareImpairmentBank
from rfhide.metrics import ber_16qam
from rfhide.modulation import (
    demodulate_16qam_hard,
    generate_random_bits,
    get_16qam_constellation,
    modulate_16qam,
)
from rfhide.utils import set_seed


def test_16qam_modulation_demodulation_noiseless_ber_zero() -> None:
    """16-QAM hard demodulation should recover noiseless Gray-coded bits."""
    set_seed(7)
    bits = generate_random_bits(batch_size=8, num_symbols=256, device="cpu")
    symbols = modulate_16qam(bits)
    recovered = demodulate_16qam_hard(symbols)

    assert torch.equal(recovered, bits)
    assert torch.allclose(ber_16qam(symbols, bits), torch.zeros(8), atol=0.0)


def test_16qam_constellation_average_power_is_one() -> None:
    """The normalized 16-QAM constellation should have unit average power."""
    points, labels = get_16qam_constellation("cpu")

    assert points.shape == (16,)
    assert labels.shape == (16, 4)
    assert torch.allclose(points.abs().pow(2).mean(), torch.tensor(1.0), atol=1e-6)


def test_awgn_snr20_noise_power_is_reasonable() -> None:
    """Measured AWGN noise power should match SNR=20 within sampling tolerance."""
    set_seed(11)
    bits = generate_random_bits(batch_size=64, num_symbols=4096, device="cpu")
    x = modulate_16qam(bits)
    y = add_awgn(x, snr_db=20.0)

    signal_power = x.abs().pow(2).mean()
    noise_power = (y - x).abs().pow(2).mean()
    expected_noise_power = signal_power / 100.0

    assert torch.isclose(noise_power, expected_noise_power, rtol=0.15, atol=2e-3)


def test_three_tx_outputs_are_different() -> None:
    """Different transmitter ids should produce different impaired outputs."""
    set_seed(13)
    bits = generate_random_bits(batch_size=4, num_symbols=256, device="cpu")
    x = modulate_16qam(bits)
    x_by_tx = x.unsqueeze(0).expand(3, -1, -1).contiguous()
    bank = HardwareImpairmentBank({"num_tx": 3}, device="cpu")

    y = bank.apply(x_by_tx, tx_ids=torch.arange(3), time_indices=torch.zeros(3))

    assert not torch.allclose(y[0], y[1])
    assert not torch.allclose(y[1], y[2])
    assert not torch.allclose(y[0], y[2])


def test_same_tx_different_time_index_keeps_output_fixed() -> None:
    """Changing time_index should not alter fixed hardware impairments."""
    set_seed(17)
    bits = generate_random_bits(batch_size=2, num_symbols=256, device="cpu")
    x = modulate_16qam(bits)
    bank = HardwareImpairmentBank({"num_tx": 3}, device="cpu")

    y_t0 = bank.apply(x, tx_ids=1, time_indices=0)
    y_t1 = bank.apply(x, tx_ids=1, time_indices=25)

    assert torch.allclose(y_t0, y_t1)
