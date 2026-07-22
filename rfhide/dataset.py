"""Multi-transmitter paired batch generation for RF signal chains.

``MultiTxBatchGenerator`` creates batches where one clean 16-QAM signal batch is
shared across all transmitters. Hardware impairment parameters vary by
transmitter and time, while SNR is sampled per batch item and shared across the
Tx dimension.
"""

from __future__ import annotations

from typing import Any

import torch

from rfhide.channel import add_awgn
from rfhide.impairments import HardwareImpairmentBank
from rfhide.modulation import generate_random_bits, modulate_16qam
from rfhide.semantic_jscc import SemanticBatchSource, semantic_enabled


class MultiTxBatchGenerator:
    """Generate paired clean/impaired/Rx batches for multiple transmitters.

    Returned shapes:
        ``bits``: ``[B, N * 4]`` flattened 16-QAM bits.
        ``x_clean``: ``[B, N]`` complex clean symbols.
        ``x_clean_tx``: ``[D, B, N]`` clean symbols repeated for each Tx.
        ``tx_ids``: ``[D]`` transmitter ids.
        ``snr_db``: ``[B]`` SNR values shared by all transmitters for each item.
        ``time_indices``: ``[D, B]`` drift time per Tx/sample pair. The default
        generator wraps sample time into a finite range and uses the same time
        for the same sample across transmitters, so only Tx hardware identity
        and its parameters differ across ``D``.
        ``y_imp``: ``[D, B, N]`` complex signal after impairments, before AWGN.
        ``y_rx``: ``[D, B, N]`` complex signal after impairments and AWGN.

    Args:
        cfg: Full experiment configuration dictionary.
        split: Split name used for future data policies; currently informational.
        device: Torch device used for generated tensors.
    """

    def __init__(self, cfg: dict[str, Any], split: str, device: torch.device | str) -> None:
        self.cfg = cfg
        self.split = split
        self.device = torch.device(device)

        signal_cfg = cfg.get("signal", {})
        data_cfg = cfg.get("data", {})
        impairment_cfg = dict(cfg.get("impairments", {}))
        impairment_cfg["sample_rate"] = float(signal_cfg.get("sample_rate", 1_000_000.0))

        self.batch_size = int(data_cfg.get("batch_size", 32))
        self.num_symbols = int(signal_cfg.get("num_symbols", signal_cfg.get("num_samples", 1024)))
        self.max_time_index = data_cfg.get("max_time_index", 100)
        self.num_tx = int(impairment_cfg.get("num_tx", 6))
        self.tx_ids = torch.arange(self.num_tx, device=self.device, dtype=torch.long)
        self.snr_db = signal_cfg.get("snr_db")
        self.snr_list = signal_cfg.get("snr_list")
        self.batch_index = 0
        self.impairment_bank = HardwareImpairmentBank(impairment_cfg, device=self.device)
        self.use_semantic = semantic_enabled(cfg)
        self.semantic_source = SemanticBatchSource(cfg, split=split, device=self.device) if self.use_semantic else None

        if self.snr_db is None and not self.snr_list:
            raise ValueError("Config must define signal.snr_db or signal.snr_list.")

    def _sample_snr(self) -> torch.Tensor:
        """Sample one SNR value per batch item, shared across transmitters."""
        if self.snr_list:
            snr_options = torch.as_tensor(self.snr_list, device=self.device, dtype=torch.float32)
            indices = torch.randint(0, snr_options.numel(), (self.batch_size,), device=self.device)
            return snr_options[indices]
        return torch.full((self.batch_size,), float(self.snr_db), device=self.device)

    def _make_time_indices(self) -> torch.Tensor:
        """Create ``[D, B]`` time indices with shared sample time across Tx."""
        start = self.batch_index * self.batch_size
        sample_times = torch.arange(
            start,
            start + self.batch_size,
            device=self.device,
            dtype=torch.float32,
        )
        if self.max_time_index is not None and float(self.max_time_index) > 0:
            sample_times = torch.remainder(sample_times, float(self.max_time_index))
        return sample_times.unsqueeze(0).expand(self.num_tx, -1).contiguous()

    def sample_batch(self) -> dict[str, torch.Tensor]:
        """Generate one paired multi-Tx batch."""
        semantic_payload: dict[str, torch.Tensor] = {}
        if self.semantic_source is not None:
            semantic_payload = self.semantic_source.sample(self.batch_size)
            x_clean = semantic_payload["x_clean"]
            bit_groups = torch.zeros(self.batch_size, self.num_symbols, 4, device=self.device, dtype=torch.long)
        else:
            bit_groups = generate_random_bits(self.batch_size, self.num_symbols, device=self.device)
            x_clean = modulate_16qam(bit_groups)
        x_clean_tx = x_clean.unsqueeze(0).expand(self.num_tx, -1, -1).contiguous()
        snr_db = self._sample_snr()
        time_indices = self._make_time_indices()
        y_imp = self.impairment_bank.apply(
            x_clean_tx,
            tx_ids=self.tx_ids,
            time_indices=time_indices,
        )
        y_rx = add_awgn(y_imp, snr_db=snr_db)
        self.batch_index += 1

        batch = {
            "bits": bit_groups.reshape(self.batch_size, self.num_symbols * 4),
            "x_clean": x_clean,
            "x_clean_tx": x_clean_tx,
            "tx_ids": self.tx_ids,
            "snr_db": snr_db,
            "time_indices": time_indices,
            "y_imp": y_imp,
            "y_rx": y_rx,
        }
        batch.update(semantic_payload)
        return batch
