"""Tests for the Step 8 CNN Eve model and eval data loading."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from rfhide.models_eve import EveCNN
from rfhide.utils import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_eve_forward_shape() -> None:
    """EveCNN should return logits with shape [B, 3]."""
    model = EveCNN(num_classes=3)
    signals = torch.randn(5, 2, 128)

    logits = model(signals)

    assert logits.shape == (5, 3)


def test_eve_loss_backward() -> None:
    """Cross-entropy loss should backpropagate through EveCNN."""
    set_seed(501)
    model = EveCNN(num_classes=3)
    signals = torch.randn(6, 2, 128)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])

    loss = F.cross_entropy(model(signals), labels)
    loss.backward()
    grad_norm = sum(param.grad.abs().sum().item() for param in model.parameters() if param.grad is not None)

    assert torch.isfinite(loss)
    assert grad_norm > 0.0


def test_eve_extract_embedding_shape() -> None:
    """Embedding extraction should return [B, 128]."""
    model = EveCNN(num_classes=3)
    signals = torch.randn(4, 2, 256)

    embedding = model.extract_embedding(signals)

    assert embedding.shape == (4, 128)


def test_eve_small_batch_train_one_step() -> None:
    """A small training step should run without error."""
    set_seed(502)
    model = EveCNN(num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    signals = torch.randn(9, 2, 128)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])

    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(signals), labels)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_eval_signal_labels_are_valid_tx_ids() -> None:
    """Collected eval labels should be contiguous transmitter IDs."""
    for name in ["uncompensated", "random_perturb", "fixed_precomp"]:
        data = torch.load(PROJECT_ROOT / "outputs" / "snr20" / "data" / f"eval_{name}.pt", map_location="cpu")
        labels = data["labels"]
        unique = sorted(int(label) for label in labels.unique().tolist())
        assert unique == list(range(unique[-1] + 1))
        assert torch.equal(labels, data["tx_id"])
