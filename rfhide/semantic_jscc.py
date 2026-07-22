"""Lightweight convolutional semantic JSCC helpers.

The semantic source can be deterministic image patches, grayscale Olivetti
faces, color LFW faces, or a local image folder. The RF hiding pipeline still
sees only the normalized JSCC IQ codeword.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def semantic_enabled(cfg: dict[str, Any]) -> bool:
    """Return whether the semantic JSCC source is enabled."""
    semantic_cfg = cfg.get("semantic", {})
    signal_cfg = cfg.get("signal", {})
    return bool(semantic_cfg.get("enabled", False)) or signal_cfg.get("modulation") == "semantic_jscc"


def _semantic_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return semantic config with defaults."""
    return cfg.get("semantic", {})


def semantic_image_size(cfg: dict[str, Any]) -> int:
    """Return the configured square semantic image size."""
    return int(_semantic_cfg(cfg).get("image_size", 32))


def semantic_image_channels(cfg: dict[str, Any]) -> int:
    """Return the configured number of semantic image channels."""
    semantic_cfg = _semantic_cfg(cfg)
    if "image_channels" in semantic_cfg:
        return int(semantic_cfg["image_channels"])
    source = str(semantic_cfg.get("source", "sample_image_patches"))
    return 3 if source in {"lfw_people_color", "image_folder_color"} else 1


def resolve_semantic_checkpoint(cfg: dict[str, Any]) -> Path:
    """Resolve the semantic JSCC checkpoint path from config."""
    semantic_cfg = _semantic_cfg(cfg)
    output_dir = Path(cfg.get("experiment", {}).get("output_dir", "outputs/snr20"))
    checkpoint = semantic_cfg.get("checkpoint", output_dir / "checkpoints" / "semantic_jscc.pt")
    return Path(checkpoint)


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, N]`` complex tensors to ``[B, 2, N]`` channels."""
    if not torch.is_complex(x):
        raise TypeError("Expected a complex tensor.")
    return torch.stack([x.real, x.imag], dim=1)


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, 2, N]`` channel tensors to complex ``[B, N]``."""
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected shape [B, 2, N], got {tuple(x.shape)}")
    return torch.complex(x[:, 0], x[:, 1])


class SemanticJSCC(nn.Module):
    """Small convolutional semantic encoder, JSCC encoder, and decoder.

    Inputs and outputs are image tensors with shape
    ``[B, C, image_size, image_size]``. The channel codeword is a normalized
    complex sequence with shape ``[B, N]``.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        semantic_cfg = _semantic_cfg(cfg)
        signal_cfg = cfg.get("signal", {})
        self.num_symbols = int(signal_cfg.get("num_symbols", semantic_cfg.get("num_symbols", 1024)))
        self.image_size = semantic_image_size(cfg)
        self.image_channels = semantic_image_channels(cfg)
        if self.image_size % 8 != 0:
            raise ValueError("semantic.image_size must be divisible by 8 for the convolutional JSCC model.")
        if self.image_channels not in {1, 3}:
            raise ValueError("semantic.image_channels must be 1 or 3.")

        latent_dim = int(semantic_cfg.get("latent_dim", 128))
        base_channels = int(semantic_cfg.get("conv_base_channels", 16))
        spatial = self.image_size // 8
        encoded_channels = base_channels * 4
        encoded_dim = encoded_channels * spatial * spatial

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(self.image_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, encoded_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(encoded_channels),
            nn.GELU(),
        )
        self.semantic_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoded_dim, latent_dim),
            nn.GELU(),
        )
        self.jscc_encoder = nn.Linear(latent_dim, 2 * self.num_symbols)
        self.decoder_input = nn.Sequential(
            nn.Linear(2 * self.num_symbols, encoded_dim),
            nn.GELU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(encoded_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, self.image_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self._encoded_shape = (encoded_channels, spatial, spatial)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to unit-power complex IQ symbols."""
        expected = (self.image_channels, self.image_size, self.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected:
            raise ValueError(f"Expected images shape [B, {expected[0]}, {expected[1]}, {expected[2]}], got {tuple(images.shape)}")
        z = self.semantic_encoder(self.encoder_conv(images.float()))
        raw = self.jscc_encoder(z).reshape(images.shape[0], 2, self.num_symbols)
        x = torch.complex(raw[:, 0], raw[:, 1])
        power = x.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
        return x / torch.sqrt(power)

    def decode(self, symbols: torch.Tensor) -> torch.Tensor:
        """Decode complex IQ symbols into reconstructed image patches."""
        if not torch.is_complex(symbols):
            raise TypeError("decode expects complex symbols.")
        features = complex_to_channels(symbols).reshape(symbols.shape[0], 2 * symbols.shape[1])
        hidden = self.decoder_input(features).reshape(symbols.shape[0], *self._encoded_shape)
        return self.decoder_conv(hidden)

    def forward(self, images: torch.Tensor, received_symbols: torch.Tensor | None = None) -> torch.Tensor:
        """Encode and decode images, optionally decoding provided received IQ."""
        symbols = self.encode(images) if received_symbols is None else received_symbols
        return self.decode(symbols)


def load_semantic_model(
    cfg: dict[str, Any],
    device: torch.device | str,
    checkpoint_path: str | Path | None = None,
    require_checkpoint: bool = True,
) -> SemanticJSCC:
    """Create a semantic JSCC model and optionally load a checkpoint."""
    model = SemanticJSCC(cfg).to(device)
    path = Path(checkpoint_path) if checkpoint_path is not None else resolve_semantic_checkpoint(cfg)
    if path.exists():
        checkpoint = torch.load(path, map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
    elif require_checkpoint:
        raise FileNotFoundError(f"Semantic JSCC checkpoint not found: {path}")
    model.eval()
    return model


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to grayscale in [0, 1]."""
    image = image.astype(np.float32) / 255.0
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def _normalize_image_array(image: np.ndarray) -> np.ndarray:
    """Return a float image array clipped to [0, 1]."""
    if image.dtype.kind in {"u", "i"}:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if image.max(initial=0.0) > 1.0:
            image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _load_patch_source_images() -> list[np.ndarray]:
    """Load bundled sample images as grayscale arrays."""
    from sklearn.datasets import load_sample_images

    return [_to_grayscale(image) for image in load_sample_images().images]


def _hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """Convert HWC or HW image arrays to CHW format."""
    if image.ndim == 2:
        return image[None, :, :].astype(np.float32)
    if image.ndim != 3:
        raise ValueError(f"Expected HW or HWC image, got shape {image.shape}")
    if image.shape[-1] == 1:
        return image[..., 0][None, :, :].astype(np.float32)
    return np.transpose(image[..., :3], (2, 0, 1)).astype(np.float32)


def _resize_chw(image: np.ndarray, image_size: int) -> np.ndarray:
    """Resize a CHW image to ``image_size`` square using torch interpolate."""
    tensor = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).numpy().astype(np.float32)


def _resize_gray(image: np.ndarray, image_size: int) -> np.ndarray:
    """Resize a grayscale image to ``image_size`` square using torch interpolate."""
    tensor = torch.as_tensor(image, dtype=torch.float32).reshape(1, 1, image.shape[0], image.shape[1])
    resized = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32)


def _center_crop_square(image: np.ndarray) -> np.ndarray:
    """Center-crop an image to a square."""
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return image[top : top + side, left : left + side, ...]


def _split_indices(count: int, split: str, seed: int) -> np.ndarray:
    """Return deterministic train/val/eval indices."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(count)
    train_end = int(0.7 * count)
    val_end = int(0.85 * count)
    if split == "train":
        return indices[:train_end]
    if split in {"val", "validation"}:
        return indices[train_end:val_end]
    return indices[val_end:]


def _make_patch_dataset(
    split: str,
    image_size: int,
    sample_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic high-resolution grayscale patch samples."""
    source_images = _load_patch_source_images()
    split_offsets = {"train": 0, "val": 100_000, "validation": 100_000, "test": 200_000, "eval": 200_000}
    rng = np.random.default_rng(seed + split_offsets.get(split, 300_000))
    images: list[np.ndarray] = []
    labels: list[int] = []
    for _ in range(sample_count):
        source_id = int(rng.integers(0, len(source_images)))
        source = source_images[source_id]
        height, width = source.shape
        top = int(rng.integers(0, height - image_size + 1))
        left = int(rng.integers(0, width - image_size + 1))
        patch = source[top : top + image_size, left : left + image_size].copy()
        if rng.random() < 0.5:
            patch = np.fliplr(patch)
        if rng.random() < 0.5:
            patch = np.flipud(patch)
        low = float(np.percentile(patch, 1.0))
        high = float(np.percentile(patch, 99.0))
        if high > low + 1e-6:
            patch = np.clip((patch - low) / (high - low), 0.0, 1.0)
        images.append(patch[None, :, :].astype(np.float32))
        labels.append(source_id)
    return np.stack(images, axis=0), np.asarray(labels, dtype=np.int64)


def _make_olivetti_faces_dataset(
    split: str,
    image_size: int,
    sample_count: int,
    seed: int,
    download_if_missing: bool,
    data_home: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load sklearn Olivetti faces and return a deterministic split."""
    from sklearn.datasets import fetch_olivetti_faces

    faces = fetch_olivetti_faces(
        data_home=str(data_home) if data_home else None,
        shuffle=False,
        download_if_missing=download_if_missing,
    )
    images = faces.images.astype(np.float32)
    labels = faces.target.astype(np.int64)
    split_ids = _split_indices(images.shape[0], split, seed)
    rng = np.random.default_rng(seed + {"train": 0, "val": 10_000, "validation": 10_000, "test": 20_000, "eval": 20_000}.get(split, 30_000))
    if split_ids.size == 0:
        raise ValueError("Olivetti split is empty.")
    if sample_count <= split_ids.size:
        chosen = split_ids[:sample_count]
    else:
        chosen = rng.choice(split_ids, size=sample_count, replace=True)

    result_images: list[np.ndarray] = []
    result_labels: list[int] = []
    for index in chosen.tolist():
        image = images[index]
        if image_size != image.shape[0]:
            image = _resize_gray(image, image_size)
        if split == "train" and rng.random() < 0.5:
            image = np.fliplr(image)
        result_images.append(image[None, :, :].astype(np.float32))
        result_labels.append(int(labels[index]))
    return np.stack(result_images, axis=0), np.asarray(result_labels, dtype=np.int64)


def _make_lfw_people_color_dataset(
    split: str,
    image_size: int,
    sample_count: int,
    seed: int,
    download_if_missing: bool,
    data_home: str | Path | None,
    resize: float,
    min_faces_per_person: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load sklearn LFW people faces as deterministic RGB samples."""
    from sklearn.datasets import fetch_lfw_people

    faces = fetch_lfw_people(
        data_home=str(data_home) if data_home else None,
        resize=resize,
        min_faces_per_person=min_faces_per_person,
        color=True,
        funneled=True,
        download_if_missing=download_if_missing,
    )
    images = _normalize_image_array(faces.images)
    labels = faces.target.astype(np.int64)
    split_ids = _split_indices(images.shape[0], split, seed)
    rng = np.random.default_rng(seed + {"train": 0, "val": 10_000, "validation": 10_000, "test": 20_000, "eval": 20_000}.get(split, 30_000))
    if split_ids.size == 0:
        raise ValueError("LFW split is empty.")
    if sample_count <= split_ids.size:
        chosen = split_ids[:sample_count]
    else:
        chosen = rng.choice(split_ids, size=sample_count, replace=True)

    result_images: list[np.ndarray] = []
    result_labels: list[int] = []
    for index in chosen.tolist():
        image = _center_crop_square(images[index])
        chw = _resize_chw(_hwc_to_chw(image), image_size)
        if split == "train" and rng.random() < 0.5:
            chw = chw[:, :, ::-1].copy()
        result_images.append(chw.astype(np.float32))
        result_labels.append(int(labels[index]))
    return np.stack(result_images, axis=0), np.asarray(result_labels, dtype=np.int64)


def _read_image_file(path: Path, image_channels: int) -> np.ndarray:
    """Read an image file in [0, 1] with the requested channels."""
    import matplotlib.image as mpimg

    image = _normalize_image_array(mpimg.imread(path))
    if image.ndim == 2:
        if image_channels == 1:
            return image
        return np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] >= 3:
        return _to_grayscale(image[..., :3]) if image_channels == 1 else image[..., :3]
    channel = image[..., 0]
    return channel if image_channels == 1 else np.repeat(channel[..., None], 3, axis=-1)


def _list_image_files(folder: Path) -> list[Path]:
    """Return supported image files from a folder recursively."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in extensions)


def _make_image_folder_dataset(
    split: str,
    image_size: int,
    sample_count: int,
    seed: int,
    folder: str | Path,
    image_channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load local portrait images from a folder."""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"semantic.image_folder does not exist: {folder_path}")
    files = _list_image_files(folder_path)
    if not files:
        raise FileNotFoundError(f"No supported image files found under: {folder_path}")

    split_ids = _split_indices(len(files), split, seed)
    rng = np.random.default_rng(seed + {"train": 0, "val": 10_000, "validation": 10_000, "test": 20_000, "eval": 20_000}.get(split, 30_000))
    if sample_count <= split_ids.size:
        chosen = split_ids[:sample_count]
    else:
        chosen = rng.choice(split_ids, size=sample_count, replace=True)

    label_map = {name: idx for idx, name in enumerate(sorted({files[index].parent.name for index in split_ids.tolist()}))}
    images: list[np.ndarray] = []
    labels: list[int] = []
    for index in chosen.tolist():
        path = files[index]
        image = _read_image_file(path, image_channels)
        image = _center_crop_square(image)
        chw = _resize_chw(_hwc_to_chw(image), image_size)
        if split == "train" and rng.random() < 0.5:
            chw = chw[:, :, ::-1].copy()
        images.append(chw.astype(np.float32))
        labels.append(label_map.get(path.parent.name, 0))
    return np.stack(images, axis=0), np.asarray(labels, dtype=np.int64)


def load_semantic_split(
    cfg: dict[str, Any],
    split: str,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load the configured semantic split as image and label tensors."""
    semantic_cfg = _semantic_cfg(cfg)
    image_size = semantic_image_size(cfg)
    image_channels = semantic_image_channels(cfg)
    seed = int(cfg.get("seed", 42))
    default_counts = {"train": 4096, "val": 768, "validation": 768, "test": 768, "eval": 768}
    count_key = f"{'val' if split == 'validation' else split}_samples"
    sample_count = int(semantic_cfg.get(count_key, default_counts.get(split, 768)))
    source = str(semantic_cfg.get("source", "sample_image_patches"))
    if source == "sample_image_patches":
        images, labels = _make_patch_dataset(split, image_size, sample_count, seed)
    elif source == "olivetti_faces":
        if image_channels != 1:
            raise ValueError("olivetti_faces supports semantic.image_channels: 1 only.")
        images, labels = _make_olivetti_faces_dataset(
            split,
            image_size,
            sample_count,
            seed,
            download_if_missing=bool(semantic_cfg.get("download_if_missing", False)),
            data_home=semantic_cfg.get("data_home"),
        )
    elif source == "lfw_people_color":
        if image_channels != 3:
            raise ValueError("lfw_people_color requires semantic.image_channels: 3.")
        min_faces = semantic_cfg.get("min_faces_per_person")
        images, labels = _make_lfw_people_color_dataset(
            split,
            image_size,
            sample_count,
            seed,
            download_if_missing=bool(semantic_cfg.get("download_if_missing", False)),
            data_home=semantic_cfg.get("data_home"),
            resize=float(semantic_cfg.get("lfw_resize", 0.5)),
            min_faces_per_person=None if min_faces is None else int(min_faces),
        )
    elif source == "image_folder":
        images, labels = _make_image_folder_dataset(
            split,
            image_size,
            sample_count,
            seed,
            folder=semantic_cfg.get("image_folder", "data/faces"),
            image_channels=image_channels,
        )
    else:
        raise ValueError(f"Unsupported semantic.source: {source}")
    image_tensor = torch.as_tensor(images, device=device, dtype=torch.float32)
    label_tensor = torch.as_tensor(labels, device=device, dtype=torch.long)
    return image_tensor, label_tensor


def load_digit_split(split: str, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible alias using a minimal default semantic config."""
    cfg = {"semantic": {"image_size": 32}, "seed": 42}
    return load_semantic_split(cfg, split, device)


class SemanticBatchSource:
    """Sample semantic images and encode them as JSCC IQ sequences."""

    def __init__(self, cfg: dict[str, Any], split: str, device: torch.device | str) -> None:
        self.cfg = cfg
        self.split = split
        self.device = torch.device(device)
        semantic_split = "train" if split == "train" else "eval"
        self.images, self.labels = load_semantic_split(cfg, semantic_split, self.device)
        self.model = load_semantic_model(cfg, self.device, require_checkpoint=True)
        self.cursor = 0

    @torch.no_grad()
    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Return a semantic batch with images, labels, and complex codewords."""
        if self.split == "train":
            indices = torch.randint(0, self.images.shape[0], (batch_size,), device=self.device)
        else:
            start = self.cursor
            end = start + batch_size
            indices = torch.arange(start, end, device=self.device) % self.images.shape[0]
            self.cursor = end % self.images.shape[0]
        images = self.images[indices]
        labels = self.labels[indices]
        self.model.eval()
        x_clean = self.model.encode(images)
        return {
            "semantic_image": images,
            "semantic_label": labels,
            "x_clean": x_clean,
        }
