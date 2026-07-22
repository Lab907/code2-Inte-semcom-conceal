"""Step 10 train one full SNR20-style pipeline per SNR level."""

from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.logging_utils import get_logger
from rfhide.utils import ensure_dir, save_json, set_seed

STEP_NAME = "Step 10 train repeated single-SNR pipelines"
SEMANTIC_SCRIPT = "10_train_semantic_jscc_snr20.py"
TRAIN_SCRIPTS = ["02_train_teacher_snr20.py", "03_build_compensation_dataset.py"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--config", required=True, help="Path to the multi-SNR YAML config.")
    parser.add_argument(
        "--base-config",
        default=None,
        help="Single-SNR template config. Defaults to base_config in the multi-SNR config, then configs/snr20.yaml.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override fixed precomp epochs.")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Override fixed precomp steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override teacher batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override teacher learning rate.")
    parser.add_argument("--semantic-epochs", type=int, default=None, help="Override semantic.epochs.")
    parser.add_argument("--semantic-batch-size", type=int, default=None, help="Override semantic.batch_size.")
    parser.add_argument("--semantic-lr", type=float, default=None, help="Override semantic.lr.")
    return parser.parse_args()


def _snr_tag(snr_db: float) -> str:
    """Return a filesystem-safe SNR tag."""
    return str(int(snr_db)) if float(snr_db).is_integer() else str(snr_db).replace(".", "p")


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _snr_list(config: dict[str, Any]) -> list[float]:
    """Read the requested SNR levels from the multi-SNR config."""
    values = config.get("signal", {}).get("snr_list")
    if not values:
        raise ValueError("Multi-SNR config must define signal.snr_list.")
    return [float(value) for value in values]


def _semantic_enabled(config: dict[str, Any]) -> bool:
    """Return whether a config uses semantic JSCC."""
    semantic_cfg = config.get("semantic", {})
    signal_cfg = config.get("signal", {})
    return bool(semantic_cfg.get("enabled", False)) or signal_cfg.get("modulation") == "semantic_jscc"


def _single_snr_config(template: dict[str, Any], snr_db: float, output_root: Path) -> dict[str, Any]:
    """Create a single-SNR config that follows the SNR20 template exactly."""
    cfg = deepcopy(template)
    tag = _snr_tag(snr_db)
    output_dir = output_root / f"snr{tag}"

    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = f"{cfg['experiment'].get('name', 'single_snr')}_snr{tag}"
    cfg["experiment"]["output_dir"] = str(output_dir.relative_to(PROJECT_ROOT))

    cfg.setdefault("signal", {})
    cfg["signal"]["snr_db"] = float(snr_db)
    cfg["signal"].pop("snr_list", None)

    if _semantic_enabled(cfg):
        cfg.setdefault("semantic", {})
        cfg["semantic"]["checkpoint"] = str((output_dir / "checkpoints" / "semantic_jscc.pt").relative_to(PROJECT_ROOT))

    cfg.setdefault("compensation_dataset", {})
    cfg["compensation_dataset"]["checkpoint"] = str((output_dir / "checkpoints" / "teacher_best.pt").relative_to(PROJECT_ROOT))

    cfg.setdefault("eval_signal_collection", {})
    cfg["eval_signal_collection"]["checkpoint"] = str((output_dir / "checkpoints" / "teacher_best.pt").relative_to(PROJECT_ROOT))
    return cfg


def _write_single_snr_configs(config: dict[str, Any], template: dict[str, Any]) -> list[dict[str, Any]]:
    """Write derived per-SNR configs under the multi-SNR output directory."""
    output_root = _resolve_project_path(config.get("experiment", {}).get("output_dir", "outputs/multisnr"))
    config_dir = ensure_dir(output_root / "configs")
    entries: list[dict[str, Any]] = []
    for snr_db in _snr_list(config):
        tag = _snr_tag(snr_db)
        snr_cfg = _single_snr_config(template, snr_db, output_root)
        eval_cfg = config.get("multisnr_eval", {})
        if eval_cfg:
            snr_cfg.setdefault("eval_signal_collection", {})
            snr_cfg["eval_signal_collection"]["batch_size"] = int(eval_cfg.get("batch_size", snr_cfg["eval_signal_collection"].get("batch_size", 32)))
            snr_cfg["eval_signal_collection"]["num_batches"] = int(eval_cfg.get("num_batches_per_snr", snr_cfg["eval_signal_collection"].get("num_batches", 4)))
            snr_cfg.setdefault("eve", {})
            for source_key, target_key in [
                ("train_samples_per_tx", "train_samples_per_tx"),
                ("val_samples_per_tx", "val_samples_per_tx"),
                ("test_samples_per_tx", "test_samples_per_tx"),
            ]:
                if source_key in eval_cfg:
                    snr_cfg["eve"][target_key] = int(eval_cfg[source_key])
        path = config_dir / f"snr{tag}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(snr_cfg, handle, sort_keys=False, allow_unicode=True)
        entries.append(
            {
                "snr_db": snr_db,
                "tag": tag,
                "config_path": path,
                "config": snr_cfg,
                "output_dir": _resolve_project_path(snr_cfg["experiment"]["output_dir"]),
            }
        )
    return entries


def _run_command(command: list[str], logger: Any) -> None:
    """Run one child Python step and fail fast if it fails."""
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _run_train_pipeline(entry: dict[str, Any], args: argparse.Namespace, logger: Any) -> None:
    """Run semantic JSCC, fixed precomp optimization, and dataset checks."""
    config_arg = str(entry["config_path"])
    scripts = list(TRAIN_SCRIPTS)
    if _semantic_enabled(entry["config"]):
        scripts.insert(0, SEMANTIC_SCRIPT)
    for script in scripts:
        command = [sys.executable, str(PROJECT_ROOT / "scripts" / script), "--config", config_arg]
        if script == SEMANTIC_SCRIPT:
            if args.semantic_epochs is not None:
                command += ["--epochs", str(args.semantic_epochs)]
            if args.semantic_batch_size is not None:
                command += ["--batch-size", str(args.semantic_batch_size)]
            if args.semantic_lr is not None:
                command += ["--lr", str(args.semantic_lr)]
        if script == "02_train_teacher_snr20.py":
            if args.epochs is not None:
                command += ["--epochs", str(args.epochs)]
            if args.steps_per_epoch is not None:
                command += ["--steps-per-epoch", str(args.steps_per_epoch)]
            if args.batch_size is not None:
                command += ["--batch-size", str(args.batch_size)]
            if args.lr is not None:
                command += ["--lr", str(args.lr)]
        _run_command(command, logger)


def main() -> None:
    """Train a complete independent single-SNR pipeline for each SNR level."""
    args = parse_args()
    config = load_config(args.config)
    base_config = args.base_config or config.get("base_config", "configs/snr20.yaml")
    template = load_config(base_config)
    set_seed(int(template.get("seed", config.get("seed", 42))))
    logger = get_logger("rfhide.multisnr_train")
    logger.info("Current step: %s", STEP_NAME)
    logger.info("Multi-SNR config: %s", args.config)
    logger.info("Base single-SNR template: %s", base_config)

    output_root = _resolve_project_path(config.get("experiment", {}).get("output_dir", "outputs/multisnr"))
    ensure_dir(output_root / "logs")
    entries = _write_single_snr_configs(config, template)
    logger.info("SNR list: %s", [entry["snr_db"] for entry in entries])

    for entry in entries:
        logger.info("Starting SNR=%s dB | output: %s", entry["snr_db"], entry["output_dir"])
        _run_train_pipeline(entry, args, logger)
        logger.info("Finished SNR=%s dB training pipeline", entry["snr_db"])

    save_json(
        {
            "mode": "repeat_snr20_pipeline_per_snr",
            "base_config": str(_resolve_project_path(base_config)),
            "snr_runs": [
                {
                    "snr_db": entry["snr_db"],
                    "config": str(entry["config_path"]),
                    "output_dir": str(entry["output_dir"]),
                    "semantic_checkpoint": (
                        str(entry["output_dir"] / "checkpoints" / "semantic_jscc.pt")
                        if _semantic_enabled(entry["config"])
                        else None
                    ),
                    "fixed_precomp_checkpoint": str(entry["output_dir"] / "checkpoints" / "teacher_best.pt"),
                }
                for entry in entries
            ],
        },
        output_root / "logs" / "multisnr_training_summary.json",
    )
    logger.info("Step 10 repeated single-SNR training passed")


if __name__ == "__main__":
    main()
