"""Run the complete independent multi-SNR workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.logging_utils import get_logger
from rfhide.utils import ensure_dir

STEP_NAME = "Run complete independent multi-SNR workflow"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--config", required=True, help="Path to the multi-SNR YAML config.")
    parser.add_argument("--base-config", default=None, help="Override base single-SNR config.")
    parser.add_argument("--quick", action="store_true", help="Use tiny overrides for a fast smoke run.")
    parser.add_argument("--skip-semantic-reconstruction", action="store_true", help="Skip per-SNR semantic reconstruction.")
    return parser.parse_args()


def _resolve_project_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _snr_tag(snr_db: float) -> str:
    """Return a filesystem-safe SNR tag."""
    return str(int(snr_db)) if float(snr_db).is_integer() else str(snr_db).replace(".", "p")


def _snr_list(config: dict[str, Any]) -> list[float]:
    """Read SNR levels from the multi-SNR config."""
    values = config.get("signal", {}).get("snr_list")
    if not values:
        raise ValueError("Multi-SNR config must define signal.snr_list.")
    return [float(value) for value in values]


def _semantic_enabled(config: dict[str, Any]) -> bool:
    """Return whether the multi-SNR config uses semantic JSCC."""
    semantic_cfg = config.get("semantic", {})
    signal_cfg = config.get("signal", {})
    return bool(semantic_cfg.get("enabled", False)) or signal_cfg.get("modulation") == "semantic_jscc"


def _run_command(command: list[str], logger: Any) -> None:
    """Run one child Python step and fail fast if it fails."""
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    """Run train, evaluation, and optional semantic reconstruction."""
    args = parse_args()
    config = load_config(args.config)
    logger = get_logger("rfhide.multisnr_full")
    logger.info("Current step: %s", STEP_NAME)
    logger.info("Multi-SNR config: %s", args.config)

    train_cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "08_train_multisnr.py"), "--config", args.config]
    eval_cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "09_eval_multisnr.py"), "--config", args.config]
    if args.base_config:
        train_cmd += ["--base-config", args.base_config]
        eval_cmd += ["--base-config", args.base_config]
    if args.quick:
        train_cmd += ["--semantic-epochs", "1", "--epochs", "1", "--steps-per-epoch", "1", "--batch-size", "8"]
        eval_cmd += ["--quick", "--max-tsne-samples", "90"]

    _run_command(train_cmd, logger)
    _run_command(eval_cmd, logger)

    if _semantic_enabled(config) and not args.skip_semantic_reconstruction:
        output_root = _resolve_project_path(config.get("experiment", {}).get("output_dir", "outputs/multisnr"))
        ensure_dir(output_root / "logs")
        for snr_db in _snr_list(config):
            tag = _snr_tag(snr_db)
            config_path = output_root / "configs" / f"snr{tag}.yaml"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "11_eval_semantic_reconstruction_snr20.py"),
                "--config",
                str(config_path),
            ]
            _run_command(command, logger)

    logger.info("Complete independent multi-SNR workflow passed")


if __name__ == "__main__":
    main()
