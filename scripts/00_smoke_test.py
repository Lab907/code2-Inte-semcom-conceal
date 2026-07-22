"""Step 0 smoke test for the RF diffusion hiding project skeleton."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rfhide.config import load_config
from rfhide.logging_utils import get_logger
from rfhide.utils import ensure_dir, get_device, set_seed


STEP_NAME = "Step 0 smoke test"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=STEP_NAME)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main() -> None:
    """Run the Step 0 smoke test."""
    args = parse_args()
    logger = get_logger("rfhide.smoke_test")
    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    prefer_cuda = bool(config.get("device", {}).get("prefer_cuda", True))
    output_dir = ensure_dir(PROJECT_ROOT / config.get("experiment", {}).get("output_dir", "outputs/smoke"))

    set_seed(seed)
    device = get_device(prefer_cuda=prefer_cuda)

    logger.info("Current step: %s", STEP_NAME)
    logger.info("Config path: %s", args.config)
    logger.info("Device: %s", device)
    logger.info("Output directory: %s", output_dir)
    logger.info("Step 0 smoke test passed")


if __name__ == "__main__":
    main()

