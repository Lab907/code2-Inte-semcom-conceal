"""Deprecated compatibility entry point for the old diffusion step."""

from __future__ import annotations


def main() -> None:
    """Explain the current fixed-precomp workflow."""
    raise SystemExit(
        "Diffusion training has been removed from the active workflow. "
        "Run scripts/02_train_teacher_snr20.py to optimize fixed Tx-side "
        "pre-compensation, then scripts/05_collect_eval_signals_snr20.py "
        "to collect fixed_precomp evaluation signals."
    )


if __name__ == "__main__":
    main()
