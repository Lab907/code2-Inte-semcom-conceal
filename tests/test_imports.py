"""Import tests for the RF hiding package."""

from __future__ import annotations


def test_package_modules_import() -> None:
    """Verify that all package modules import without running future logic."""
    import rfhide.channel
    import rfhide.config
    import rfhide.dataset
    import rfhide.features
    import rfhide.fixed_precomp
    import rfhide.impairments
    import rfhide.logging_utils
    import rfhide.losses
    import rfhide.metrics
    import rfhide.models_eve
    import rfhide.models_teacher
    import rfhide.modulation
    import rfhide.semantic_jscc
    import rfhide.utils


def test_core_helpers_import() -> None:
    """Verify that Step 0 helper functions are exposed and callable enough."""
    from rfhide.config import load_config
    from rfhide.logging_utils import get_logger
    from rfhide.utils import count_parameters, ensure_dir, get_device, save_json, set_seed

    assert callable(load_config)
    assert callable(get_logger)
    assert callable(count_parameters)
    assert callable(ensure_dir)
    assert callable(get_device)
    assert callable(save_json)
    assert callable(set_seed)
