"""Compatibility shim for warp APIs that moved between versions.

mjlab 1.2.0's sim/sim.py references `warp.context.runtime.driver_version`,
which exists in warp ≥1.14 but not in warp 1.13 (the version currently on
PyPI and installed in this venv). Importing this module BEFORE mjlab patches
the missing attribute so mjlab can read it.

Drop this import at the very top of any script that needs to use mjlab.
"""

import types
import warp as wp


def _ensure_wp_context_runtime() -> None:
    if hasattr(wp, "context") and hasattr(wp.context, "runtime"):
        return

    # Build a tiny fake `wp.context.runtime` object that exposes
    # `driver_version` as the new mjlab API expects.
    runtime = types.SimpleNamespace()
    try:
        runtime.driver_version = wp.get_cuda_driver_version()
    except Exception:
        runtime.driver_version = None

    if not hasattr(wp, "context"):
        wp.context = types.SimpleNamespace()
    wp.context.runtime = runtime


_ensure_wp_context_runtime()
