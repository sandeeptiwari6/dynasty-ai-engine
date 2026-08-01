"""
Native-threading guardrails for the agent runtime.

LightGBM, PyTorch (pulled in by sentence-transformers), and scikit-learn each
vendor their *own* copy of the OpenMP runtime (``libomp.dylib``). A single
dynasty-scout query exercises both the LightGBM forecasters (ML tools) and the
PyTorch embedding model (RAG tools), so two different OpenMP runtimes end up
live in the same process.

On macOS that combination segfaults deep inside libomp's worker-thread barrier
machinery (``__kmp_fork_barrier`` / ``__kmp_launch_worker``) the moment both
runtimes try to spin up their thread pools. Forcing OpenMP to stay
single-threaded means those conflicting worker-thread pools are never created,
which sidesteps the crash entirely. ``KMP_DUPLICATE_LIB_OK`` additionally lets
the duplicate runtimes coexist instead of aborting on load.

These variables are only honoured if they are set *before* the native libraries
are imported, which is why this module is imported at the very top of
``agent/__init__.py`` (and the entry points) — ahead of any torch/lightgbm
import. ``setdefault`` is used so an operator can still override them (e.g. on a
Linux deploy that ships a single shared libomp and wants real multithreading).

The workload — single-row gradient-boosted inference and short-text embeddings —
is not meaningfully slower single-threaded, so this is an all-upside guardrail.
"""
import os

_OPENMP_GUARDRAILS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}


def configure_native_threading() -> None:
    """Set single-threaded OpenMP env vars unless the operator already set them."""
    for var, value in _OPENMP_GUARDRAILS.items():
        os.environ.setdefault(var, value)


# Applied on import so that `import agent.runtime` (done first thing in the
# package __init__) takes effect before any native library is loaded.
configure_native_threading()
