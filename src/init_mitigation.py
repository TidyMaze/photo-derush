"""Early initialization helpers to protect GUI processes from native-threading crashes.

This module is designed to be imported as early as possible by entry scripts
so it can set environment variables and enable Python-level faulthandler
before any native C extensions (PyTorch, MKL, OpenMP) are initialized.
"""

from __future__ import annotations

import logging
import os

# Conservative defaults to avoid OpenMP/MKL spawning many threads from GUI process
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_BLOCKTIME", "1")

# Try to reduce verbose Qt logging early
os.environ.setdefault("QT_LOGGING_RULES", "qt.*=false")

# Enable faulthandler early so Python-level tracebacks are written to stderr/logs
try:
    import faulthandler

    faulthandler.enable()
except Exception:
    # best-effort; don't hard-fail if unavailable
    pass

# Best-effort: if PyTorch is importable, limit its internal thread pools
try:
    import torch

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        # some PyTorch builds don't expose these; ignore if missing
        pass
except Exception:
    # torch not installed; nothing to do
    pass

# Ensure logging exists for downstream modules
try:
    logging.getLogger("init_mitigation").debug("init_mitigation applied: OMP/torch thread limits and faulthandler")
except Exception:
    pass
