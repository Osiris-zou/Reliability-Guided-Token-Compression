"""ACEC / ToMe-style token merging utilities.

This package keeps the import path used by the experiment scripts:

    import tome
    tome.patch.timm(model, ...)

The patch modules implement ToMe-style bipartite soft matching and the
confidence-aware edge calibration used in the paper.
"""
from . import patch
from . import merge
from . import utils

__all__ = ["patch", "merge", "utils"]
