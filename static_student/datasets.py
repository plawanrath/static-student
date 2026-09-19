"""Loaders for third-party data kept under data/raw/ (never tracked; see data/README.md for sources and hashes)."""
from __future__ import annotations

import pickle
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOGEVOL_SYSTEMS = ("spark2", "spark3", "hadoop2", "hadoop3")
_ALLOWED = {("builtins", "list"), ("collections", "defaultdict"), ("numpy", "dtype"),
            ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}


class _AllowListUnpickler(pickle.Unpickler):
    """The LOGEVOL files are pickles from a third party. Only the container and numpy scalar types they are known to
    use may be constructed; anything else aborts the load instead of importing it."""

    def find_class(self, module: str, name: str):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"refusing to import {module}.{name}")
        return super().find_class(module, name)


def load_logevol(system: str, split: str) -> dict:
    """{session id: {"label": 0|1, "templates": [...], "Content": [...]}} for one system and split
    (train | valid | test). Content is the message body; the dataset ships it without the log header."""
    if system not in LOGEVOL_SYSTEMS:
        raise ValueError(system)
    path = REPO / "data/raw/logevol/Logevol" / system / f"session_{split}.pkl"
    with open(path, "rb") as fh:
        return _AllowListUnpickler(fh).load()
