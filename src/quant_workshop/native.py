from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer

from quant_workshop.market import FloatArray

NATIVE_DIR: Path = Path(__file__).resolve().parents[2] / "native"
LIBRARY_SUFFIX: str = (
    ".dll" if sys.platform == "win32" else (".dylib" if sys.platform == "darwin" else ".so")
)


def load_downside(library_path: Path) -> Callable[[FloatArray], float]:
    library = ctypes.CDLL(str(library_path.resolve()))
    kernel = library.downside_deviation
    kernel.argtypes = [
        ndpointer(dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED")),
        ctypes.c_size_t,
    ]
    kernel.restype = ctypes.c_double

    def downside(values: FloatArray) -> float:
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("Expected a non-empty, finite, one-dimensional array.")
        contiguous: FloatArray = np.require(values, dtype=np.float64, requirements=["C", "A"])
        return float(kernel(contiguous, contiguous.size))

    return downside
