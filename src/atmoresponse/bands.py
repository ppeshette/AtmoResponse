"""Band lookup helpers."""

from __future__ import annotations

import numpy as np


def band_index(wavelengths: np.ndarray, target_nm: float) -> int:
    """Return the index of the band nearest ``target_nm``."""

    return int(np.argmin(np.abs(np.asarray(wavelengths, dtype=float) - target_nm)))

