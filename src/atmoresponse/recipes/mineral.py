"""Fixed mineral spectral-feature scores used by the AtmoResponse examples."""

from __future__ import annotations

import numpy as np

from ._spectral import sample_linear

ALOH_LEFT_NM = 2150.0
ALOH_CENTER_NM = 2200.0
ALOH_RIGHT_NM = 2250.0


def aloh_2200_depth(reflectance, wavelengths_nm) -> np.ndarray:
    """Continuum-removed 2.20 um AlOH feature depth.

    The continuum connects fixed 2.15 and 2.25 um shoulders. The score is
    ``1 - R2200 / continuum2200``; positive values indicate an absorption.
    """

    left = sample_linear(reflectance, wavelengths_nm, ALOH_LEFT_NM)
    center = sample_linear(reflectance, wavelengths_nm, ALOH_CENTER_NM)
    right = sample_linear(reflectance, wavelengths_nm, ALOH_RIGHT_NM)
    continuum = left + (right - left) * (
        (ALOH_CENTER_NM - ALOH_LEFT_NM) / (ALOH_RIGHT_NM - ALOH_LEFT_NM)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1.0 - center / continuum
