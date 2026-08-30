"""Fixed vegetation reflectance algorithms used by the AtmoResponse examples."""

from __future__ import annotations

import numpy as np

from ._spectral import nearest_reflectance

R704_NM = 704.0
R815_NM = 815.0

WI_REF_NM = 865.0
WI_1240_NM = 1240.0
WI_1530_NM = 1530.0


def canopy_chlorophyll_rsi(reflectance, wavelengths_nm) -> np.ndarray:
    """Inoue et al. (2016) canopy chlorophyll ratio: ``R815 / R704``.

    The score is returned directly, without fitting a local chlorophyll-content
    regression. The caller is responsible for applying a vegetation/canopy mask.
    """

    r815 = nearest_reflectance(reflectance, wavelengths_nm, R815_NM)
    r704 = nearest_reflectance(reflectance, wavelengths_nm, R704_NM)
    with np.errstate(invalid="ignore", divide="ignore"):
        return r815 / r704


def vegetation_water_indices(reflectance, wavelengths_nm) -> tuple[np.ndarray, np.ndarray]:
    """Sims and Gamon (2003) satellite-adapted water indices.

    Returns ``(WI_1240, WI_1530)`` as ``R865 / R1240`` and ``R865 / R1530``.
    """

    r_ref = nearest_reflectance(reflectance, wavelengths_nm, WI_REF_NM)
    r_1240 = nearest_reflectance(reflectance, wavelengths_nm, WI_1240_NM)
    r_1530 = nearest_reflectance(reflectance, wavelengths_nm, WI_1530_NM)
    with np.errstate(invalid="ignore", divide="ignore"):
        return r_ref / r_1240, r_ref / r_1530
