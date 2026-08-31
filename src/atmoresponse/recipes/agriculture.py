"""Fixed vegetation reflectance algorithms used by the AtmoResponse examples."""

from __future__ import annotations

import numpy as np

from ._spectral import nearest_reflectance

R704_NM = 704.0
R815_NM = 815.0

WI_REF_NM = 865.0
WI_1240_NM = 1240.0
WI_1530_NM = 1530.0

CANOPY_RED_NM = 670.0
CANOPY_NIR_NM = 815.0
CANOPY_RED_EDGE_NM = (704.0, 740.0, 782.0)


def canopy_chlorophyll_rsi(reflectance, wavelengths_nm) -> np.ndarray:
    """Inoue et al. (2016) canopy chlorophyll ratio: ``R815 / R704``.

    The score is returned directly, without fitting a local chlorophyll-content
    regression. The caller is responsible for applying a vegetation/canopy mask.
    """

    r815 = nearest_reflectance(reflectance, wavelengths_nm, R815_NM)
    r704 = nearest_reflectance(reflectance, wavelengths_nm, R704_NM)
    with np.errstate(invalid="ignore", divide="ignore"):
        return r815 / r704


def canopy_present(
    reflectance,
    wavelengths_nm,
    *,
    savi_threshold: float = 0.25,
    soil_adjustment: float = 0.5,
) -> np.ndarray:
    """Boolean canopy-presence screen, corroborating green vegetation three ways.

    A pixel passes when the soil-adjusted vegetation index (SAVI, Huete 1988)
    exceeds ``savi_threshold``, at least one red-edge band sits above the red
    band, and the near-infrared band sits above the red band. This is a
    corroboration mask for deciding where a canopy reflectance algorithm is
    meaningful, not a land-cover classification.
    """

    red = nearest_reflectance(reflectance, wavelengths_nm, CANOPY_RED_NM)
    nir = nearest_reflectance(reflectance, wavelengths_nm, CANOPY_NIR_NM)
    red_edge = np.stack(
        [nearest_reflectance(reflectance, wavelengths_nm, nm) for nm in CANOPY_RED_EDGE_NM],
        axis=-1,
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        savi = (1.0 + soil_adjustment) * (nir - red) / (nir + red + soil_adjustment)
    passes = np.isfinite(savi) & (savi > savi_threshold)
    passes &= np.nanmax(red_edge, axis=-1) > red
    passes &= nir > red
    return passes


def vegetation_water_indices(reflectance, wavelengths_nm) -> tuple[np.ndarray, np.ndarray]:
    """Sims and Gamon (2003) satellite-adapted water indices.

    Returns ``(WI_1240, WI_1530)`` as ``R865 / R1240`` and ``R865 / R1530``.
    """

    r_ref = nearest_reflectance(reflectance, wavelengths_nm, WI_REF_NM)
    r_1240 = nearest_reflectance(reflectance, wavelengths_nm, WI_1240_NM)
    r_1530 = nearest_reflectance(reflectance, wavelengths_nm, WI_1530_NM)
    with np.errstate(invalid="ignore", divide="ignore"):
        return r_ref / r_1240, r_ref / r_1530
