"""Water-index recipes for hyperspectral surface reflectance."""

from __future__ import annotations

import numpy as np

from ._spectral import nearest_reflectance

GREEN_NM = 560.0
SWIR_NM = 1600.0


def mndwi(
    reflectance,
    wavelengths_nm,
    *,
    green_nm: float = GREEN_NM,
    swir_nm: float = SWIR_NM,
) -> np.ndarray:
    """Modified Normalized Difference Water Index from nearest spectral bands."""

    green = nearest_reflectance(reflectance, wavelengths_nm, green_nm)
    swir = nearest_reflectance(reflectance, wavelengths_nm, swir_nm)
    denom = green + swir
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(denom) > 1e-6, (green - swir) / denom, np.nan)


def water_candidate(
    reflectance,
    wavelengths_nm,
    *,
    threshold: float = 0.0,
    green_nm: float = GREEN_NM,
    swir_nm: float = SWIR_NM,
) -> np.ndarray:
    """Boolean water candidate mask from MNDWI."""

    score = mndwi(reflectance, wavelengths_nm, green_nm=green_nm, swir_nm=swir_nm)
    return np.isfinite(score) & (score > threshold)
