"""LUT coefficient algebra used by atmospheric-response calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorrectionCoefficients:
    """One LUT cell's radiance-inversion coefficients."""

    xa: float
    xb: float
    xc: float


def reflectance_from_radiance(
    coefficients: CorrectionCoefficients,
    radiance: float | np.ndarray,
) -> float | np.ndarray:
    """Convert at-sensor radiance to Lambertian surface reflectance."""

    y = coefficients.xa * np.asarray(radiance, dtype=float) - coefficients.xb
    reflectance = y / (1.0 + coefficients.xc * y)
    if np.isscalar(radiance):
        return float(reflectance)
    return reflectance


def radiance_from_reflectance(
    coefficients: CorrectionCoefficients,
    reflectance: float | np.ndarray,
) -> float | np.ndarray:
    """Convert Lambertian surface reflectance to at-sensor radiance."""

    rho = np.asarray(reflectance, dtype=float)
    y = rho / (1.0 - coefficients.xc * rho)
    radiance = (y + coefficients.xb) / coefficients.xa
    if np.isscalar(reflectance):
        return float(radiance)
    return radiance

