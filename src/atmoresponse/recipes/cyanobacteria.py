"""Wynne/NOAA Cyanobacteria Index spectral-shape algorithm."""

from __future__ import annotations

import numpy as np

from ._spectral import nearest_reflectance, validate_spectra

CI_LO = 665.0
CI_CENTER = 681.0
CI_HI = 709.0

PC_LO = 620.0
PC_CENTER = 665.0
PC_HI = 681.0

CI_DETECT = 0.001
CI_RISK = 0.0002


def spectral_shape(r_lo, r_center, r_hi, lo_nm, center_nm, hi_nm) -> np.ndarray:
    """Baseline-interpolated spectral shape at the center band."""

    r_lo = np.asarray(r_lo, dtype="f8")
    r_center = np.asarray(r_center, dtype="f8")
    r_hi = np.asarray(r_hi, dtype="f8")
    fraction = (center_nm - lo_nm) / (hi_nm - lo_nm)
    baseline = r_lo + (r_hi - r_lo) * fraction
    return r_center - baseline


def cyanobacteria_index(reflectance, wavelengths_nm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ci, ss_665, cyanobacteria_dominant)`` for band-last reflectance.

    ``ci`` is ``-SS(681)`` on the 665/709 nm baseline. ``ss_665`` is ``SS(665)``
    on the 620/681 nm baseline. ``cyanobacteria_dominant`` requires both scores
    to be positive.
    """

    spectra, wavelengths = validate_spectra(reflectance, wavelengths_nm)
    r620 = nearest_reflectance(spectra, wavelengths, PC_LO)
    r665 = nearest_reflectance(spectra, wavelengths, CI_LO)
    r681 = nearest_reflectance(spectra, wavelengths, CI_CENTER)
    r709 = nearest_reflectance(spectra, wavelengths, CI_HI)

    ci = -spectral_shape(r665, r681, r709, CI_LO, CI_CENTER, CI_HI)
    ss_665 = spectral_shape(r620, r665, r681, PC_LO, PC_CENTER, PC_HI)
    return ci, ss_665, (ci > 0) & (ss_665 > 0)
