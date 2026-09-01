"""Canopy corroboration mask for the Rajanpur canopy-chlorophyll example: a pixel
counts as canopy only when a broadband vegetation index and an HSI-native red-edge
signal agree.

MSI side: SAVI (Huete 1988), the soil-brightness-corrected NDVI form, chosen over
plain NDVI because it targets the failure mode Inoue's own PROSAIL simulation
flagged (R815 is soil-background sensitive). ``SAVI = [(NIR - red) / (NIR + red + L)]
x (1 + L)``, L = 0.5 (Huete's general-purpose default). Red and NIR follow the
Landsat TM convention.

HSI side: Dawson and Curran (1998) three-point Lagrangian red edge position (REP).
The paper confines the red edge to 680 to 740 nm, so the first-derivative search is
restricted to that window. The paper reports no numeric REP range separating
vegetated from non-vegetated pixels, so ``RED_EDGE_CANOPY_RANGE_NM`` is a
conventional healthy-canopy range from the broader red-edge literature.

Pure numpy. ``canopy_present`` in :mod:`atmoresponse.recipes.agriculture` is a
lighter, unrelated screen; this is the one the walkthrough uses.
"""

from __future__ import annotations

import numpy as np

from ._spectral import FILL_LIMIT, nearest_reflectance, validate_spectra

CANOPY_RED_NM = 660.0
CANOPY_NIR_NM = 830.0
SAVI_L = 0.5
SAVI_CANOPY_MIN = 0.2                      # conventional vegetated-pixel floor, not from Huete
RED_EDGE_WINDOW_NM = (680.0, 740.0)        # Dawson and Curran (1998) red-edge region
RED_EDGE_CANOPY_RANGE_NM = (700.0, 728.0)  # conventional healthy-canopy REP range


def red_edge_position(reflectance, wavelengths_nm) -> np.ndarray:
    """Dawson and Curran (1998) three-point Lagrangian red edge position, in nm.

    Restricts the first-derivative spectrum to the paper's own 680 to 740 nm
    region, then fits the Lagrangian parabola (their Eqs. 2 to 4) around that
    window's interior band of maximum derivative. NaN where the window holds fewer
    than four bands or an input reflectance is a fill value. ``reflectance`` is
    band-last.
    """

    spectra, wavelengths = validate_spectra(reflectance, wavelengths_nm)
    lo, hi = RED_EDGE_WINDOW_NM
    inside = np.flatnonzero((wavelengths >= lo) & (wavelengths <= hi))
    if inside.size < 4:
        raise ValueError("fewer than four bands inside the 680 to 740 nm red-edge region")
    region = spectra[..., inside[0]:inside[-1] + 1]
    region = np.where(region <= FILL_LIMIT, np.nan, region)
    wl_region = wavelengths[inside]
    d_wl = (wl_region[1:] + wl_region[:-1]) / 2.0
    with np.errstate(invalid="ignore"):
        deriv = np.diff(region, axis=-1) / np.diff(wl_region)
    if deriv.shape[-1] < 3:
        raise ValueError("fewer than three derivative points inside the red-edge region")

    # The window's interior band of maximum first derivative. All-NaN columns
    # (nodata border pixels) are substituted with -inf so argmax returns per
    # column; the isfinite check below still resolves them to NaN.
    interior = np.where(np.isnan(deriv[..., 1:-1]), -np.inf, deriv[..., 1:-1])
    i = 1 + np.argmax(interior, axis=-1)

    def deriv_at(offset):
        return np.take_along_axis(deriv, (i + offset)[..., None], axis=-1)[..., 0]

    d_im1, d_i, d_ip1 = deriv_at(-1), deriv_at(0), deriv_at(1)
    l_im1, l_i, l_ip1 = d_wl[i - 1], d_wl[i], d_wl[i + 1]
    a = d_im1 * (l_im1 - l_i) * (l_im1 - l_ip1)
    b = d_i * (l_i - l_im1) * (l_i - l_ip1)
    c = d_ip1 * (l_ip1 - l_im1) * (l_ip1 - l_i)
    with np.errstate(invalid="ignore", divide="ignore"):
        rep = (a * (l_i + l_ip1) + b * (l_im1 + l_ip1) + c * (l_im1 + l_i)) / (2.0 * (a + b + c))
    valid = np.isfinite(d_im1) & np.isfinite(d_i) & np.isfinite(d_ip1)
    return np.where(valid, rep, np.nan)


def canopy_mask(reflectance, wavelengths_nm, *, savi_threshold: float = SAVI_CANOPY_MIN) -> np.ndarray:
    """Canopy corroboration mask: ``SAVI > savi_threshold`` AND the red edge
    position inside the healthy-canopy range. ``reflectance`` is band-last and
    must carry enough bands across 680 to 740 nm for the red edge fit."""

    red = nearest_reflectance(reflectance, wavelengths_nm, CANOPY_RED_NM)
    nir = nearest_reflectance(reflectance, wavelengths_nm, CANOPY_NIR_NM)
    with np.errstate(invalid="ignore", divide="ignore"):
        savi = ((nir - red) / (nir + red + SAVI_L)) * (1.0 + SAVI_L)
    rep = red_edge_position(reflectance, wavelengths_nm)
    lo, hi = RED_EDGE_CANOPY_RANGE_NM
    savi_ok = np.isfinite(savi) & (savi > savi_threshold)
    rep_ok = np.isfinite(rep) & (rep > lo) & (rep < hi)
    return savi_ok & rep_ok
