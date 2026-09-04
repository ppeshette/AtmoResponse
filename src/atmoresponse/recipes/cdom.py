"""QAA v6 semi-analytical retrieval of CDOM-plus-detritus absorption.

Reference: Lee, Z., Carder, K.L., Arnone, R.A. (2002), "Deriving inherent optical
properties from water color: a multiband quasi-analytical algorithm for optically
deep waters", Applied Optics 41(27), 5755-5772. Constants and formulas follow the
2020 IOCCG QAA_v6 note (``qaa_v6_202011``). Pure-water absorption is Pope & Fry
(1997) and pure-water backscatter is the Morel (1974) power law.

QAA is a spectral inversion rather than a blue-green band ratio. It uses the
u to a to b_b relationships and the rrs(443)/rrs(555) ratio to split total
absorption into a phytoplankton part and a colored-dissolved-and-detrital part,
then extrapolates particulate backscatter spectrally. The dissolved-and-detrital
absorption at 443 nm, ``a_dg(443)``, and its spectral slope ``S`` are the
retrieval outputs of interest for atmospheric sensitivity, because a wrong blue
reflectance moves absorption between the two terms rather than losing it.

The retrieval:

1. Rrs to sub-surface rrs:  ``rrs = Rrs / (0.52 + 1.7 Rrs)``.
2. rrs to u:  ``u = (-g0 + sqrt(g0**2 + 4 g1 rrs)) / (2 g1)``.
3. Guard Rrs(670): imperfect atmospheric correction can corrupt it, so it is
   held within ``[0.9 Rrs(555)**1.7, 20 Rrs(555)**1.5]`` and, if outside, is
   replaced by ``1.27 Rrs(555)**1.47 + 0.00018 (Rrs(490) / Rrs(555))**-3.19``.
4. ``a(lambda0)``: for non-turbid water (small Rrs at 670 nm) ``lambda0`` is
   555 nm and ``a(555)`` is an empirical function of a blue-green ratio,
   otherwise ``lambda0`` is 670 nm with the red-band form.
5. ``bbp(lambda0) = u(lambda0) a(lambda0) / (1 - u(lambda0)) - bbw(lambda0)``.
6. ``eta = 2.0 (1 - 1.2 exp(-0.9 rrs(443) / rrs(555)))``.
7. ``bbp(lambda) = bbp(lambda0) (lambda0 / lambda) ** eta``.
8. ``a(lambda) = (1 - u(lambda)) (bbw(lambda) + bbp(lambda)) / u(lambda)``.
9. ``a_dg(443) = [(a(412) - zeta a(443)) - (aw(412) - zeta aw(443))] / (xi - zeta)``
   with ``zeta = 0.74 + 0.2 / (0.8 + r)``, ``S = 0.015 + 0.002 / (0.6 + r)``,
   ``xi = exp(S (443 - 411))``, and ``r = rrs(443) / rrs(555)``.

A wrong pure-water constant biases the absolute retrieval but largely cancels in
a Realized Sensitivity difference, the same way the reconstruction offset does.
"""
from __future__ import annotations

import numpy as np

# QAA v6 anchor wavelengths in nanometres.
BLUE1_NM = 412.0
BLUE2_NM = 443.0
BLUE3_NM = 490.0
GREEN_NM = 555.0
RED_NM = 670.0
ANCHORS_NM = (BLUE1_NM, BLUE2_NM, BLUE3_NM, GREEN_NM, RED_NM)

# The a_dg slope term (step 9) uses the 443 and 411 nm gap, per the 2020 note.
ADG_SLOPE_LO_NM = 411.0
ADG_SLOPE_HI_NM = 443.0

_G0 = 0.089
_G1 = 0.1245

# Pope & Fry (1997) pure-water absorption, m^-1, 400-700 nm at 10 nm spacing
# (Sogandares & Fry 1997 below 440 nm). np.interp clamps at the ends.
_AW_NM = np.arange(400.0, 701.0, 10.0)
_AW = np.array([
    0.00663, 0.00473, 0.00454, 0.00478, 0.00635, 0.00922, 0.00979, 0.01060,
    0.01270, 0.01500, 0.02040, 0.03250, 0.04090, 0.04340, 0.04740, 0.05650,
    0.06190, 0.06950, 0.08960, 0.13510, 0.22240, 0.26440, 0.27550, 0.29160,
    0.31080, 0.34000, 0.41000, 0.43900, 0.46500, 0.51600, 0.62400,
])


def pure_water_absorption(wavelengths_nm) -> np.ndarray:
    """``aw(lambda)`` in m^-1, Pope & Fry (1997), linearly interpolated."""
    return np.interp(np.asarray(wavelengths_nm, dtype=float), _AW_NM, _AW)


def pure_water_backscatter(wavelengths_nm) -> np.ndarray:
    """``bbw(lambda)`` in m^-1, the Morel (1974) power law, half the seawater
    scattering coefficient."""
    wl = np.asarray(wavelengths_nm, dtype=float)
    return 0.00144 * (500.0 / wl) ** 4.32


def _rrs_below(Rrs):
    return Rrs / (0.52 + 1.7 * Rrs)


def guard_rrs_670(rrs_670: float, rrs_555: float, rrs_490: float) -> float:
    """Hold Rrs(670) within the QAA v6 limits, or replace it.

    Imperfect atmospheric correction can drive the measured red reflectance
    outside a physically plausible range (2020 IOCCG note, Eqs 7 to 9). Values
    below ``0.9 Rrs(555)**1.7`` or above ``20 Rrs(555)**1.5``, or non-positive,
    are replaced by the empirical estimate
    ``1.27 Rrs(555)**1.47 + 0.00018 (Rrs(490) / Rrs(555))**-3.19``.
    """
    if rrs_555 <= 0.0 or rrs_490 <= 0.0:
        return rrs_670
    lower = 0.9 * rrs_555 ** 1.7
    upper = 20.0 * rrs_555 ** 1.5
    if lower <= rrs_670 <= upper:
        return rrs_670
    return 1.27 * rrs_555 ** 1.47 + 0.00018 * (rrs_490 / rrs_555) ** -3.19


def _u(rrs):
    return (-_G0 + np.sqrt(_G0 ** 2 + 4.0 * _G1 * rrs)) / (2.0 * _G1)


def qaa_adg(Rrs, wavelengths_nm, *, report_nm=(BLUE2_NM,)):
    """QAA v6 for one spectrum.

    ``Rrs`` and ``wavelengths_nm`` are 1-D arrays of the same length, with
    ``wavelengths_nm`` finite, strictly increasing, and covering at least
    420 to 650 nm so the interior QAA anchors are bracketed.

    Returns a dict with scalars ``a_dg_443``, ``a_ph_443``, ``S`` (the a_dg
    slope), ``eta``, ``lambda0``, and arrays ``a_dg``, ``a_ph``, ``a``, and
    ``bbp`` at ``report_nm``. Where the inversion is undefined (non-positive
    sub-surface rrs at an anchor, non-positive particulate backscatter at
    lambda0, or a degenerate slope) the scalars are NaN and the arrays are
    filled with NaN, rather than raising.
    """
    Rrs = np.asarray(Rrs, dtype=float)
    wl = np.asarray(wavelengths_nm, dtype=float)
    if Rrs.ndim != 1 or wl.ndim != 1 or Rrs.shape != wl.shape:
        raise ValueError("Rrs and wavelengths_nm must be 1-D arrays of the same length")
    if not np.isfinite(wl).all() or np.any(np.diff(wl) <= 0):
        raise ValueError("wavelengths_nm must be finite and strictly increasing")
    if wl.min() > 420.0 or wl.max() < 650.0:
        raise ValueError("wavelengths_nm must cover at least 420 to 650 nm for the QAA anchors")

    nan = {"a_dg_443": np.nan, "a_ph_443": np.nan, "S": np.nan, "eta": np.nan,
           "lambda0": np.nan}
    for name in ("a_dg", "a_ph", "a", "bbp"):
        nan[name] = np.full(len(report_nm), np.nan)

    def at(target):
        return float(np.interp(target, wl, Rrs))

    Rrs_a = {nm: at(nm) for nm in ANCHORS_NM}
    Rrs_a[RED_NM] = guard_rrs_670(Rrs_a[RED_NM], Rrs_a[GREEN_NM], Rrs_a[BLUE3_NM])
    rrs_a = {nm: _rrs_below(v) for nm, v in Rrs_a.items()}
    if any(v <= 0 for v in rrs_a.values()):
        return nan
    u_a = {nm: _u(v) for nm, v in rrs_a.items()}

    aw_a = {nm: float(pure_water_absorption(nm)) for nm in ANCHORS_NM}
    bbw_a = {nm: float(pure_water_backscatter(nm)) for nm in ANCHORS_NM}

    if Rrs_a[RED_NM] < 0.0015:
        lam0 = GREEN_NM
        denom = rrs_a[GREEN_NM] + 5.0 * rrs_a[RED_NM] ** 2 / rrs_a[BLUE3_NM]
        chi = np.log10((rrs_a[BLUE2_NM] + rrs_a[BLUE3_NM]) / denom)
        a_lam0 = aw_a[GREEN_NM] + 10.0 ** (-1.146 - 1.366 * chi - 0.469 * chi ** 2)
    else:
        lam0 = RED_NM
        a_lam0 = aw_a[RED_NM] + 0.39 * (
            Rrs_a[RED_NM] / (Rrs_a[BLUE2_NM] + Rrs_a[BLUE3_NM])) ** 1.14

    bbp_lam0 = u_a[lam0] * a_lam0 / (1.0 - u_a[lam0]) - bbw_a[lam0]
    if not np.isfinite(bbp_lam0) or bbp_lam0 <= 0:
        return nan
    eta = 2.0 * (1.0 - 1.2 * np.exp(-0.9 * rrs_a[BLUE2_NM] / rrs_a[GREEN_NM]))

    def a_total(target):
        u_t = _u(_rrs_below(at(target)))
        bbw_t = float(pure_water_backscatter(target))
        bbp_t = bbp_lam0 * (lam0 / target) ** eta
        return (1.0 - u_t) * (bbw_t + bbp_t) / u_t, bbp_t

    a_blue1, _ = a_total(BLUE1_NM)
    a_443, _ = a_total(BLUE2_NM)

    r = rrs_a[BLUE2_NM] / rrs_a[GREEN_NM]
    zeta = 0.74 + 0.2 / (0.8 + r)
    S = 0.015 + 0.002 / (0.6 + r)
    xi = np.exp(S * (ADG_SLOPE_HI_NM - ADG_SLOPE_LO_NM))
    if np.isclose(xi, zeta):
        return nan
    a_dg_443 = ((a_blue1 - zeta * a_443) - (aw_a[BLUE1_NM] - zeta * aw_a[BLUE2_NM])) / (xi - zeta)
    a_ph_443 = a_443 - aw_a[BLUE2_NM] - a_dg_443

    report = np.asarray(report_nm, dtype=float)
    a_arr = np.array([a_total(t)[0] for t in report])
    bbp_arr = np.array([a_total(t)[1] for t in report])
    a_dg_arr = a_dg_443 * np.exp(-S * (report - BLUE2_NM))
    a_ph_arr = a_arr - pure_water_absorption(report) - a_dg_arr

    return {"a_dg_443": float(a_dg_443), "a_ph_443": float(a_ph_443), "S": float(S),
            "eta": float(eta), "lambda0": float(lam0),
            "a_dg": a_dg_arr, "a_ph": a_ph_arr, "a": a_arr, "bbp": bbp_arr}


def cdom_absorption(reflectance, wavelengths_nm):
    """Return ``(a_dg_443, a_ph_443, S)`` from QAA v6 for band-last water
    reflectance.

    Tanager ships water-leaving reflectance, and QAA wants remote-sensing
    reflectance ``Rrs = reflectance / pi``. That conversion is applied here.
    A 1-D spectrum returns three scalars. A 2-D ``(n_pixels, n_bands)`` array
    returns three ``(n_pixels,)`` arrays, with QAA run per pixel.
    """
    spectra = np.asarray(reflectance, dtype="f8")
    wavelengths = np.asarray(wavelengths_nm, dtype="f8")
    if spectra.shape[-1] != wavelengths.shape[0]:
        raise ValueError("reflectance must have wavelengths_nm along its final dimension")

    if spectra.ndim == 1:
        out = qaa_adg(spectra / np.pi, wavelengths)
        return out["a_dg_443"], out["a_ph_443"], out["S"]

    rows = spectra.reshape(-1, spectra.shape[-1])
    a_dg = np.empty(len(rows))
    a_ph = np.empty(len(rows))
    slope = np.empty(len(rows))
    for i, row in enumerate(rows):
        out = qaa_adg(row / np.pi, wavelengths)
        a_dg[i], a_ph[i], slope[i] = out["a_dg_443"], out["a_ph_443"], out["S"]
    shape = spectra.shape[:-1]
    return a_dg.reshape(shape), a_ph.reshape(shape), slope.reshape(shape)
