import numpy as np
import pytest

from atmoresponse.recipes import as_algorithm, cdom_absorption
from atmoresponse.recipes.cdom import (
    ANCHORS_NM,
    BLUE1_NM,
    BLUE2_NM,
    BLUE3_NM,
    GREEN_NM,
    RED_NM,
    pure_water_absorption,
    pure_water_backscatter,
    qaa_adg,
)

WL = np.arange(410.0, 681.0, 5.0)


def _synthetic_rrs(kind: str = "clear") -> np.ndarray:
    """A smooth positive Rrs shaped like real water: a green shoulder, blue
    suppressed. 'turbid' lifts the whole curve past the red branch point."""
    base = 0.004 * np.exp(-((WL - 555.0) / 90.0) ** 2) + 0.0006
    return base + 0.006 if kind == "turbid" else base


def _independent_a_dg(Rrs: np.ndarray) -> tuple[float, float, float]:
    """A second, deliberately separate implementation of the QAA v6 a_dg(443)
    and slope path, to catch a transcription error in the module."""
    r = {nm: float(np.interp(nm, WL, Rrs)) for nm in ANCHORS_NM}
    rrs = {nm: v / (0.52 + 1.7 * v) for nm, v in r.items()}
    g0, g1 = 0.089, 0.1245
    u = {nm: (-g0 + np.sqrt(g0**2 + 4 * g1 * v)) / (2 * g1) for nm, v in rrs.items()}
    aw = {nm: float(pure_water_absorption(nm)) for nm in ANCHORS_NM}
    bbw = {nm: float(pure_water_backscatter(nm)) for nm in ANCHORS_NM}

    # 2020 IOCCG QAA_v6 note: turbid test at 670, decomposition slope over the
    # 443 to 411 nm gap. Hard-coded here so this stays a conformance check on the
    # module rather than a mirror of its constants.
    assert RED_NM == 670.0 and BLUE1_NM == 412.0
    Rrs_670 = r[RED_NM]
    assert 0.9 * r[GREEN_NM] ** 1.7 <= Rrs_670 <= 20.0 * r[GREEN_NM] ** 1.5  # guard is a no-op here
    if Rrs_670 < 0.0015:
        lam0 = 555.0
        denom = rrs[GREEN_NM] + 5 * rrs[RED_NM] ** 2 / rrs[BLUE3_NM]
        chi = np.log10((rrs[BLUE2_NM] + rrs[BLUE3_NM]) / denom)
        a0 = aw[GREEN_NM] + 10 ** (-1.146 - 1.366 * chi - 0.469 * chi**2)
    else:
        lam0 = 670.0
        a0 = aw[RED_NM] + 0.39 * (r[RED_NM] / (r[BLUE2_NM] + r[BLUE3_NM])) ** 1.14

    bbp0 = u[lam0] * a0 / (1 - u[lam0]) - bbw[lam0]
    eta = 2.0 * (1 - 1.2 * np.exp(-0.9 * rrs[BLUE2_NM] / rrs[GREEN_NM]))

    def a_total(t):
        ut = float((-g0 + np.sqrt(g0**2 + 4 * g1 * (np.interp(t, WL, Rrs) / (0.52 + 1.7 * np.interp(t, WL, Rrs))))) / (2 * g1))
        bbwt = float(pure_water_backscatter(t))
        bbpt = bbp0 * (lam0 / t) ** eta
        return (1 - ut) * (bbwt + bbpt) / ut

    a412, a443 = a_total(412.0), a_total(443.0)
    ratio = rrs[BLUE2_NM] / rrs[GREEN_NM]
    zeta = 0.74 + 0.2 / (0.8 + ratio)
    slope = 0.015 + 0.002 / (0.6 + ratio)
    xi = np.exp(slope * (443.0 - 411.0))
    a_dg = ((a412 - zeta * a443) - (aw[BLUE1_NM] - zeta * aw[BLUE2_NM])) / (xi - zeta)
    return a_dg, slope, lam0


@pytest.mark.parametrize("kind", ["clear", "turbid"])
def test_qaa_matches_an_independent_implementation(kind):
    Rrs = _synthetic_rrs(kind)
    got = qaa_adg(Rrs, WL)
    want_a_dg, want_S, want_lam0 = _independent_a_dg(Rrs)

    assert got["lambda0"] == want_lam0
    assert got["a_dg_443"] == pytest.approx(want_a_dg, rel=1e-9)
    assert got["S"] == pytest.approx(want_S, rel=1e-9)
    assert got["a_dg_443"] > 0


def test_qaa_returns_nan_when_undefined():
    out = qaa_adg(np.full_like(WL, -0.001), WL)
    assert np.isnan(out["a_dg_443"]) and np.isnan(out["S"])
    assert np.isnan(out["a_dg"]).all()


def test_guard_rrs_670_holds_within_limits_and_replaces_outside():
    from atmoresponse.recipes.cdom import guard_rrs_670

    rrs_555, rrs_490 = 0.005, 0.006
    lower, upper = 0.9 * rrs_555 ** 1.7, 20.0 * rrs_555 ** 1.5
    inside = 0.5 * (lower + upper)
    assert guard_rrs_670(inside, rrs_555, rrs_490) == inside

    estimate = 1.27 * rrs_555 ** 1.47 + 0.00018 * (rrs_490 / rrs_555) ** -3.19
    assert guard_rrs_670(upper * 100, rrs_555, rrs_490) == pytest.approx(estimate)
    assert guard_rrs_670(-0.001, rrs_555, rrs_490) == pytest.approx(estimate)
    assert guard_rrs_670(0.01, 0.0, rrs_490) == 0.01  # cannot apply, left as is


def test_guard_rrs_670_changes_the_retrieval_on_a_corrupted_red_band():
    Rrs = _synthetic_rrs("clear").copy()
    Rrs[np.argmin(np.abs(WL - 670.0))] *= 8.0  # a bad atmospheric correction
    guarded = qaa_adg(Rrs, WL)["a_dg_443"]

    # Without the guard the inflated red band would push lambda0 to the turbid
    # branch and shift a_dg; the guard pulls it back near the clean retrieval.
    clean = qaa_adg(_synthetic_rrs("clear"), WL)["a_dg_443"]
    assert abs(guarded - clean) / clean < 0.15


def test_qaa_rejects_bad_wavelength_coverage_and_shape():
    with pytest.raises(ValueError, match="same length"):
        qaa_adg(_synthetic_rrs()[:5], WL)
    with pytest.raises(ValueError, match="420 to 650"):
        qaa_adg(np.ones(6), np.array([500.0, 510.0, 520.0, 530.0, 540.0, 550.0]))


def test_a_dg_slope_extrapolation_is_exponential():
    out = qaa_adg(_synthetic_rrs(), WL, report_nm=(BLUE2_NM, BLUE3_NM, GREEN_NM))
    targets = np.array([BLUE2_NM, BLUE3_NM, GREEN_NM])
    expected = out["a_dg_443"] * np.exp(-out["S"] * (targets - BLUE2_NM))
    np.testing.assert_allclose(out["a_dg"], expected, rtol=1e-12)


def test_cdom_absorption_1d_and_2d_agree_and_apply_the_pi_conversion():
    reflectance = _synthetic_rrs() * np.pi  # cdom_absorption divides by pi internally
    a_dg_1d, a_ph_1d, s_1d = cdom_absorption(reflectance, WL)

    a_dg_2d, a_ph_2d, s_2d = cdom_absorption(np.stack([reflectance, reflectance]), WL)
    assert a_dg_2d.shape == (2,)
    assert a_dg_2d[0] == pytest.approx(a_dg_1d)
    assert s_2d[1] == pytest.approx(s_1d)

    direct = qaa_adg(_synthetic_rrs(), WL)
    assert a_dg_1d == pytest.approx(direct["a_dg_443"])


def test_as_algorithm_selects_a_dg_443():
    reflectance = _synthetic_rrs() * np.pi
    algo = as_algorithm(cdom_absorption, result_index=0)
    spectrum = dict(zip(WL, reflectance))
    assert algo(spectrum) == pytest.approx(cdom_absorption(reflectance, WL)[0])
