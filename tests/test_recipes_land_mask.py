import numpy as np

from atmoresponse.recipes.land_mask import (
    RED_EDGE_CANOPY_RANGE_NM,
    canopy_mask,
    red_edge_position,
)

_WL = np.arange(650.0, 845.0, 5.0)


def _red_edge_spectrum(inflection_nm):
    return 0.03 + 0.45 / (1.0 + np.exp(-(_WL - inflection_nm) / 8.0))


def test_red_edge_position_recovers_the_inflection():
    spectra = np.stack([_red_edge_spectrum(705.0), _red_edge_spectrum(720.0)])
    rep = red_edge_position(spectra, _WL)
    np.testing.assert_allclose(rep, [705.0, 720.0], atol=3.0)


def test_red_edge_position_needs_four_window_bands():
    coarse = np.array([660.0, 700.0, 740.0, 800.0])
    try:
        red_edge_position(np.ones((2, coarse.size)), coarse)
    except ValueError as exc:
        assert "four bands" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for too few red-edge bands")


def test_canopy_mask_requires_savi_and_an_in_range_red_edge():
    lo, hi = RED_EDGE_CANOPY_RANGE_NM
    canopy = _red_edge_spectrum(0.5 * (lo + hi))
    bright_soil = 0.15 + 0.0007 * (_WL - 650.0)          # high SAVI-ish, no red edge
    late_edge = _red_edge_spectrum(hi + 25.0)             # red edge above the canopy range
    mask = canopy_mask(np.stack([canopy, bright_soil, late_edge]), _WL)
    np.testing.assert_array_equal(mask, [True, False, False])


def test_canopy_mask_fill_values_resolve_to_false():
    canopy = _red_edge_spectrum(714.0)
    filled = np.full_like(_WL, -9999.0)
    mask = canopy_mask(np.stack([canopy, filled]), _WL)
    np.testing.assert_array_equal(mask, [True, False])
