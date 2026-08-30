"""Offline tests for the LUT consumer layer.

These build a tiny synthetic shard store with hand-computable answers rather than
the real downloadable LUT archive, so they need no network and no ``LUT_STORE``.
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

from atmoresponse import lut

# A minimal axes dict: sza has two points chosen so cos(sza) lands on round
# numbers (cos 0 = 1.0, cos 60 = 0.5); every other geometry axis is a single
# point (drops out of interpolation); aod has two points for the log/log
# discriminator; two bands so exact band selection is exercised.
AXES = {"axes": {
    "sza": {"values": [0.0, 60.0]},
    "vza": {"values": [10.0, 15.0, 20.0]},
    "raa": {"values": [90.0]},
    "aerosol": {"values": ["Maritime"]},
    "cwv": {"values": [1.0, 3.0]},
    "ozone": {"values": [0.33]},
    "aod": {"values": [0.1, 0.2]},
    "band": {"values": [0, 1], "centre_nm": [500.0, 600.0], "fwhm_nm": [5.0, 5.0]},
}}


def _write_shard(root, shard_id, key, aod_idx, band_idx, coefs, status=None):
    """Write one synthetic shard in the on-disk format ``read_shard`` expects."""
    root.mkdir(parents=True, exist_ok=True)
    if status is None:
        status = np.zeros((len(aod_idx), len(band_idx)), dtype=np.int8)
    arrays = {("path_refl" if f == "path" else f): np.asarray(coefs[f], dtype=np.float64)
              for f in lut.COEF_FIELDS}
    np.savez(
        root / f"shard_{shard_id:09d}",
        key_names=np.array(lut.KEY_AXES),
        key_idx=np.array([key[a] for a in lut.KEY_AXES], dtype=np.int16),
        aod_idx=np.asarray(aod_idx, dtype=np.int16),
        band_idx=np.asarray(band_idx, dtype=np.int16),
        status=np.asarray(status, dtype=np.int8),
        provenance=np.array(json.dumps({})),
        **arrays,
    )


# xa: 1.0 -> 4.0 across sza, so the geometric mean at the cos-midpoint is exactly
#     2.0 -- a clean discriminator between log (2.0) and linear (2.5).
# xb: -1.0 -> 1.0, crosses zero, so the per-field positivity guard falls back to
#     linear (0.0 at the midpoint) rather than fail on log(negative).
# path: an exact power law path = 10000 * aod**3 (10.0 at aod=0.1, 80.0 at
#     aod=0.2), so log(path) is exactly linear in log(aod) and log/log
#     interpolation is exact at any query point.
_COEFS_LO = {"xa": np.array([[1.0, 10.0], [1.0, 10.0]]),
             "xb": np.array([[-1.0, -1.0], [-1.0, -1.0]]),
             "xc": np.array([[0.05, 0.05], [0.05, 0.05]]),
             "path": np.array([[10.0, 10.0], [80.0, 80.0]]),
             "trans": np.array([[0.9, 0.9], [0.9, 0.9]]),
             "sphalb": np.array([[0.05, 0.05], [0.05, 0.05]])}
_COEFS_HI = {"xa": np.array([[4.0, 40.0], [4.0, 40.0]]),
             "xb": np.array([[1.0, 1.0], [1.0, 1.0]]),
             "xc": np.array([[0.05, 0.05], [0.05, 0.05]]),
             "path": np.array([[10.0, 10.0], [80.0, 80.0]]),
             "trans": np.array([[0.9, 0.9], [0.9, 0.9]]),
             "sphalb": np.array([[0.05, 0.05], [0.05, 0.05]])}


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "shards"
    key_lo = {"sza": 0, "vza": 0, "raa": 0, "aerosol": 0, "cwv": 0, "ozone": 0}
    key_hi = {**key_lo, "sza": 1}
    status_lo = np.array([[lut.STATUS_OK, lut.STATUS_OK],
                          [lut.STATUS_OK, lut.STATUS_FAILED]], dtype=np.int8)
    _write_shard(root, 0, key_lo, [0, 1], [0, 1], _COEFS_LO, status_lo)
    _write_shard(root, 1, key_hi, [0, 1], [0, 1], _COEFS_HI)
    _write_shard(root, 2, {**key_lo, "vza": 2}, [0, 1], [0, 1], _COEFS_LO, status_lo)
    _write_shard(root, 3, {**key_hi, "vza": 2}, [0, 1], [0, 1], _COEFS_HI)
    lut.clear_shard_cache()
    yield str(root)
    lut.clear_shard_cache()


def test_exact_grid_point_reproduces_stored_cell(store):
    c = lut.lookup(0.0, 10.0, 90.0, "Maritime", 1.0, 0.33, 0.1, 0, root=store, axes=AXES)
    assert c["xa"] == 1.0 and c["path"] == 10.0


def test_sza_interpolates_in_log_space(store):
    sza_mid = math.degrees(math.acos(0.75))  # midpoint of cos(0)=1, cos(60)=0.5
    c = lut.lookup(sza_mid, 10.0, 90.0, "Maritime", 1.0, 0.33, 0.1, 0, root=store, axes=AXES)
    assert abs(c["xa"] - 2.0) < 1e-9  # geometric mean, not arithmetic 2.5
    assert abs(c["xb"] - 0.0) < 1e-9  # zero-crossing field falls back to linear


def test_aod_interpolates_in_log_log_space(store):
    aod_mid = math.sqrt(0.1 * 0.2)
    c = lut.lookup(0.0, 10.0, 90.0, "Maritime", 1.0, 0.33, aod_mid, 0, root=store, axes=AXES)
    assert abs(c["path"] - 10000.0 * aod_mid ** 3) < 1e-6  # exact on the power-law fixture


def test_out_of_range_raises_rather_than_extrapolating(store):
    with pytest.raises(ValueError):
        lut.lookup(90.0, 10.0, 90.0, "Maritime", 1.0, 0.33, 0.1, 0, root=store, axes=AXES)


def test_clamp_holds_aod_below_floor_at_boundary_and_warns(store):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = lut.lookup(0.0, 10.0, 90.0, "Maritime", 1.0, 0.33, 1e-6, 0,
                       root=store, axes=AXES, clamp=True)
    held = lut.lookup(0.0, 10.0, 90.0, "Maritime", 1.0, 0.33, 0.1, 0, root=store, axes=AXES)
    assert c == held  # held at the 0.1 boundary node, not extrapolated
    assert any("outside the LUT's tabulated range" in str(w.message) for w in caught)


def test_clamp_false_still_raises_out_of_range(store):
    with pytest.raises(ValueError):
        lut.lookup(0.0, 10.0, 90.0, "Maritime", 1.0, 0.33, 5.0, 0, root=store, axes=AXES)


def test_untabulated_aerosol_raises(store):
    with pytest.raises(ValueError):
        lut.lookup(0.0, 10.0, 90.0, "Desert", 1.0, 0.33, 0.1, 0, root=store, axes=AXES)


def test_failed_cell_raises_rather_than_returning_nan(store):
    with pytest.raises(ValueError):
        lut.lookup(0.0, 10.0, 90.0, "Maritime", 1.0, 0.33, 0.2, 1, root=store, axes=AXES)


def test_sparse_vza_node_interpolates_populated_neighbours(store):
    c = lut.lookup(0.0, 15.0, 90.0, "Maritime", 1.0, 0.33, 0.1, 0, root=store, axes=AXES)
    assert c["xa"] == 1.0


def test_single_cwv_level_holds_and_warns(store):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c = lut.lookup(0.0, 10.0, 90.0, "Maritime", 3.5, 0.33, 0.1, 0, root=store, axes=AXES)
    assert c["xa"] == 1.0
    assert any("only CWV=1.0 populated" in str(w.message) for w in caught)


def test_lookup_spectrum_matches_scalar_lookup(store):
    sza_mid = math.degrees(math.acos(0.75))
    spectrum = lut.lookup_spectrum(sza_mid, 10.0, 90.0, "Maritime", 1.0, 0.33, 0.1,
                                   [0, 1], root=store, axes=AXES)
    scalar = [lut.lookup(sza_mid, 10.0, 90.0, "Maritime", 1.0, 0.33, 0.1, b,
                         root=store, axes=AXES) for b in (0, 1)]
    for field in lut.COEF_FIELDS:
        np.testing.assert_allclose(spectrum[field], [s[field] for s in scalar], atol=1e-12)


def test_fold_raa_folds_absolute_pairs_into_0_180():
    cases = [(190.0, 10.0, 180.0), (350.0, 10.0, 20.0), (10.0, 350.0, 20.0), (0.0, 0.0, 0.0)]
    for view_a, sun_a, expected in cases:
        assert abs(lut.fold_raa(view_a, sun_a) - expected) < 1e-9
    np.testing.assert_allclose(
        lut.fold_raa_array([c[0] for c in cases], [c[1] for c in cases]),
        [c[2] for c in cases],
    )


def test_correct_from_lut_round_trips_known_reflectance_at_ref_doy(store):
    rho_true = 0.05
    L = lut.radiance_from_reflectance(1.0, -1.0, 0.05, rho_true)
    month, day = lut.doy_to_month_day(lut.REF_DOY)  # Earth-Sun normalisation is identity here
    rho_back = lut.correct_from_lut(0.0, 0.0, 10.0, 90.0, month, day, "Maritime",
                                    0.1, 1.0, 0.5, L, ozone=0.33, root=store, axes=AXES)
    assert abs(rho_back - rho_true) < 1e-9


def test_batch_correct_matches_per_pixel_and_flags_gap(store):
    month, day = lut.doy_to_month_day(lut.REF_DOY)
    wl_um = np.array([0.5, 0.6])
    # Two pixels at exact grid points; the second reads band 1 at aod=0.2 where
    # the low-sza shard has a FAILED cell, so its row is all-NaN and gap is True.
    sun_z = np.array([0.0, 0.0])
    sun_a = np.array([0.0, 0.0])
    view_z = np.array([10.0, 10.0])
    view_a = np.array([90.0, 90.0])
    aod = np.array([0.1, 0.2])
    cwv = np.array([1.0, 1.0])
    L = np.array([
        lut.radiance_from_reflectance(_COEFS_LO["xa"][0], _COEFS_LO["xb"][0],
                                      _COEFS_LO["xc"][0], 0.05),
        [1.0, 1.0],
    ])
    refl, gap, clamped = lut.correct_spectrum_batch_from_lut(
        sun_z, sun_a, view_z, view_a, month, day, "Maritime", aod, cwv, wl_um, L,
        ozone=0.33, root=store, axes=AXES,
    )
    assert gap.tolist() == [False, True]
    assert clamped.tolist() == [False, False]
    np.testing.assert_allclose(refl[0], 0.05, atol=1e-9)
    assert np.all(~np.isfinite(refl[1]))


def test_batch_single_cwv_hold_is_flagged_as_clamped(store):
    month, day = lut.doy_to_month_day(lut.REF_DOY)
    wl_um = np.array([0.5, 0.6])
    n = 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, _, clamped = lut.correct_spectrum_batch_from_lut(
            np.zeros(n), np.zeros(n), np.full(n, 10.0), np.full(n, 90.0),
            month, day, "Maritime", np.full(n, 0.1),
            np.array([1.0, 2.5]),  # only CWV=1.0 is populated; 2.5 is held there
            wl_um, np.ones((n, 2)), ozone=0.33, root=store, axes=AXES,
        )
    assert clamped.tolist() == [False, True]
    assert any("only CWV=1.0 populated" in str(w.message) for w in caught)


def test_batch_clamp_flags_clamped_pixels(store):
    month, day = lut.doy_to_month_day(lut.REF_DOY)
    wl_um = np.array([0.5, 0.6])
    n = 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        refl, gap, clamped = lut.correct_spectrum_batch_from_lut(
            np.zeros(n), np.zeros(n), np.full(n, 10.0), np.full(n, 90.0),
            month, day, "Maritime",
            np.array([0.1, 1e-6]),  # second pixel's AOD is below the fixture floor
            np.ones(n), wl_um, np.ones((n, 2)),
            ozone=0.33, root=store, axes=AXES, clamp=True,
        )
    assert clamped.tolist() == [False, True]
    assert any("outside the LUT's tabulated range" in str(w.message) for w in caught)


def test_shard_keys_needed_and_install_shards_roundtrip(store):
    axes = AXES
    aero_idx = 0
    keys = lut.shard_keys_needed(store, aero_idx, "Maritime", axes,
                                 sza=np.array([0.0, 60.0]), vza=np.array([10.0, 10.0]),
                                 raa=np.array([90.0, 90.0]), cwv=np.array([1.0, 1.0]),
                                 ozone=0.33)
    assert keys  # both sza corners are touched
    shards = lut.read_shards(store, keys)
    lut.clear_shard_cache()
    lut.install_shards(store, shards)
    # A lookup now succeeds without re-globbing (cache pre-populated).
    c = lut.lookup(0.0, 10.0, 90.0, "Maritime", 1.0, 0.33, 0.1, 0, root=store, axes=axes)
    assert c["xa"] == 1.0
