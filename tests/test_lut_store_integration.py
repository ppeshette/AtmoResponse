"""Integration checks against a real Tanager LUT store.

Unlike ``test_lut_store.py`` (synthetic shards, hand-computable answers), these
run the consumer layer against the actual downloadable archive. They are gated
so the default test run stays offline:

* Set ``LUT_STORE_TANAGER`` to an already-unpacked store (the directory holding
  ``shards/``) to run without any network access.
* Otherwise set ``ATMORESPONSE_LUT_INTEGRATION=1`` to let the fixture pull the
  published archive with ``download_lut`` into a temporary cache.
* With neither set, every test here is skipped.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from atmoresponse import lut
from atmoresponse.downloads import download_lut

_ENV_STORE = "LUT_STORE_TANAGER"
_ENV_ENABLE = "ATMORESPONSE_LUT_INTEGRATION"


def _truthy(value: str | None) -> bool:
    return bool(value) and value.lower() not in {"0", "false", "no"}


@pytest.fixture(scope="module")
def shard_root(tmp_path_factory) -> str:
    """The ``shards/`` directory of a real Tanager store, or skip."""
    configured = os.environ.get(_ENV_STORE)
    if configured:
        store = Path(configured).expanduser()
        shards = store / "shards"
        if not any(shards.glob("shard_*.npz")):
            pytest.skip(f"{_ENV_STORE}={store} has no shards under {shards}")
        return str(shards)
    if _truthy(os.environ.get(_ENV_ENABLE)):
        dest = tmp_path_factory.mktemp("lut_store_tanager")
        store = download_lut("tanager", dest=dest)
        return str(store / "shards")
    pytest.skip(f"set {_ENV_STORE} or {_ENV_ENABLE} to run the real-LUT integration tests")


@pytest.fixture(autouse=True)
def _clean_cache():
    lut.clear_shard_cache()
    yield
    lut.clear_shard_cache()


@pytest.fixture(scope="module")
def sample(shard_root):
    """A query that lands exactly on one populated, solved cell of the store."""
    axes = lut.load_axes()
    catalog = lut.scan_shards(shard_root)
    assert catalog, "store scan found no shards"

    for _key_tuple, (_shard_id, path) in sorted(catalog.items()):
        shard = lut.read_shard(path)
        ok = np.argwhere(shard["status"] == lut.STATUS_OK)
        if ok.size == 0:
            continue
        row, col = (int(v) for v in ok[0])
        key = shard["key"]
        return {
            "sza": float(lut.axis_values(axes, "sza")[key["sza"]]),
            "vza": float(lut.axis_values(axes, "vza")[key["vza"]]),
            "raa": float(lut.axis_values(axes, "raa")[key["raa"]]),
            "aerosol": axes["axes"]["aerosol"]["values"][key["aerosol"]],
            "cwv": float(lut.axis_values(axes, "cwv")[key["cwv"]]),
            "ozone": float(lut.axis_values(axes, "ozone")[key["ozone"]]),
            "aod": float(lut.axis_values(axes, "aod")[int(shard["aod_idx"][row])]),
            "band": int(shard["band_idx"][col]),
            "solved_aod_idx": [int(i) for i in shard["aod_idx"]],
            "solved_status": shard["status"],
            "solved_band_idx": [int(i) for i in shard["band_idx"]],
        }
    pytest.skip("no solved cell found in the store")


def test_real_lookup_returns_physical_coefficients(shard_root, sample):
    c = lut.lookup(sample["sza"], sample["vza"], sample["raa"], sample["aerosol"],
                   sample["cwv"], sample["ozone"], sample["aod"], sample["band"],
                   root=shard_root)
    for field in lut.COEF_FIELDS:
        assert np.isfinite(c[field]), f"{field} is not finite"
    assert c["path"] >= 0.0
    assert 0.0 < c["trans"] <= 1.0
    assert 0.0 <= c["sphalb"] < 1.0
    assert c["xa"] > 0.0  # radiance -> reflectance gain


def test_lookup_spectrum_matches_scalar(shard_root, sample):
    status = sample["solved_status"]
    bands = sample["solved_band_idx"]
    # bands solved at the same aod row the scalar sample used
    aod_row = sample["solved_aod_idx"].index(
        next(i for i in sample["solved_aod_idx"]
             if lut.axis_values(lut.load_axes(), "aod")[i] == sample["aod"]))
    solved = [b for b, ok in zip(bands, status[aod_row]) if ok == lut.STATUS_OK][:8]
    assert solved

    spectrum = lut.lookup_spectrum(sample["sza"], sample["vza"], sample["raa"],
                                   sample["aerosol"], sample["cwv"], sample["ozone"],
                                   sample["aod"], solved, root=shard_root)
    for band_pos, band in enumerate(solved):
        scalar = lut.lookup(sample["sza"], sample["vza"], sample["raa"], sample["aerosol"],
                            sample["cwv"], sample["ozone"], sample["aod"], band,
                            root=shard_root)
        for field in lut.COEF_FIELDS:
            assert spectrum[field][band_pos] == pytest.approx(scalar[field], rel=1e-12)


def test_higher_aod_raises_path_reflectance(shard_root, sample):
    status = sample["solved_status"]
    aod_values = lut.axis_values(lut.load_axes(), "aod")
    band_pos = sample["solved_band_idx"].index(sample["band"])
    solved_aods = sorted(
        aod_values[idx]
        for pos, idx in enumerate(sample["solved_aod_idx"])
        if status[pos, band_pos] == lut.STATUS_OK
    )
    if len(solved_aods) < 2:
        pytest.skip("need two solved aod nodes on one band")
    low = lut.lookup(sample["sza"], sample["vza"], sample["raa"], sample["aerosol"],
                     sample["cwv"], sample["ozone"], float(solved_aods[0]), sample["band"],
                     root=shard_root)
    high = lut.lookup(sample["sza"], sample["vza"], sample["raa"], sample["aerosol"],
                      sample["cwv"], sample["ozone"], float(solved_aods[-1]), sample["band"],
                      root=shard_root)
    assert np.isfinite(low["path"]) and np.isfinite(high["path"])
    assert high["path"] > low["path"]


def test_correct_from_lut_round_trips_a_known_reflectance(shard_root, sample):
    month, day = lut.doy_to_month_day(lut.REF_DOY)
    axes = lut.load_axes()
    centre_nm, _ = lut.band_table(axes)
    wl_um = float(centre_nm[sample["band"]]) / 1000.0

    c = lut.lookup(sample["sza"], sample["vza"], sample["raa"], sample["aerosol"],
                   sample["cwv"], sample["ozone"], sample["aod"], sample["band"],
                   root=shard_root)
    rho_true = 0.1
    radiance = lut.radiance_from_reflectance(c["xa"], c["xb"], c["xc"], rho_true)

    rho_back = lut.correct_from_lut(
        sample["sza"], 0.0, sample["vza"], sample["raa"], month, day,
        sample["aerosol"], sample["aod"], sample["cwv"], wl_um, radiance,
        ozone=sample["ozone"], root=shard_root,
    )
    assert rho_back == pytest.approx(rho_true, abs=1e-9)


def test_bundled_axes_cover_the_store(shard_root):
    axes = lut.load_axes()
    centre_nm, fwhm_nm = lut.band_table(axes)
    assert len(centre_nm) == len(fwhm_nm) == len(axes["axes"]["band"]["values"])
    for _key, (_shard_id, path) in lut.scan_shards(shard_root).items():
        shard = lut.read_shard(path)
        assert int(shard["band_idx"].max()) < len(centre_nm)
        assert int(shard["aod_idx"].max()) < len(lut.axis_values(axes, "aod"))
        break
