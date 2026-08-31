"""Unit tests for the sensitivity engine.

These inject a fake ``correct`` and build a minimal synthetic HDF5 scene, so they
need no LUT shard store. The real-LUT integration path is covered separately.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest

from atmoresponse import emit, sensitivity
from atmoresponse.sensitivity import (
    LabeledScore,
    reconstruction_gap,
    run_emit,
    run_tanager,
    variance_fraction,
)
from atmoresponse.tanager_ortho import GRID


def _write_synthetic_scene(cache_root, scene_id, shipped_aod, cwv, radiance_by_pixel):
    """A minimal 1xN HDF5 scene pair with exactly the fields ``run_tanager()`` reads.

    ``radiance_by_pixel`` is (N, 2): N pixels, 2 bands at 700/800 nm.
    """
    n = len(shipped_aod)
    scene_dir = cache_root / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    wl = np.array([700.0, 800.0])
    with h5py.File(scene_dir / f"{scene_id}_ortho_sr.h5", "w") as sr:
        sr.create_dataset(GRID + "aerosol_optical_depth", data=np.asarray(shipped_aod).reshape(1, n))
        sr.create_dataset(GRID + "column_water_vapour", data=np.asarray(cwv).reshape(1, n))
    with h5py.File(scene_dir / f"{scene_id}_ortho_radiance.h5", "w") as l1:
        cube = np.moveaxis(np.asarray(radiance_by_pixel).reshape(1, n, 2), -1, 0)  # (band, row, col)
        ds = l1.create_dataset(GRID + "toa_radiance", data=cube)
        ds.attrs["wavelengths"] = wl
        zeros = np.zeros((1, n))
        for field in ("sun_zenith", "sun_azimuth", "sensor_zenith", "sensor_azimuth"):
            l1.create_dataset(GRID + field, data=zeros)


def _full_mask(sr, aoi):
    return np.ones((aoi[1] - aoi[0], aoi[3] - aoi[2]), dtype=bool)


def _fake_correct(*, aot550, L_obs, **_):
    """Toy correction: subtract the AOD from every band's radiance."""
    return np.asarray(L_obs, dtype=float) - aot550


def _sum_algorithm(reflectance):
    return sum(reflectance.values())


# Module-level picklable callables for the workers>1 (spawn) test.
class _PicklableAlgo:
    def __call__(self, reflectance):
        return sum(reflectance.values())


def _picklable_correct(*, aot550, L_obs, **_):
    return np.asarray(L_obs, dtype=float) - aot550


def test_run_continuous_realized_sensitivity_delta(tmp_path):
    scene_id = "20250101_000000_00_0000"
    shipped_aod = np.array([0.10, 0.30, 0.50])
    radiance = np.ones((3, 2))
    _write_synthetic_scene(tmp_path, scene_id, shipped_aod, np.ones(3), radiance)

    result = run_tanager(scene_id, (0, 1, 0, 3), _full_mask, [700.0, 800.0], 0.20, "Continental",
                 algorithm=_sum_algorithm, cache=tmp_path, correct=_fake_correct)

    want_delta = -(shipped_aod - 0.20) * 2  # 2 bands, each contributes -aod550
    np.testing.assert_allclose(result.delta, want_delta)
    assert result.class_changed is None
    assert len(result.rows) == 3
    assert not result.clamped.any()  # injected correct never clamps


def test_run_reads_aoi_block_then_masks(tmp_path):
    scene_id = "20250101_000000_00_0001"
    _write_synthetic_scene(tmp_path, scene_id, np.array([0.1, 0.3, 0.5]), np.ones(3),
                           np.array([[1.0, 2.0], [9.0, 9.0], [3.0, 4.0]]))
    mask = lambda sr, aoi: np.array([[True, False, True]])

    result = run_tanager(scene_id, (0, 1, 0, 3), mask, [700.0, 800.0], 0.2, "Continental",
                 algorithm=_sum_algorithm, cache=tmp_path, correct=_fake_correct)

    assert np.array_equal(result.rows, [0, 0])
    assert np.array_equal(result.cols, [0, 2])


def test_run_classification_and_grouping(tmp_path):
    scene_id = "20250101_000000_00_0002"
    # shipped_aod[1] sits above the classifier's 0.5 threshold, reference below it:
    # pixel 1 crosses high<->low, pixels 0/2 do not.
    shipped_aod = np.array([0.05, 0.60, 0.05])
    _write_synthetic_scene(tmp_path, scene_id, shipped_aod, np.ones(3), np.ones((3, 2)))

    def classify(reflectance):
        value = min(reflectance.values())
        return LabeledScore(value=value, label="low" if value > 0.5 else "high")

    raw = run_tanager(scene_id, (0, 1, 0, 3), _full_mask, [700.0, 800.0], 0.10, "Continental",
              algorithm=classify, cache=tmp_path, correct=_fake_correct)
    grouped = run_tanager(scene_id, (0, 1, 0, 3), _full_mask, [700.0, 800.0], 0.10, "Continental",
                  algorithm=classify, cache=tmp_path, correct=_fake_correct,
                  group_labels={"low": "same", "high": "same"})

    assert list(raw.class_changed) == [False, True, False]
    assert not grouped.class_changed.any()


def test_run_fit_derives_algorithm_from_radiance(tmp_path):
    scene_id = "20250101_000000_00_0003"
    _write_synthetic_scene(tmp_path, scene_id, np.array([0.1, 0.3, 0.5]), np.ones(3),
                           np.ones((3, 2)))
    seen = {}

    def fit(wl_nm, radiance):
        seen["radiance_shape"] = radiance.shape
        return _sum_algorithm

    result = run_tanager(scene_id, (0, 1, 0, 3), _full_mask, [700.0, 800.0], 0.2, "Continental",
                 fit=fit, cache=tmp_path, correct=_fake_correct)

    assert seen["radiance_shape"] == (3, 2)  # fit sees the raw radiance population
    assert len(result.delta) == 3


def test_run_requires_exactly_one_of_algorithm_or_fit(tmp_path):
    scene_id = "20250101_000000_00_0004"
    _write_synthetic_scene(tmp_path, scene_id, np.array([0.1]), np.ones(1), np.ones((1, 2)))
    with pytest.raises(ValueError, match="exactly one"):
        run_tanager(scene_id, (0, 1, 0, 1), _full_mask, [700.0, 800.0], 0.2, "Continental",
            cache=tmp_path, correct=_fake_correct)


def test_scored_slices_without_recomputing(tmp_path):
    scene_id = "20250101_000000_00_0005"
    shipped_aod = np.array([0.10, 0.30, 0.50])
    _write_synthetic_scene(tmp_path, scene_id, shipped_aod, np.ones(3), np.ones((3, 2)))
    result = run_tanager(scene_id, (0, 1, 0, 3), _full_mask, [700.0, 800.0], 0.2, "Continental",
                 algorithm=_sum_algorithm, cache=tmp_path, correct=_fake_correct)

    region = np.zeros(result.shape, dtype=bool)
    region[0, 1] = True
    sliced = result.scored(region)

    assert len(sliced.rows) == 1 and sliced.cols[0] == 1
    assert np.isclose(sliced.delta[0], result.delta[1])
    assert sliced.cwv_g_cm2.shape == (1,)


def test_value_map_and_delta_map(tmp_path):
    scene_id = "20250101_000000_00_0006"
    _write_synthetic_scene(tmp_path, scene_id, np.array([0.1, 0.3, 0.5]), np.ones(3),
                           np.ones((3, 2)))
    result = run_tanager(scene_id, (0, 1, 0, 3), _full_mask, [700.0, 800.0], 0.2, "Continental",
                 algorithm=_sum_algorithm, cache=tmp_path, correct=_fake_correct)

    np.testing.assert_array_equal(result.value_map(result.delta), result.delta_map())
    shipped_map = result.value_map(result.at_shipped)
    np.testing.assert_allclose(shipped_map[0, result.cols], result.at_shipped)


def test_run_workers_matches_serial(tmp_path):
    scene_id = "20250101_000000_00_0007"
    _write_synthetic_scene(tmp_path, scene_id, np.array([0.10, 0.30, 0.50]), np.ones(3),
                           np.ones((3, 2)))
    common = dict(algorithm=_PicklableAlgo(), cache=tmp_path, correct=_picklable_correct,
                  chunksize=2)
    serial = run_tanager(scene_id, (0, 1, 0, 3), _full_mask, [700.0, 800.0], 0.2, "Continental",
                 **common)
    parallel = run_tanager(scene_id, (0, 1, 0, 3), _full_mask, [700.0, 800.0], 0.2, "Continental",
                   workers=2, **common)

    np.testing.assert_allclose(serial.delta, parallel.delta)
    np.testing.assert_allclose(serial.at_shipped, parallel.at_shipped)


def test_variance_fraction_additivity_guard():
    rng = np.random.default_rng(0)
    at_reference = rng.normal(0.3, 0.05, 500)
    delta = rng.normal(0.0, 0.01, 500)  # independent of at_reference
    vf = variance_fraction(at_reference, delta)
    assert vf.reliable
    assert 0.0 <= vf.atmosphere_fraction <= 1.0

    # A raw-ratio blow-up: a few pixels dominate every variance term at once.
    blown = at_reference.copy()
    blown[:5] = 50.0
    delta_blown = delta.copy()
    delta_blown[:5] = -49.0
    vf_bad = variance_fraction(blown, delta_blown)
    assert not vf_bad.reliable
    assert np.isnan(vf_bad.atmosphere_fraction)
    assert np.isfinite(vf_bad.coverage)


_OBS_BANDS = [
    b"Path length", b"To-sensor azimuth", b"To-sensor zenith",
    b"To-sun azimuth", b"To-sun zenith", b"Solar phase", b"Slope",
    b"Aspect", b"Cosine(i)", b"UTC Time", b"Earth-sun distance",
]


def _write_synthetic_emit_scene(tmp_path, scene_id, shipped_aod, cwv, radiance_by_pixel):
    """Minimal 1xN EMIT product set with exactly the datasets ``run_emit`` reads.

    ``radiance_by_pixel`` is (N, 2): N pixels, 2 bands at 700/800 nm.
    """
    n = len(shipped_aod)
    scene_dir = tmp_path / "scenes" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    wl = np.array([700.0, 800.0])
    paths = {key: scene_dir / name.format(sid=scene_id)
             for key, name in emit._PRODUCT_FILENAMES.items()}

    with h5py.File(paths["rfl"], "w") as rfl:
        rfl.create_dataset("reflectance", data=np.zeros((1, n, 2)))
        rfl.create_group("sensor_band_parameters").create_dataset("wavelengths", data=wl)
    with h5py.File(paths["rad"], "w") as rad:
        rad.create_dataset("radiance", data=np.asarray(radiance_by_pixel).reshape(1, n, 2))
        rad.create_group("sensor_band_parameters").create_dataset("wavelengths", data=wl)
    with h5py.File(paths["obs"], "w") as obs:
        cube = np.zeros((1, n, len(_OBS_BANDS)))
        obs.create_dataset("obs", data=cube)
        obs.create_group("sensor_band_parameters").create_dataset(
            "observation_bands", data=np.array(_OBS_BANDS, dtype="S32"))
    with h5py.File(paths["mask"], "w") as mask_h5:
        bands = np.zeros((1, n, 2))
        bands[0, :, 0] = shipped_aod
        bands[0, :, 1] = cwv
        mask_h5.create_dataset("mask", data=bands)
        mask_h5.create_group("sensor_band_parameters").create_dataset(
            "mask_bands", data=np.array([b"AOD550", b"H2O (g cm-2)"], dtype="S32"))


def _emit_mask(rfl, aoi):
    return np.ones((aoi[1] - aoi[0], aoi[3] - aoi[2]), dtype=bool)


def test_run_emit_realized_sensitivity_with_injected_correct(tmp_path):
    scene_id = "20250101T000000_2500101_001"
    shipped_aod = np.array([0.10, 0.30, 0.50])
    radiance = np.ones((3, 2))
    _write_synthetic_emit_scene(tmp_path, scene_id, shipped_aod, np.ones(3), radiance)

    result = run_emit(scene_id, (0, 1, 0, 3), _emit_mask, [700.0, 800.0], 0.20, "Maritime",
                      algorithm=_sum_algorithm, cache=tmp_path, correct=_fake_correct)

    # _fake_correct subtracts AOD from each band; _sum_algorithm sums two bands,
    # so delta = at_shipped - at_reference = -2 * (shipped_aod - 0.20).
    np.testing.assert_allclose(result.delta, -2.0 * (shipped_aod - 0.20))
    assert result.shape == (1, 3)
    assert not result.clamped.any()
    assert result.reference_aod == 0.20


def test_run_emit_reads_obs_geometry(tmp_path, monkeypatch):
    scene_id = "20250101T000000_2500101_002"
    _write_synthetic_emit_scene(tmp_path, scene_id, np.array([0.2]), np.ones(1), np.ones((1, 2)))

    seen = {}

    def spy_correct(*, sun_z, view_z, aot550, L_obs, **_):
        seen["sun_z"], seen["view_z"] = sun_z, view_z
        return np.asarray(L_obs, dtype=float) - aot550

    # non-zero geometry in the OBS cube: To-sun zenith band 4, To-sensor zenith band 2
    with h5py.File(emit.scene_paths(scene_id, tmp_path)["obs"], "r+") as obs:
        obs["obs"][0, 0, 4] = 33.0
        obs["obs"][0, 0, 2] = 4.0

    run_emit(scene_id, (0, 1, 0, 1), _emit_mask, [700.0, 800.0], 0.2, "Maritime",
             algorithm=_sum_algorithm, cache=tmp_path, correct=spy_correct)
    assert seen["sun_z"] == 33.0
    assert seen["view_z"] == 4.0


def test_run_emit_rejects_both_algorithm_and_fit(tmp_path):
    scene_id = "20250101T000000_2500101_003"
    _write_synthetic_emit_scene(tmp_path, scene_id, np.array([0.2]), np.ones(1), np.ones((1, 2)))
    with pytest.raises(ValueError, match="exactly one"):
        run_emit(scene_id, (0, 1, 0, 1), _emit_mask, [700.0], 0.2, "Maritime", cache=tmp_path,
                 correct=_fake_correct)


def test_resolve_lut_root_maps_a_directory_to_its_shard_root():
    assert sensitivity._resolve_lut_root(None, "DEFAULT") == "DEFAULT"
    resolved = sensitivity._resolve_lut_root("/tmp/lut_archive", "DEFAULT")
    assert resolved.replace("\\", "/").endswith("lut_archive/shards")


def test_run_tanager_threads_the_lut_directory(tmp_path, monkeypatch):
    scene_id = "20250101_000000_00_0000"
    _write_synthetic_scene(tmp_path, scene_id, np.array([0.2]), np.ones(1), np.ones((1, 2)))

    seen = {}
    real = sensitivity._run_from_arrays

    def spy(*args, **kwargs):
        seen["lut_root"] = kwargs.get("lut_root")
        return real(*args, **kwargs)

    monkeypatch.setattr(sensitivity, "_run_from_arrays", spy)
    run_tanager(scene_id, (0, 1, 0, 1), _full_mask, [700.0, 800.0], 0.2, "Continental",
                algorithm=_sum_algorithm, cache=tmp_path, correct=_fake_correct,
                lut=tmp_path / "my_lut")
    assert seen["lut_root"].replace("\\", "/").endswith("my_lut/shards")


def test_reconstruction_gap_magnitudes():
    at_shipped = np.array([0.20, 0.25, 0.30, np.nan])
    at_isofit = np.array([0.19, 0.27, 0.28, 0.10])
    rg = reconstruction_gap(at_shipped, at_isofit)
    assert np.isnan(rg.gap[3])
    assert rg.median_abs_gap == pytest.approx(0.02)
    assert rg.p90_abs_gap >= rg.median_abs_gap
