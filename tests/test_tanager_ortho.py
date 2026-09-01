import h5py
import numpy as np

from atmoresponse.tanager_ortho import (
    GRID,
    column_water_vapour,
    geometry,
    land_valid_mask,
    radiance_cube,
    radiance_at,
    radiance_window,
    reflectance_cube,
    reflectance_at,
    scene_paths,
    shipped_aod,
    shipped_aod_summary,
    validate_aoi,
    wavelengths_nm,
)
from atmoresponse.bands import band_index

ROWS, COLS = 4, 4
WL_NM = np.array([500.0, 600.0, 700.0, 800.0, 900.0, 1000.0])


def _fake_sr():
    f = h5py.File("sr", mode="w", driver="core", backing_store=False)
    cube = np.zeros((len(WL_NM), ROWS, COLS))
    for i in range(len(WL_NM)):
        cube[i] = i
    sr = f.create_dataset(GRID + "surface_reflectance", data=cube)
    sr.attrs["wavelengths"] = WL_NM

    cloud = np.zeros((ROWS, COLS), dtype=int)
    cirrus = np.zeros((ROWS, COLS), dtype=int)
    nodata = np.zeros((ROWS, COLS), dtype=int)
    cloud[0, 0] = 1
    cirrus[1, 1] = 1
    nodata[2, 2] = 1
    f.create_dataset(GRID + "beta_cloud_mask", data=cloud)
    f.create_dataset(GRID + "beta_cirrus_mask", data=cirrus)
    f.create_dataset(GRID + "nodata_pixels", data=nodata)
    f.create_dataset(GRID + "aerosol_optical_depth", data=np.arange(ROWS * COLS).reshape(ROWS, COLS) * 0.01)
    f.create_dataset(GRID + "column_water_vapour", data=np.arange(ROWS * COLS).reshape(ROWS, COLS) * 0.1)
    return f


def _fake_l1():
    f = h5py.File("l1", mode="w", driver="core", backing_store=False)
    cube = np.zeros((len(WL_NM), ROWS, COLS))
    for i in range(len(WL_NM)):
        cube[i] = i
    ds = f.create_dataset(GRID + "toa_radiance", data=cube)
    ds.attrs["wavelengths"] = WL_NM
    for name in ("sun_zenith", "sun_azimuth", "sensor_zenith", "sensor_azimuth"):
        f.create_dataset(GRID + name, data=np.arange(ROWS * COLS).reshape(ROWS, COLS).astype(float))
    return f


def test_band_index_uses_nearest_wavelength():
    assert band_index(WL_NM, 505.0) == 0
    assert band_index(WL_NM, 998.0) == 5


def test_scene_paths_use_data_dir_layout(tmp_path):
    sr_path, l1_path = scene_paths("scene-a", tmp_path)

    assert sr_path == tmp_path / "scenes" / "scene-a" / "scene-a_ortho_sr.h5"
    assert l1_path == tmp_path / "scenes" / "scene-a" / "scene-a_ortho_radiance.h5"


def test_selector_requires_exactly_one_mode():
    with _fake_l1() as f:
        for kwargs in ({}, {"aoi": (0, 2, 0, 2), "rows": [0], "cols": [0]}):
            try:
                geometry(f, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected ValueError for {kwargs}")


def test_validate_aoi_rejects_bad_bounds():
    with _fake_sr() as f:
        validate_aoi(f, (0, ROWS, 0, COLS))
        for bad in (
            (-1, ROWS, 0, COLS),
            (0, ROWS, -1, COLS),
            (2, 2, 0, COLS),
            (2, 1, 0, COLS),
            (0, ROWS + 1, 0, COLS),
            (0, ROWS, 0, COLS + 1),
        ):
            try:
                validate_aoi(f, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected ValueError for aoi={bad}")


def test_wavelengths_nm_reads_sr_and_l1_datasets():
    with _fake_sr() as sr, _fake_l1() as l1:
        np.testing.assert_array_equal(wavelengths_nm(sr, "surface_reflectance"), WL_NM)
        np.testing.assert_array_equal(wavelengths_nm(l1, "toa_radiance"), WL_NM)


def test_land_valid_mask_screens_cloud_cirrus_nodata():
    with _fake_sr() as f:
        mask = land_valid_mask(f, aoi=(0, ROWS, 0, COLS))

    assert mask.shape == (ROWS, COLS)
    assert not mask[0, 0]
    assert not mask[1, 1]
    assert not mask[2, 2]
    assert mask[3, 3]
    assert mask.sum() == ROWS * COLS - 3


def test_land_valid_mask_pixel_list_matches_aoi_block():
    with _fake_sr() as f:
        block = land_valid_mask(f, aoi=(0, ROWS, 0, COLS))
        rows, cols = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij")
        scattered = land_valid_mask(f, rows=rows.ravel(), cols=cols.ravel())

    np.testing.assert_array_equal(scattered.reshape(ROWS, COLS), block)


def test_shipped_aod_and_cwv_match_between_modes():
    with _fake_sr() as f:
        aod_block = shipped_aod(f, aoi=(1, 3, 1, 3))
        cwv_block = column_water_vapour(f, aoi=(1, 3, 1, 3))
        rows = [1, 1, 2, 2]
        cols = [1, 2, 1, 2]
        aod_scattered = shipped_aod(f, rows=rows, cols=cols)
        cwv_scattered = column_water_vapour(f, rows=rows, cols=cols)

    np.testing.assert_array_equal(aod_scattered.reshape(2, 2), aod_block)
    np.testing.assert_array_equal(cwv_scattered.reshape(2, 2), cwv_block)


def test_shipped_aod_summary_returns_neutral_aod_summary():
    with _fake_sr() as f:
        summary = shipped_aod_summary(
            f,
            aoi=(0, 2, 0, 2),
            valid_mask=np.array([[True, True], [True, False]]),
        )

    assert summary.value == 0.01
    assert summary.statistic == "median"
    assert summary.count == 3
    assert summary.mean == np.mean([0.0, 0.01, 0.04])
    assert summary.minimum == 0.0
    assert summary.maximum == 0.04
    assert summary.detail == "Tanager shipped aerosol_optical_depth"


def test_geometry_reads_all_four_fields_consistently():
    with _fake_l1() as f:
        block = geometry(f, aoi=(0, ROWS, 0, COLS))
        rows, cols = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij")
        scattered = geometry(f, rows=rows.ravel(), cols=cols.ravel())

    assert set(block) == {"sun_z", "sun_a", "view_z", "view_a"}
    for key in block:
        np.testing.assert_array_equal(scattered[key].reshape(ROWS, COLS), block[key])


def test_radiance_window_selects_band_range():
    with _fake_l1() as f:
        wl, cube = radiance_window(f, 650.0, 950.0, aoi=(0, ROWS, 0, COLS))

    np.testing.assert_array_equal(wl, [700.0, 800.0, 900.0])
    assert cube.shape == (ROWS, COLS, 3)
    np.testing.assert_array_equal(cube[0, 0], [2.0, 3.0, 4.0])


def test_radiance_at_finds_nearest_bands():
    with _fake_l1() as f:
        wl, cube = radiance_at(f, [505.0, 998.0], aoi=(0, ROWS, 0, COLS))

    np.testing.assert_array_equal(wl, [500.0, 1000.0])
    np.testing.assert_array_equal(cube[2, 1], [0.0, 5.0])


def test_radiance_at_dense_targets_with_duplicate_and_unsorted_bands():
    # A spectral library's own grid: several targets snap to one band, out of order.
    targets = [905.0, 495.0, 510.0, 610.0, 590.0]
    with _fake_l1() as f:
        wl, cube = radiance_at(f, targets, aoi=(0, ROWS, 0, COLS))
        rows, cols = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij")
        _, scattered = radiance_at(f, targets, rows=rows.ravel(), cols=cols.ravel())

    np.testing.assert_array_equal(wl, [900.0, 500.0, 500.0, 600.0, 600.0])
    np.testing.assert_array_equal(scattered.reshape(ROWS, COLS, len(targets)), cube)


def test_radiance_at_pixel_list_matches_aoi_block():
    with _fake_l1() as f:
        _, block = radiance_at(f, [600.0, 900.0], aoi=(0, ROWS, 0, COLS))
        rows, cols = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij")
        _, scattered = radiance_at(f, [600.0, 900.0], rows=rows.ravel(), cols=cols.ravel())

    np.testing.assert_array_equal(scattered.reshape(ROWS, COLS, 2), block)


def test_reflectance_at_finds_nearest_bands():
    with _fake_sr() as f:
        wl, cube = reflectance_at(f, [505.0, 998.0], aoi=(0, ROWS, 0, COLS))

    np.testing.assert_array_equal(wl, [500.0, 1000.0])
    np.testing.assert_array_equal(cube[2, 1], [0.0, 5.0])


def test_reflectance_at_pixel_list_matches_aoi_block():
    with _fake_sr() as f:
        _, block = reflectance_at(f, [600.0, 900.0], aoi=(0, ROWS, 0, COLS))
        rows, cols = np.meshgrid(np.arange(ROWS), np.arange(COLS), indexing="ij")
        _, scattered = reflectance_at(f, [600.0, 900.0], rows=rows.ravel(), cols=cols.ravel())

    np.testing.assert_array_equal(scattered.reshape(ROWS, COLS, 2), block)


def test_reflectance_cube_returns_neutral_cube():
    valid = np.ones((ROWS, COLS), dtype=bool)
    with _fake_sr() as f:
        cube = reflectance_cube(f, aoi=(0, ROWS, 0, COLS), valid_mask=valid)

    assert cube.values.shape == (ROWS, COLS, len(WL_NM))
    np.testing.assert_array_equal(cube.wavelengths_nm, WL_NM)
    assert cube.mask is valid
    assert cube.metadata == {"source": "tanager", "quantity": "surface_reflectance"}


def test_radiance_cube_returns_neutral_cube_with_geometry():
    with _fake_l1() as f:
        cube = radiance_cube(f, aoi=(0, ROWS, 0, COLS))

    assert cube.values.shape == (ROWS, COLS, len(WL_NM))
    np.testing.assert_array_equal(cube.wavelengths_nm, WL_NM)
    assert set(cube.geometry) == {"sun_z", "sun_a", "view_z", "view_a"}
    assert cube.metadata == {"source": "tanager", "quantity": "toa_radiance"}
