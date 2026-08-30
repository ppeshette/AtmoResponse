import h5py
import numpy as np
import pytest

from atmoresponse import emit

ROWS, COLS = 3, 4
WL_NM = np.array([500.0, 600.0, 700.0])


def _fake_rfl():
    f = h5py.File("emit-rfl", mode="w", driver="core", backing_store=False)
    cube = np.arange(ROWS * COLS * len(WL_NM), dtype=float).reshape(ROWS, COLS, len(WL_NM))
    cube[0, 0, 1] = -9999.0
    reflectance = f.create_dataset("reflectance", data=cube)
    reflectance.attrs["_FillValue"] = np.array([-9999.0])
    band_parameters = f.create_group("sensor_band_parameters")
    wavelengths = band_parameters.create_dataset("wavelengths", data=WL_NM)
    wavelengths.attrs["_FillValue"] = np.array([-9999.0])
    band_parameters.create_dataset("good_wavelengths", data=np.array([1, 0, 1], dtype=np.uint8))
    return f


def _fake_rad():
    f = h5py.File("emit-rad", mode="w", driver="core", backing_store=False)
    cube = np.arange(ROWS * COLS * len(WL_NM), dtype=float).reshape(ROWS, COLS, len(WL_NM))
    cube[0, 0, 1] = -9999.0
    radiance = f.create_dataset("radiance", data=cube)
    radiance.attrs["_FillValue"] = np.array([-9999.0])
    band_parameters = f.create_group("sensor_band_parameters")
    wavelengths = band_parameters.create_dataset("wavelengths", data=WL_NM)
    wavelengths.attrs["_FillValue"] = np.array([-9999.0])
    return f


def _fake_mask():
    f = h5py.File("emit-mask", mode="w", driver="core", backing_store=False)
    mask = np.zeros((ROWS, COLS, 8), dtype=float)
    mask[:, :, 2] = 1.0
    mask[:, :, 5] = np.arange(ROWS * COLS).reshape(ROWS, COLS) * 0.01
    mask[:, :, 6] = np.arange(ROWS * COLS).reshape(ROWS, COLS) * 0.1
    mask[0, 0, 5] = -9999.0
    dataset = f.create_dataset("mask", data=mask)
    dataset.attrs["_FillValue"] = np.array([-9999.0])
    band_parameters = f.create_group("sensor_band_parameters")
    band_parameters.create_dataset(
        "mask_bands",
        data=np.array([
            b"Cloud flag",
            b"Cirrus flag",
            b"Water flag",
            b"Spacecraft Flag",
            b"Dilated Cloud Flag",
            b"AOD550",
            b"H2O (g cm-2)",
            b"Aggregate Flag",
        ], dtype="S32"),
    )
    return f


def test_wavelength_and_good_wavelength_readers():
    with _fake_rfl() as f:
        np.testing.assert_array_equal(emit.wavelengths_nm(f), WL_NM)
        np.testing.assert_array_equal(emit.good_wavelengths(f), [True, False, True])


def test_surface_reflectance_at_reads_nearest_bands_and_fill_as_nan():
    with _fake_rfl() as f:
        wl, values = emit.surface_reflectance_at(f, [505.0, 690.0], aoi=(0, 2, 0, 2))

    np.testing.assert_array_equal(wl, [500.0, 700.0])
    assert values.shape == (2, 2, 2)
    assert values[0, 0, 0] == 0.0


def test_reflectance_cube_returns_neutral_cube_with_good_wavelength_metadata():
    valid = np.ones((2, 2), dtype=bool)
    with _fake_rfl() as f:
        cube = emit.reflectance_cube(f, aoi=(0, 2, 0, 2), valid_mask=valid)

    assert cube.values.shape == (2, 2, len(WL_NM))
    assert np.isnan(cube.values[0, 0, 1])
    np.testing.assert_array_equal(cube.wavelengths_nm, WL_NM)
    np.testing.assert_array_equal(cube.metadata["good_wavelengths"], [True, False, True])
    assert cube.metadata["source"] == "emit"


def test_radiance_at_reads_nearest_bands_and_fill_as_nan():
    with _fake_rad() as f:
        wl, values = emit.radiance_at(f, [505.0, 690.0], aoi=(0, 2, 0, 2))

    np.testing.assert_array_equal(wl, [500.0, 700.0])
    assert values.shape == (2, 2, 2)
    assert values[0, 0, 0] == 0.0


def test_radiance_cube_returns_neutral_cube():
    with _fake_rad() as f:
        cube = emit.radiance_cube(f, aoi=(0, 2, 0, 2))

    assert cube.values.shape == (2, 2, len(WL_NM))
    assert np.isnan(cube.values[0, 0, 1])
    np.testing.assert_array_equal(cube.wavelengths_nm, WL_NM)
    assert cube.metadata == {"source": "emit", "quantity": "radiance"}


def test_mask_band_reads_by_name():
    with _fake_mask() as f:
        assert emit.mask_band_index(f, "AOD550") == 5
        aod = emit.mask_band(f, "AOD550", aoi=(0, 2, 0, 2))

    assert np.isnan(aod[0, 0])
    assert aod[1, 1] == 0.05


def test_mask_band_rejects_unknown_name():
    with _fake_mask() as f:
        with pytest.raises(KeyError, match="not found"):
            emit.mask_band_index(f, "not a band")


def test_aod550_and_h2o_helpers():
    with _fake_mask() as f:
        aod = emit.aod550(f, rows=[1, 2], cols=[1, 2])
        h2o = emit.h2o(f, rows=[1, 2], cols=[1, 2])

    np.testing.assert_array_equal(aod, [0.05, 0.1])
    np.testing.assert_array_equal(h2o, [0.5, 1.0])


def test_shipped_aod_summary_returns_neutral_summary():
    with _fake_mask() as f:
        summary = emit.shipped_aod_summary(
            f,
            aoi=(0, 2, 0, 2),
            valid_mask=np.array([[True, True], [True, False]]),
        )

    assert summary.value == 0.025
    assert summary.count == 2
    assert summary.detail == "EMIT shipped AOD550"
