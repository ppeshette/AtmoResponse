import h5py
import numpy as np

from atmoresponse import masks
from atmoresponse.tanager_ortho import GRID


ROWS, COLS = 4, 4
WL_NM = np.array([560.0, 1600.0, 704.0])


def _fake_sr():
    f = h5py.File("sr", mode="w", driver="core", backing_store=False)
    cube = np.zeros((len(WL_NM), ROWS, COLS), dtype=float)
    cube[0] = 0.50
    cube[1] = 0.05
    cube[2] = 0.30
    cube[0, 3, 3] = 0.05
    cube[1, 3, 3] = 0.50
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
    f.create_dataset(GRID + "aerosol_optical_depth", data=np.array(
        [
            [0.05, 0.10, 0.20, 0.30],
            [0.40, 0.50, 0.60, np.nan],
            [0.20, 0.20, 0.20, 0.20],
            [0.20, 0.20, 0.20, 0.20],
        ]
    ))
    return f


def test_combine_all_and_any_check_shapes():
    a = np.array([[True, False], [True, True]])
    b = np.array([[True, True], [False, True]])

    np.testing.assert_array_equal(masks.combine_all(a, b), [[True, False], [False, True]])
    np.testing.assert_array_equal(masks.combine_any(a, b), [[True, True], [True, True]])

    try:
        masks.combine_all(a, np.array([True, False]))
    except ValueError:
        pass
    else:
        raise AssertionError("expected shape mismatch")


def test_erode_shrinks_boolean_mask_by_square_neighborhood():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True

    eroded = masks.erode(mask, pixels=1)

    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    np.testing.assert_array_equal(eroded, expected)
    np.testing.assert_array_equal(masks.erode(mask, pixels=0), mask)


def test_tanager_clear_combines_quality_masks():
    with _fake_sr() as sr:
        clear = masks.tanager_clear(sr)

    assert not clear[0, 0]
    assert not clear[1, 1]
    assert not clear[2, 2]
    assert clear[0, 1]


def test_aod_in_lut_uses_axis_bounds():
    axes = {"axes": {"aod": {"values": [0.10, 0.50]}}}

    with _fake_sr() as sr:
        selected = masks.aod_in_lut(sr, axes=axes)

    assert not selected[0, 0]
    assert selected[0, 1]
    assert selected[1, 1]
    assert not selected[1, 2]
    assert not selected[1, 3]


def test_tanager_water_consumes_recipe_and_quality_mask():
    with _fake_sr() as sr:
        selected = masks.tanager_water(sr, erode_pixels=0)

    assert not selected[0, 0]
    assert not selected[1, 1]
    assert not selected[2, 2]
    assert not selected[3, 3]
    assert selected[0, 1]
