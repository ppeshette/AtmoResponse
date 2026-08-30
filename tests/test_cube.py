import numpy as np
import pytest

from atmoresponse.cube import HyperspectralCube


def test_cube_validates_band_last_shape():
    cube = HyperspectralCube(
        values=np.ones((2, 3, 4)),
        wavelengths_nm=np.array([500.0, 600.0, 700.0, 800.0]),
    )

    assert cube.sample_shape == (2, 3)
    assert cube.band_count == 4


def test_cube_rejects_shape_mismatches():
    with pytest.raises(ValueError, match="band dimension"):
        HyperspectralCube(values=np.ones((2, 3, 4)), wavelengths_nm=np.array([500.0, 600.0]))

    with pytest.raises(ValueError, match="mask shape"):
        HyperspectralCube(
            values=np.ones((2, 3, 4)),
            wavelengths_nm=np.array([500.0, 600.0, 700.0, 800.0]),
            mask=np.ones((2, 2), dtype=bool),
        )

    with pytest.raises(ValueError, match="geometry"):
        HyperspectralCube(
            values=np.ones((2, 3, 4)),
            wavelengths_nm=np.array([500.0, 600.0, 700.0, 800.0]),
            geometry={"sun_z": np.ones((3, 2))},
        )


def test_cube_nearest_band_returns_wavelength_and_values():
    values = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    cube = HyperspectralCube(values=values, wavelengths_nm=np.array([500.0, 600.0, 700.0, 800.0]))

    wavelength, plane = cube.nearest_band(690.0)

    assert wavelength == 700.0
    np.testing.assert_array_equal(plane, values[..., 2])


def test_cube_subset_wavelengths_keeps_context():
    values = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    mask = np.ones((2, 3), dtype=bool)
    geometry = {"sun_z": np.ones((2, 3))}
    cube = HyperspectralCube(
        values=values,
        wavelengths_nm=np.array([500.0, 600.0, 700.0, 800.0]),
        mask=mask,
        geometry=geometry,
        metadata={"source": "fixture"},
    )

    subset = cube.subset_wavelengths([510.0, 790.0])

    np.testing.assert_array_equal(subset.wavelengths_nm, [500.0, 800.0])
    np.testing.assert_array_equal(subset.values, values[..., [0, 3]])
    assert subset.mask is mask
    assert subset.metadata == {"source": "fixture"}
