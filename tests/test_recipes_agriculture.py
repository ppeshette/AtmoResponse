import numpy as np

from atmoresponse.recipes.agriculture import (
    R704_NM,
    R815_NM,
    WI_1240_NM,
    WI_1530_NM,
    WI_REF_NM,
    canopy_chlorophyll_rsi,
    vegetation_water_indices,
)


def test_rsi_picks_nearest_bands_on_band_last_cube():
    wavelengths = np.array([600.0, 703.8, 706.5, 814.6, 816.9, 900.0])
    reflectance = (np.arange(6, dtype="f8") * 10.0).reshape(1, 1, 6)

    rsi = canopy_chlorophyll_rsi(reflectance, wavelengths)

    np.testing.assert_allclose(rsi, [[30.0 / 10.0]])


def test_rsi_ratio_direction_and_fill_values():
    wavelengths = np.array([R704_NM, R815_NM])

    np.testing.assert_allclose(canopy_chlorophyll_rsi([0.20, 0.40], wavelengths), 2.0)
    assert np.isnan(canopy_chlorophyll_rsi([0.20, -9999.0], wavelengths))
    assert np.isnan(canopy_chlorophyll_rsi([-9999.0, 0.40], wavelengths))


def test_vegetation_water_indices_share_reference_band():
    wavelengths = np.array([600.0, 863.9, 867.2, 1238.5, 1242.0, 1528.6, 1533.0])
    reflectance = (np.arange(7, dtype="f8") * 10.0).reshape(1, 1, 7)

    wi_1240, wi_1530 = vegetation_water_indices(reflectance, wavelengths)

    np.testing.assert_allclose(wi_1240, [[10.0 / 30.0]])
    np.testing.assert_allclose(wi_1530, [[10.0 / 50.0]])


def test_vegetation_water_indices_ratio_direction_and_fill_values():
    wavelengths = np.array([WI_REF_NM, WI_1240_NM, WI_1530_NM])

    wi_1240, wi_1530 = vegetation_water_indices([0.40, 0.20, 0.10], wavelengths)
    np.testing.assert_allclose(wi_1240, 2.0)
    np.testing.assert_allclose(wi_1530, 4.0)

    wi_1240, wi_1530 = vegetation_water_indices([-9999.0, 0.20, 0.10], wavelengths)
    assert np.isnan(wi_1240)
    assert np.isnan(wi_1530)
