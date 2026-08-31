import numpy as np

from atmoresponse.recipes.agriculture import (
    CANOPY_NIR_NM,
    CANOPY_RED_EDGE_NM,
    CANOPY_RED_NM,
    R704_NM,
    R815_NM,
    WI_1240_NM,
    WI_1530_NM,
    WI_REF_NM,
    canopy_chlorophyll_rsi,
    canopy_present,
    vegetation_water_indices,
)

_CANOPY_WL = np.array([CANOPY_RED_NM, *CANOPY_RED_EDGE_NM, CANOPY_NIR_NM])


def test_canopy_present_passes_a_vegetation_spectrum_and_rejects_soil():
    # red low, red-edge rising, NIR high -> a canopy
    canopy = np.array([0.04, 0.10, 0.25, 0.40, 0.45])
    # flat bright soil: SAVI low, NIR not above red -> rejected
    soil = np.array([0.30, 0.30, 0.30, 0.30, 0.28])

    result = canopy_present(np.stack([canopy, soil]), _CANOPY_WL)

    assert result.tolist() == [True, False]


def test_canopy_present_threshold_is_tunable():
    weak = np.array([0.10, 0.12, 0.14, 0.18, 0.19])

    assert not canopy_present(weak, _CANOPY_WL, savi_threshold=0.25)
    assert canopy_present(weak, _CANOPY_WL, savi_threshold=0.05)


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
