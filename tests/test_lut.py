import numpy as np

from atmoresponse.lut import (
    CorrectionCoefficients,
    band_table,
    load_axes,
    month_day_to_doy,
    nearest_band_index,
    normalise_radiance,
    radiance_from_reflectance,
    reflectance_from_radiance,
)


def test_radiance_reflectance_round_trip_scalar():
    reflectance = 0.23
    radiance = radiance_from_reflectance(0.31, 0.012, 0.18, reflectance)
    recovered = reflectance_from_radiance(0.31, 0.012, 0.18, radiance)

    np.testing.assert_allclose(recovered, reflectance)


def test_radiance_reflectance_round_trip_array():
    reflectance = np.array([0.02, 0.11, 0.23, 0.41])
    radiance = radiance_from_reflectance(0.31, 0.012, 0.18, reflectance)
    recovered = reflectance_from_radiance(0.31, 0.012, 0.18, radiance)

    np.testing.assert_allclose(recovered, reflectance)


def test_correction_coefficients_wrapper_matches_functions():
    coefficients = CorrectionCoefficients(xa=0.31, xb=0.012, xc=0.18)
    reflectance = np.array([0.05, 0.2, 0.4])

    radiance = coefficients.radiance_from_reflectance(reflectance)
    np.testing.assert_allclose(radiance, radiance_from_reflectance(0.31, 0.012, 0.18, reflectance))
    np.testing.assert_allclose(coefficients.reflectance_from_radiance(radiance), reflectance)


def test_bundled_axes_have_expected_structure():
    axes = load_axes()
    for name in ("sza", "vza", "raa", "aerosol", "cwv", "ozone", "aod", "band"):
        assert axes["axes"][name]["values"]
    assert "Maritime" in axes["axes"]["aerosol"]["values"]

    centre_nm, fwhm_nm = band_table(axes)
    assert centre_nm.shape == fwhm_nm.shape
    assert centre_nm.size == len(axes["axes"]["band"]["values"])


def test_nearest_band_index_snaps_to_nearest_centre():
    axes = load_axes()
    centre_nm, _ = band_table(axes)
    idx = nearest_band_index(axes, 0.665)  # 665 nm, passed in um
    assert abs(centre_nm[idx] - 665.0) <= np.abs(centre_nm - 665.0).min() + 1e-9


def test_normalise_radiance_reference_doy_identity_and_perihelion_scaling():
    np.testing.assert_allclose(normalise_radiance(100.0, month_day_to_doy(7, 1)), 100.0)
    # Perihelion (early January): observed radiance rescales down toward the July reference.
    assert float(normalise_radiance(100.0, 3)) < 100.0
