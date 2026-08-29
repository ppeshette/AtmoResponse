import numpy as np

from atmoresponse.lut import (
    CorrectionCoefficients,
    radiance_from_reflectance,
    reflectance_from_radiance,
)


def test_radiance_reflectance_round_trip_scalar():
    coefficients = CorrectionCoefficients(xa=0.31, xb=0.012, xc=0.18)
    reflectance = 0.23

    radiance = radiance_from_reflectance(coefficients, reflectance)
    recovered = reflectance_from_radiance(coefficients, radiance)

    np.testing.assert_allclose(recovered, reflectance)


def test_radiance_reflectance_round_trip_array():
    coefficients = CorrectionCoefficients(xa=0.31, xb=0.012, xc=0.18)
    reflectance = np.array([0.02, 0.11, 0.23, 0.41])

    radiance = radiance_from_reflectance(coefficients, reflectance)
    recovered = reflectance_from_radiance(coefficients, radiance)

    np.testing.assert_allclose(recovered, reflectance)
