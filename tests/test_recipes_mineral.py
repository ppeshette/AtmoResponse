import numpy as np
import pytest

from atmoresponse.recipes.mineral import aloh_2200_depth


def test_flat_spectrum_has_zero_aloh_depth():
    depth = aloh_2200_depth([0.4, 0.4, 0.4], [2100.0, 2200.0, 2300.0])

    np.testing.assert_allclose(depth, 0.0)


def test_centered_absorption_has_positive_aloh_depth():
    depth = aloh_2200_depth([0.4, 0.2, 0.4], [2150.0, 2200.0, 2250.0])

    np.testing.assert_allclose(depth, 0.5)


def test_aloh_depth_accepts_band_last_cube():
    spectra = np.array([[[0.4, 0.2, 0.4], [0.4, 0.3, 0.4]]])

    depth = aloh_2200_depth(spectra, [2150.0, 2200.0, 2250.0])

    np.testing.assert_allclose(depth, [[0.5, 0.25]])


def test_aloh_depth_rejects_out_of_range_feature():
    with pytest.raises(ValueError, match="outside"):
        aloh_2200_depth([0.4, 0.4], [2200.0, 2300.0])
