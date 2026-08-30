import numpy as np

from atmoresponse.recipes.cyanobacteria import (
    CI_CENTER,
    CI_DETECT,
    CI_HI,
    CI_LO,
    CI_RISK,
    PC_CENTER,
    PC_HI,
    PC_LO,
    cyanobacteria_index,
    spectral_shape,
)


def test_spectral_shape_zero_on_a_straight_line():
    fraction = (CI_CENTER - CI_LO) / (CI_HI - CI_LO)
    r_lo = 0.10
    r_hi = 0.20
    r_center = r_lo + (r_hi - r_lo) * fraction

    np.testing.assert_allclose(
        spectral_shape(r_lo, r_center, r_hi, CI_LO, CI_CENTER, CI_HI),
        0.0,
        atol=1e-12,
    )


def test_spectral_shape_sign():
    baseline = 0.10

    above = spectral_shape(0.10, baseline + 0.02, 0.10, PC_LO, PC_CENTER, PC_HI)
    below = spectral_shape(0.10, baseline - 0.02, 0.10, PC_LO, PC_CENTER, PC_HI)

    np.testing.assert_allclose(above, 0.02)
    np.testing.assert_allclose(below, -0.02)


def test_ci_thresholds_are_not_interchangeable():
    assert CI_RISK < CI_DETECT


def test_cyanobacteria_index_wires_baselines_and_signs():
    wavelengths = np.array([PC_LO, CI_LO, CI_CENTER, CI_HI])
    reflectance = np.array([0.05, 0.08, 0.10, 0.12])

    ci, ss_665, cyano_dominant = cyanobacteria_index(reflectance, wavelengths)

    want_ci = -spectral_shape(0.08, 0.10, 0.12, CI_LO, CI_CENTER, CI_HI)
    want_ss = spectral_shape(0.05, 0.08, 0.10, PC_LO, PC_CENTER, PC_HI)
    np.testing.assert_allclose(ci, want_ci)
    np.testing.assert_allclose(ss_665, want_ss)
    assert bool(cyano_dominant) == bool((want_ci > 0) and (want_ss > 0))


def test_cyanobacteria_index_accepts_band_last_cube_and_masks_fill():
    wavelengths = np.array([PC_LO, CI_LO, CI_CENTER, CI_HI])
    reflectance = np.array([
        [[0.05, 0.08, 0.10, 0.12], [0.05, -9999.0, 0.10, 0.12]],
    ])

    ci, ss_665, cyano_dominant = cyanobacteria_index(reflectance, wavelengths)

    assert ci.shape == (1, 2)
    assert ss_665.shape == (1, 2)
    assert cyano_dominant.shape == (1, 2)
    assert np.isnan(ci[0, 1])
    assert np.isnan(ss_665[0, 1])
    assert not bool(cyano_dominant[0, 1])
