import numpy as np

from atmoresponse.recipes.water import mndwi, water_candidate


WL_NM = np.array([540.0, 560.0, 1590.0, 1610.0])


def test_mndwi_uses_nearest_bands_and_band_last_arrays():
    reflectance = np.array(
        [
            [[0.20, 0.50, 0.05, 0.10], [0.10, 0.20, 0.30, 0.40]],
            [[0.01, 0.02, 0.02, 0.03], [0.30, 0.40, 0.10, 0.20]],
        ]
    )

    score = mndwi(reflectance, WL_NM, green_nm=560.0, swir_nm=1600.0)

    np.testing.assert_allclose(score[0, 0], (0.50 - 0.05) / (0.50 + 0.05))
    np.testing.assert_allclose(score[0, 1], (0.20 - 0.30) / (0.20 + 0.30))


def test_mndwi_rejects_fill_and_near_zero_denominator():
    reflectance = np.array(
        [
            [-999.0, -999.0, 0.05, 0.10],
            [0.10, 0.20, -0.20, -0.30],
        ]
    )

    score = mndwi(reflectance, WL_NM, green_nm=560.0, swir_nm=1600.0)

    assert np.isnan(score[0])
    assert np.isnan(score[1])


def test_water_candidate_applies_threshold():
    reflectance = np.array(
        [
            [0.20, 0.50, 0.05, 0.10],
            [0.10, 0.20, 0.30, 0.40],
        ]
    )

    assert water_candidate(reflectance, WL_NM, green_nm=560.0, swir_nm=1600.0).tolist() == [
        True,
        False,
    ]
    assert water_candidate(
        reflectance,
        WL_NM,
        threshold=0.9,
        green_nm=560.0,
        swir_nm=1600.0,
    ).tolist() == [False, False]
