import numpy as np
import pytest

from atmoresponse.cube import HyperspectralCube
from atmoresponse.recipes.wildfire_sam import (
    WILDFIRE_SAM_TARGET_LABEL,
    classify_wildfire_sam,
    load_wildfire_sam_library,
    prepare_wildfire_sam,
    wildfire_sam_labels,
    wildfire_sam_score,
)


def test_load_wildfire_sam_library_exposes_public_metadata():
    library = load_wildfire_sam_library()

    assert library.reflectance.shape == (28, 901)
    assert library.wavelengths_nm[0] == 400.0
    assert library.wavelengths_nm[-1] == 1300.0
    assert library.target_label == WILDFIRE_SAM_TARGET_LABEL
    assert library.labels[library.target_index] == WILDFIRE_SAM_TARGET_LABEL
    assert library.group_labels.count("burned") == 14
    assert library.group_labels.count("unburned") == 14
    assert sorted(set(library.entry_sources)) == [
        "Landmann & Roy 2004, SAFARI 2000 fire residue, doi:10.3334/ORNLDAAC/751",
        "USGS Digital Spectral Library splib06a (Clark et al. 2007)",
    ]
    assert "400-1300 nm" in library.metadata["intended_use"]


def test_classify_wildfire_sam_uses_cube_bands_inside_library_range():
    library = load_wildfire_sam_library()
    wavelengths = np.array([390.0, 400.0, 850.0, 1300.0, 1310.0])
    reflectance = np.interp(wavelengths, library.wavelengths_nm, library.reflectance[0])
    cube = HyperspectralCube(values=reflectance.reshape(1, 1, -1), wavelengths_nm=wavelengths)

    result = classify_wildfire_sam(cube, library)

    assert result.class_index.tolist() == [[0]]
    np.testing.assert_allclose(result.angles[0, 0, 0], 0.0, atol=1e-12)


def test_prepare_wildfire_sam_reuses_resampled_endmembers():
    library = load_wildfire_sam_library()
    wavelengths = np.array([390.0, 400.0, 850.0, 1300.0, 1310.0])
    prepared = prepare_wildfire_sam(wavelengths, library)
    expected = np.array([
        np.interp(wavelengths, library.wavelengths_nm, library.reflectance[0]),
        np.interp(wavelengths, library.wavelengths_nm, library.reflectance[5]),
    ])

    scores = prepared.evaluate_many(expected)
    result = prepared.classify_values(expected.reshape(2, 1, -1))

    assert prepared.band_mask.tolist() == [False, True, True, True, False]
    assert prepared.endmembers.shape == (28, 3)
    assert [score.label for score in scores] == ["burned", "unburned"]
    np.testing.assert_array_equal(result.class_index, [[0], [5]])


def test_prepare_wildfire_sam_can_stride_in_range_bands():
    library = load_wildfire_sam_library()
    wavelengths = np.array([390.0, 400.0, 850.0, 1300.0, 1310.0])
    prepared = prepare_wildfire_sam(wavelengths, library, band_stride=2)

    assert prepared.band_mask.tolist() == [False, True, False, True, False]
    np.testing.assert_allclose(prepared.selected_wavelengths_nm, [400.0, 1300.0])
    assert prepared.endmembers.shape == (28, 2)


def test_classify_wildfire_sam_respects_cube_mask_and_fill_values():
    library = load_wildfire_sam_library()
    values = np.stack([library.reflectance[0], library.reflectance[5], library.reflectance[0]])
    values[2, 0] = -9999.0
    cube = HyperspectralCube(
        values=values.reshape(3, 1, -1),
        wavelengths_nm=library.wavelengths_nm,
        mask=np.array([[True], [False], [True]]),
    )

    result = classify_wildfire_sam(cube, library)

    np.testing.assert_array_equal(result.class_index, [[0], [-1], [-1]])


def test_wildfire_sam_labels_can_return_grouped_or_raw_labels():
    library = load_wildfire_sam_library()
    grouped = wildfire_sam_labels(np.array([0, 5, -1]), library)
    raw = wildfire_sam_labels(np.array([0, 5, -1]), library, grouped=False)

    assert grouped.tolist() == ["burned", "unburned", None]
    assert raw.tolist() == [library.labels[0], library.labels[5], None]


def test_wildfire_sam_score_reports_angle_to_target_and_grouped_label():
    library = load_wildfire_sam_library()
    score = wildfire_sam_score(library.reflectance[0], library.wavelengths_nm, library)
    prepared_score = prepare_wildfire_sam(library.wavelengths_nm, library)(library.reflectance[0])

    assert score.label == "burned"
    assert prepared_score == score
    np.testing.assert_allclose(score.value, 0.0, atol=1e-12)
    assert score.margin > 0.0


def test_wildfire_sam_rejects_out_of_range_wavelengths():
    library = load_wildfire_sam_library()

    with pytest.raises(ValueError, match="no wavelengths"):
        wildfire_sam_score([0.1, 0.2], [1400.0, 1500.0], library)
