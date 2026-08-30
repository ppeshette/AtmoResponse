import numpy as np
import pytest

from atmoresponse.sensitivity import LabeledScore
from atmoresponse.recipes.sam import (
    classify_by_angle,
    labeled_sam_score,
    labels_for_indices,
    resample_library,
    resample_spectrum,
    sam_angles,
)


def test_resample_spectrum_interpolates_without_extrapolating():
    values = resample_spectrum([500.0, 600.0, 700.0], [0.1, 0.3, 0.5], [550.0, 650.0])

    np.testing.assert_allclose(values, [0.2, 0.4])

    with pytest.raises(ValueError, match="beyond library coverage"):
        resample_spectrum([500.0, 600.0], [0.1, 0.3], [450.0])


def test_resample_library_resamples_each_row():
    library = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])

    resampled = resample_library([500.0, 600.0, 700.0], library, [550.0, 650.0])

    np.testing.assert_allclose(resampled, [[0.5, 1.5], [1.5, 0.5]])


def test_sam_angles_match_identity_and_orthogonal_cases():
    angles = sam_angles([[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]])

    np.testing.assert_allclose(angles, [[0.0, np.pi / 2], [np.pi / 2, 0.0]], atol=1e-12)


def test_sam_angles_preserve_band_last_sample_shape():
    spectra = np.array([
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 1.0], [0.0, 0.0]],
    ])

    angles = sam_angles(spectra, [[1.0, 0.0], [0.0, 1.0]])

    assert angles.shape == (2, 2, 2)
    np.testing.assert_allclose(angles[0, 0], [0.0, np.pi / 2], atol=1e-12)
    assert np.isnan(angles[1, 1]).all()


def test_classify_by_angle_marks_invalid_spectra():
    result = classify_by_angle([[1.0, 0.0], [np.nan, 1.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]])

    np.testing.assert_allclose(result.angles[0], [0.0, np.pi / 2], atol=1e-12)
    np.testing.assert_array_equal(result.class_index, [0, -1, -1])


def test_labels_for_indices_maps_invalid_to_none_by_default():
    labels = labels_for_indices(np.array([[0, 1, -1]]), ["burned", "unburned"])

    assert labels.tolist() == [["burned", "unburned", None]]


def test_labels_for_indices_can_collapse_group_labels():
    labels = labels_for_indices(
        np.array([[0, 1, 2, -1]]),
        ["char_dark", "char_bright", "soil"],
        group_labels=["burned", "burned", "soil"],
    )

    assert labels.tolist() == [["burned", "burned", "soil", None]]


def test_labeled_sam_score_uses_target_angle_nearest_label_and_margin():
    score = labeled_sam_score(
        [0.95, 0.05],
        [[1.0, 0.0], [0.0, 1.0]],
        labels=["burned", "unburned"],
        target_index=0,
    )

    assert isinstance(score, LabeledScore)
    assert score.label == "burned"
    assert score.value < 0.1
    assert score.margin > 1.0


def test_labeled_sam_score_keeps_target_score_separate_from_nearest_label():
    score = labeled_sam_score(
        [0.05, 0.95],
        [[1.0, 0.0], [0.0, 1.0]],
        labels=["burned", "unburned"],
        target_index=0,
    )

    assert score.label == "unburned"
    assert score.value > 1.0
    assert score.margin > 1.0


def test_labeled_sam_score_reports_group_label_when_provided():
    score = labeled_sam_score(
        [0.95, 0.05],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        labels=["char_dark", "char_bright", "soil"],
        group_labels=["burned", "burned", "soil"],
        target_index=0,
    )

    assert score.label == "burned"
    assert score.value < 0.1
