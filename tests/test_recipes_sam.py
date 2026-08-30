import numpy as np
import pytest

from atmoresponse.sensitivity import LabeledScore
from atmoresponse.recipes.sam import (
    classify_by_angle,
    labeled_sam_score,
    labels_for_indices,
    prepare_sam_classifier,
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


def test_prepare_sam_classifier_reuses_resampled_endmembers():
    prepared = prepare_sam_classifier(
        [450.0, 500.0, 600.0, 700.0, 750.0],
        [500.0, 600.0, 700.0],
        [[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]],
        labels=["target", "other"],
        target_index=0,
    )

    result = prepared.classify_values([
        [999.0, 1.0, 0.0, 0.0, 999.0],
        [999.0, 0.0, 0.5, 1.0, 999.0],
    ])

    assert prepared.band_mask.tolist() == [False, True, True, True, False]
    np.testing.assert_allclose(prepared.endmembers, [[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]])
    np.testing.assert_array_equal(result.class_index, [0, 1])


def test_prepare_sam_classifier_can_stride_in_range_bands():
    prepared = prepare_sam_classifier(
        [450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0],
        [500.0, 600.0, 700.0],
        [[1.0, 0.0, 0.0], [0.0, 0.5, 1.0]],
        labels=["target", "other"],
        target_index=0,
        band_stride=2,
    )

    assert prepared.band_mask.tolist() == [False, True, False, True, False, True, False]
    np.testing.assert_allclose(prepared.selected_wavelengths_nm, [500.0, 600.0, 700.0])
    assert prepared.endmembers.shape == (2, 3)


def test_prepared_sam_classifier_evaluate_many_uses_grouped_labels():
    prepared = prepare_sam_classifier(
        [500.0, 600.0],
        [500.0, 600.0],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        labels=["char_dark", "char_bright", "soil"],
        group_labels=["burned", "burned", "unburned"],
        target_index=0,
    )

    scores = prepared.evaluate_many([[0.95, 0.05], [0.05, 0.95]])

    assert [score.label for score in scores] == ["burned", "unburned"]
    assert scores[0] == prepared.evaluate([0.95, 0.05])
    assert scores[0].value < scores[1].value


def test_prepared_sam_classifier_respects_mask_and_fill_values():
    prepared = prepare_sam_classifier(
        [500.0, 600.0],
        [500.0, 600.0],
        [[1.0, 0.0], [0.0, 1.0]],
        labels=["target", "other"],
        target_index=0,
        fill_limit=-900.0,
    )

    result = prepared.classify_values(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[-9999.0, 0.0]]],
        mask=np.array([[True], [False], [True]]),
    )

    np.testing.assert_array_equal(result.class_index, [[0], [-1], [-1]])


def test_prepare_sam_classifier_rejects_invalid_band_stride():
    with pytest.raises(ValueError, match="band_stride"):
        prepare_sam_classifier(
            [500.0, 600.0],
            [500.0, 600.0],
            [[1.0, 0.0], [0.0, 1.0]],
            labels=["target", "other"],
            target_index=0,
            band_stride=0,
        )
