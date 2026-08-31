import numpy as np
import pytest

from atmoresponse.recipes.endmembers import endmembers_from_labels


def _labeled_scene():
    """A 4x4 scene, 3 bands. Class 1 pixels are 0.1/0.2/0.3, class 2 pixels are
    0.5, the rest are class 0 background."""
    wl = np.array([500.0, 600.0, 700.0])
    labels = np.array([
        [1, 1, 1, 1],
        [1, 1, 2, 2],
        [2, 2, 2, 2],
        [0, 0, 0, 0],
    ])
    cube = np.zeros((4, 4, 3))
    cube[labels == 1] = [0.1, 0.2, 0.3]
    cube[labels == 2] = [0.5, 0.5, 0.5]
    cube[labels == 0] = [0.9, 0.9, 0.9]
    return cube, labels, wl


def test_endmembers_from_labels_returns_per_class_means_in_map_order():
    cube, labels, wl = _labeled_scene()

    lib = endmembers_from_labels(cube, labels, wl, {2: "bright", 1: "veg"}, min_pixels=1)

    assert lib.labels == ("bright", "veg")
    np.testing.assert_allclose(lib.endmembers, [[0.5, 0.5, 0.5], [0.1, 0.2, 0.3]])
    np.testing.assert_array_equal(lib.wavelengths_nm, wl)
    assert lib.pixel_counts == {"bright": 6, "veg": 6}


def test_endmembers_from_labels_accepts_pixel_list_form():
    cube, labels, wl = _labeled_scene()
    spectra = cube.reshape(-1, 3)
    flat_labels = labels.reshape(-1)

    lib = endmembers_from_labels(spectra, flat_labels, wl, {1: "veg"}, min_pixels=1)

    np.testing.assert_allclose(lib.endmembers, [[0.1, 0.2, 0.3]])


def test_endmembers_from_labels_excludes_nonfinite_and_masked_pixels():
    cube, labels, wl = _labeled_scene()
    cube[0, 0] = np.nan  # a class-1 pixel, dropped for being non-finite
    valid = np.ones(labels.shape, dtype=bool)
    valid[1, 0] = False  # another class-1 pixel, dropped by the caller's mask

    lib = endmembers_from_labels(cube, labels, wl, {1: "veg"}, valid=valid, min_pixels=1)

    assert lib.pixel_counts == {"veg": 4}
    np.testing.assert_allclose(lib.endmembers, [[0.1, 0.2, 0.3]])


def test_endmembers_from_labels_drops_class_below_min_pixels_but_reports_it():
    cube, labels, wl = _labeled_scene()

    lib = endmembers_from_labels(cube, labels, wl, {1: "veg", 0: "bg"}, min_pixels=5)

    assert lib.labels == ("veg",)
    assert lib.pixel_counts == {"veg": 6, "bg": 4}


def test_endmembers_from_labels_raises_when_no_class_qualifies():
    cube, labels, wl = _labeled_scene()

    with pytest.raises(ValueError, match="no class reached min_pixels"):
        endmembers_from_labels(cube, labels, wl, {0: "bg"}, min_pixels=99)


def test_endmembers_from_labels_rejects_shape_and_wavelength_mismatch():
    cube, labels, wl = _labeled_scene()

    with pytest.raises(ValueError, match="labels must match"):
        endmembers_from_labels(cube, labels[:2], wl, {1: "veg"})
    with pytest.raises(ValueError, match="wavelengths_nm must be 1-D"):
        endmembers_from_labels(cube, labels, wl[:2], {1: "veg"})
