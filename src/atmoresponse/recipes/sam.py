"""Fixed-library spectral-angle mapping primitives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from atmoresponse.sensitivity import LabeledScore


@dataclass(frozen=True)
class SamResult:
    """Angles and nearest fixed-library row for each input spectrum."""

    angles: np.ndarray
    class_index: np.ndarray


def resample_spectrum(source_wavelengths_nm, source_reflectance, target_wavelengths_nm) -> np.ndarray:
    """Linearly resample one fixed spectrum onto target wavelengths in nm."""

    source_wavelengths = np.asarray(source_wavelengths_nm, dtype="f8")
    source_values = np.asarray(source_reflectance, dtype="f8")
    target_wavelengths = np.asarray(target_wavelengths_nm, dtype="f8")
    if source_wavelengths.ndim != 1 or target_wavelengths.ndim != 1:
        raise ValueError("wavelengths must be one-dimensional")
    if source_values.ndim != 1 or source_values.size != source_wavelengths.size:
        raise ValueError("source_reflectance must match source_wavelengths_nm")
    if not np.isfinite(source_wavelengths).all() or not np.isfinite(target_wavelengths).all():
        raise ValueError("wavelengths must be finite")
    if not np.isfinite(source_values).all():
        raise ValueError("source_reflectance must be finite")
    if np.any(np.diff(source_wavelengths) <= 0):
        raise ValueError("source_wavelengths_nm must be strictly increasing")
    if target_wavelengths.size and (
        target_wavelengths.min() < source_wavelengths[0]
        or target_wavelengths.max() > source_wavelengths[-1]
    ):
        raise ValueError("target wavelengths extend beyond library coverage")
    return np.interp(target_wavelengths, source_wavelengths, source_values)


def resample_library(library_wavelengths_nm, library_reflectance, target_wavelengths_nm) -> np.ndarray:
    """Linearly resample fixed-library rows onto target wavelengths in nm."""

    source_wavelengths = np.asarray(library_wavelengths_nm, dtype="f8")
    spectra = np.asarray(library_reflectance, dtype="f8")
    if spectra.ndim != 2 or spectra.shape[1] != source_wavelengths.size:
        raise ValueError("library_reflectance must have shape (n_entries, n_wavelengths)")
    return np.array([
        resample_spectrum(source_wavelengths, spectrum, target_wavelengths_nm)
        for spectrum in spectra
    ])


def sam_angles(spectra, endmembers) -> np.ndarray:
    """Return SAM angles for ``(..., bands)`` spectra vs ``(entries, bands)``."""

    observed = np.asarray(spectra, dtype="f8")
    library = np.asarray(endmembers, dtype="f8")
    if observed.ndim < 1 or library.ndim != 2 or observed.shape[-1] != library.shape[1]:
        raise ValueError("spectra and endmembers must share their final band dimension")

    sample_shape = observed.shape[:-1]
    flat = observed.reshape(-1, observed.shape[-1])
    result = np.full((flat.shape[0], library.shape[0]), np.nan, dtype="f8")

    library_norm = np.linalg.norm(library, axis=1)
    spectrum_norm = np.linalg.norm(flat, axis=1)
    valid_library = np.isfinite(library).all(axis=1) & (library_norm > 0)
    valid_spectra = np.isfinite(flat).all(axis=1) & (spectrum_norm > 0)
    valid = valid_spectra[:, None] & valid_library[None, :]

    denominator = spectrum_norm[:, None] * library_norm[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.sum(flat[:, None, :] * library[None, :, :], axis=2) / denominator
    result[valid] = np.arccos(np.clip(cosine[valid], -1.0, 1.0))
    return result.reshape(*sample_shape, library.shape[0])


def classify_by_angle(spectra, endmembers) -> SamResult:
    """Return all angles and the nearest fixed-library index for each spectrum."""

    angles = sam_angles(spectra, endmembers)
    flat = angles.reshape(-1, angles.shape[-1])
    indices = np.full(flat.shape[0], -1, dtype=int)
    valid = np.isfinite(flat).any(axis=1)
    indices[valid] = np.nanargmin(flat[valid], axis=1)
    return SamResult(angles=angles, class_index=indices.reshape(angles.shape[:-1]))


def _labels_array(labels: Sequence[str], group_labels: Sequence[str] | None = None) -> np.ndarray:
    label_array = np.asarray(labels, dtype=object)
    if label_array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if group_labels is None:
        return label_array

    grouped = np.asarray(group_labels, dtype=object)
    if grouped.ndim != 1 or grouped.shape != label_array.shape:
        raise ValueError("group_labels must match labels")
    return grouped


def labels_for_indices(
    class_index,
    labels: Sequence[str],
    invalid_label: str | None = None,
    group_labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Map SAM class indices to labels, preserving ``-1`` as ``invalid_label``."""

    label_array = _labels_array(labels, group_labels)
    indices = np.asarray(class_index, dtype=int)
    out = np.full(indices.shape, invalid_label, dtype=object)
    valid = (indices >= 0) & (indices < label_array.size)
    out[valid] = label_array[indices[valid]]
    return out


def labeled_sam_score(
    spectrum,
    endmembers,
    labels: Sequence[str],
    target_index: int,
    group_labels: Sequence[str] | None = None,
) -> LabeledScore:
    """Return a ``LabeledScore`` for one spectrum against a fixed SAM library."""

    angles = np.asarray(sam_angles(spectrum, endmembers), dtype="f8")
    if angles.ndim != 1:
        raise ValueError("spectrum must be one-dimensional")
    if len(labels) != angles.size:
        raise ValueError("labels must match the number of endmembers")
    label_array = _labels_array(labels, group_labels)
    if target_index < 0 or target_index >= angles.size:
        raise ValueError("target_index is outside the endmember library")

    finite = np.flatnonzero(np.isfinite(angles))
    if finite.size == 0:
        return LabeledScore(value=float("nan"), label="invalid", margin=float("nan"))
    order = finite[np.argsort(angles[finite])]
    margin = float(angles[order[1]] - angles[order[0]]) if order.size > 1 else float("nan")
    return LabeledScore(value=float(angles[target_index]), label=str(label_array[order[0]]), margin=margin)
