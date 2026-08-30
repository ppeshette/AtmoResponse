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


@dataclass(frozen=True)
class PreparedSamClassifier:
    """Fixed-library SAM classifier prepared for one wavelength grid."""

    wavelengths_nm: np.ndarray
    band_mask: np.ndarray
    endmembers: np.ndarray
    labels_raw: tuple[str, ...]
    target_index: int
    group_labels: tuple[str, ...] | None = None
    fill_limit: float | None = None

    @property
    def selected_wavelengths_nm(self) -> np.ndarray:
        """Wavelengths retained for SAM scoring."""

        return self.wavelengths_nm[self.band_mask]

    def _values(self, values, mask=None) -> np.ndarray:
        array = np.asarray(values, dtype="f8")
        if array.ndim < 1 or array.shape[-1] != self.band_mask.size:
            raise ValueError("values must be band-last and match the prepared wavelengths")
        prepared = array[..., self.band_mask].copy()
        if self.fill_limit is not None:
            prepared[prepared <= self.fill_limit] = np.nan
        if mask is not None:
            valid = np.asarray(mask, dtype=bool)
            if valid.shape != prepared.shape[:-1]:
                raise ValueError("mask shape must match the non-band dimensions")
            prepared = np.where(valid[..., None], prepared, np.nan)
        return prepared

    def classify_values(self, values, mask=None) -> SamResult:
        """Classify a band-last value array using pre-resampled endmembers."""

        return classify_by_angle(self._values(values, mask), self.endmembers)

    def labels(
        self,
        class_index,
        *,
        grouped: bool = True,
        invalid_label: str | None = None,
    ) -> np.ndarray:
        """Map SAM class indices to raw labels or grouped labels."""

        group_labels = self.group_labels if grouped else None
        return labels_for_indices(class_index, self.labels_raw, invalid_label, group_labels)

    def evaluate(self, spectrum) -> LabeledScore:
        """Score one spectrum by angle to the target library row."""

        return _labeled_score_from_angles(
            sam_angles(self._values(spectrum), self.endmembers),
            self.labels_raw,
            self.target_index,
            self.group_labels,
        )

    def __call__(self, spectrum) -> LabeledScore:
        """Callable alias for sensitivity runners that accept scalar algorithms."""

        return self.evaluate(spectrum)

    def evaluate_many(self, spectra) -> list[LabeledScore]:
        """Score many spectra in one vectorized SAM pass."""

        angles = sam_angles(self._values(spectra), self.endmembers).reshape(
            -1,
            len(self.labels_raw),
        )
        return [
            _labeled_score_from_angles(
                row,
                self.labels_raw,
                self.target_index,
                self.group_labels,
            )
            for row in angles
        ]


def resample_spectrum(
    source_wavelengths_nm,
    source_reflectance,
    target_wavelengths_nm,
) -> np.ndarray:
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


def resample_library(
    library_wavelengths_nm,
    library_reflectance,
    target_wavelengths_nm,
) -> np.ndarray:
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


def _library_band_mask(wavelengths_nm, library_wavelengths_nm, band_stride: int) -> np.ndarray:
    wavelengths = np.asarray(wavelengths_nm, dtype="f8")
    library_wavelengths = np.asarray(library_wavelengths_nm, dtype="f8")
    if wavelengths.ndim != 1 or library_wavelengths.ndim != 1:
        raise ValueError("wavelengths must be one-dimensional")
    if library_wavelengths.size == 0:
        raise ValueError("library_wavelengths_nm must not be empty")
    if band_stride < 1:
        raise ValueError("band_stride must be at least 1")
    mask = (
        (wavelengths >= library_wavelengths[0])
        & (wavelengths <= library_wavelengths[-1])
    )
    if not mask.any():
        raise ValueError("no wavelengths fall within the SAM library range")
    selected = np.flatnonzero(mask)
    if band_stride > 1:
        mask[:] = False
        mask[selected[::band_stride]] = True
    return mask


def _labeled_score_from_angles(
    angles,
    labels: Sequence[str],
    target_index: int,
    group_labels: Sequence[str] | None = None,
) -> LabeledScore:
    angle_array = np.asarray(angles, dtype="f8")
    if angle_array.ndim != 1:
        raise ValueError("angles must be one-dimensional")
    if len(labels) != angle_array.size:
        raise ValueError("labels must match the number of endmembers")
    label_array = _labels_array(labels, group_labels)
    if target_index < 0 or target_index >= angle_array.size:
        raise ValueError("target_index is outside the endmember library")

    finite = np.flatnonzero(np.isfinite(angle_array))
    if finite.size == 0:
        return LabeledScore(value=float("nan"), label="invalid", margin=float("nan"))
    order = finite[np.argsort(angle_array[finite])]
    margin = (
        float(angle_array[order[1]] - angle_array[order[0]])
        if order.size > 1
        else float("nan")
    )
    return LabeledScore(
        value=float(angle_array[target_index]),
        label=str(label_array[order[0]]),
        margin=margin,
    )


def prepare_sam_classifier(
    wavelengths_nm,
    library_wavelengths_nm,
    library_reflectance,
    labels: Sequence[str],
    target_index: int,
    *,
    group_labels: Sequence[str] | None = None,
    fill_limit: float | None = None,
    band_stride: int = 1,
) -> PreparedSamClassifier:
    """Prepare a fixed SAM library once for repeated runs on a wavelength grid."""

    label_array = _labels_array(labels, group_labels)
    library_reflectance = np.asarray(library_reflectance, dtype="f8")
    if library_reflectance.ndim != 2:
        raise ValueError("library_reflectance must have shape (n_entries, n_wavelengths)")
    if label_array.size != library_reflectance.shape[0]:
        raise ValueError("labels must match the number of endmembers")
    if target_index < 0 or target_index >= label_array.size:
        raise ValueError("target_index is outside the endmember library")

    wavelengths = np.asarray(wavelengths_nm, dtype="f8")
    band_mask = _library_band_mask(wavelengths, library_wavelengths_nm, band_stride)
    endmembers = resample_library(
        library_wavelengths_nm,
        library_reflectance,
        wavelengths[band_mask],
    )
    grouped = None if group_labels is None else tuple(str(label) for label in group_labels)
    return PreparedSamClassifier(
        wavelengths_nm=wavelengths.copy(),
        band_mask=band_mask,
        endmembers=endmembers,
        labels_raw=tuple(str(label) for label in labels),
        target_index=target_index,
        group_labels=grouped,
        fill_limit=fill_limit,
    )


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
    return _labeled_score_from_angles(angles, labels, target_index, group_labels)
