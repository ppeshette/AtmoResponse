"""Fixed-library wildfire SAM example assets and wrappers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files

import numpy as np

from atmoresponse.cube import HyperspectralCube
from atmoresponse.sensitivity import LabeledScore

from .sam import (
    SamResult,
    classify_by_angle,
    labeled_sam_score,
    labels_for_indices,
    resample_library,
)

WILDFIRE_SAM_TARGET_LABEL = "burned_surface_mixed"
_ASSET_PACKAGE = "atmoresponse.assets.endmembers"
_LIBRARY_ASSET = "wildfire_vnir_sam_curated.npz"
_MANIFEST_ASSET = "wildfire_vnir_sam_curated.json"
_FILL_LIMIT = -900.0


@dataclass(frozen=True)
class WildfireSamLibrary:
    """Public fixed wildfire endmember library used by the Malibu SAM example."""

    wavelengths_nm: np.ndarray
    reflectance: np.ndarray
    labels: tuple[str, ...]
    group_labels: tuple[str, ...]
    material_classes: tuple[str, ...]
    source_entries: tuple[str, ...]
    entry_sources: tuple[str, ...]
    metadata: Mapping[str, object]
    target_label: str = WILDFIRE_SAM_TARGET_LABEL

    @property
    def target_index(self) -> int:
        """Index of the library row used as the burned-surface score target."""

        return self.labels.index(self.target_label)


def _tuple(values) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def load_wildfire_sam_library() -> WildfireSamLibrary:
    """Load the bundled 400-1300 nm wildfire SAM endmember library."""

    package_files = files(_ASSET_PACKAGE)
    with package_files.joinpath(_MANIFEST_ASSET).open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with package_files.joinpath(_LIBRARY_ASSET).open("rb") as handle:
        with np.load(handle, allow_pickle=False) as library:
            return WildfireSamLibrary(
                wavelengths_nm=library["wavelengths_nm"].copy(),
                reflectance=library["reflectance"].copy(),
                labels=_tuple(library["labels"]),
                group_labels=_tuple(library["group_labels"]),
                material_classes=_tuple(library["material_classes"]),
                source_entries=_tuple(library["source_entries"]),
                entry_sources=_tuple(library["entry_sources"]),
                metadata=metadata,
                target_label=str(metadata.get("target_label", WILDFIRE_SAM_TARGET_LABEL)),
            )


def _library_band_mask(wavelengths_nm, library: WildfireSamLibrary) -> np.ndarray:
    wavelengths = np.asarray(wavelengths_nm, dtype="f8")
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths_nm must be one-dimensional")
    mask = (wavelengths >= library.wavelengths_nm[0]) & (wavelengths <= library.wavelengths_nm[-1])
    if not mask.any():
        raise ValueError("no wavelengths fall within the wildfire SAM library range")
    return mask


def _prepared_values(values, mask=None) -> np.ndarray:
    prepared = np.asarray(values, dtype="f8").copy()
    prepared[prepared <= _FILL_LIMIT] = np.nan
    if mask is not None:
        valid = np.asarray(mask, dtype=bool)
        if valid.shape != prepared.shape[:-1]:
            raise ValueError("mask shape must match the non-band dimensions")
        prepared = np.where(valid[..., None], prepared, np.nan)
    return prepared


def classify_wildfire_sam(
    cube: HyperspectralCube,
    library: WildfireSamLibrary | None = None,
) -> SamResult:
    """Classify a band-last reflectance cube against the bundled wildfire library.

    The cube should already be limited to physically meaningful post-fire land pixels through
    ``cube.mask`` or caller-side filtering. Open water and unrelated surfaces are not screened here.
    """

    library = library or load_wildfire_sam_library()
    band_mask = _library_band_mask(cube.wavelengths_nm, library)
    wavelengths = cube.wavelengths_nm[band_mask]
    endmembers = resample_library(library.wavelengths_nm, library.reflectance, wavelengths)
    values = _prepared_values(cube.values[..., band_mask], cube.mask)
    return classify_by_angle(values, endmembers)


def wildfire_sam_labels(
    class_index,
    library: WildfireSamLibrary | None = None,
    *,
    grouped: bool = True,
    invalid_label: str | None = None,
) -> np.ndarray:
    """Map wildfire SAM class indices to raw library labels or burned/unburned groups."""

    library = library or load_wildfire_sam_library()
    group_labels = library.group_labels if grouped else None
    return labels_for_indices(class_index, library.labels, invalid_label, group_labels)


def wildfire_sam_score(
    spectrum,
    wavelengths_nm,
    library: WildfireSamLibrary | None = None,
) -> LabeledScore:
    """Score one spectrum by angle to the burned-surface target row."""

    library = library or load_wildfire_sam_library()
    band_mask = _library_band_mask(wavelengths_nm, library)
    spectrum = np.asarray(spectrum, dtype="f8")
    if spectrum.ndim != 1 or spectrum.size != band_mask.size:
        raise ValueError("spectrum must be one-dimensional and match wavelengths_nm")
    wavelengths = np.asarray(wavelengths_nm, dtype="f8")[band_mask]
    values = _prepared_values(spectrum[band_mask])
    endmembers = resample_library(library.wavelengths_nm, library.reflectance, wavelengths)
    return labeled_sam_score(
        values,
        endmembers,
        library.labels,
        target_index=library.target_index,
        group_labels=library.group_labels,
    )
