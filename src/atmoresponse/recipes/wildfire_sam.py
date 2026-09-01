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
    PreparedSamClassifier,
    SamResult,
    labels_for_indices,
    prepare_sam_classifier,
)

WILDFIRE_SAM_TARGET_LABEL = "burned_surface_mixed"
_ASSET_PACKAGE = "atmoresponse.assets.endmembers"
_LIBRARY_ASSET = "wildfire_vnir_sam_curated.npz"
_MANIFEST_ASSET = "wildfire_vnir_sam_curated.json"
_FILL_LIMIT = -900.0


@dataclass(frozen=True)
class WildfireSamLibrary:
    """Public fixed wildfire endmember library used by the wildfire SAM example."""

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


def load_palisades_fire_perimeter() -> np.ndarray:
    """The January 2025 Palisades fire perimeter (WFIGS, public), rasterized onto
    the Malibu Tanager scene ``20250123_185518_92_4001`` as a scene-shaped boolean.
    Pass it as ``sensitivity_figure``'s ``scoring_region`` to compare the wildfire
    SAM inside the burn against the rest of the scene."""

    with files(_ASSET_PACKAGE).joinpath("palisades_fire_perimeter.npz").open("rb") as handle:
        with np.load(handle, allow_pickle=False) as data:
            return data["mask"].copy()


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


def prepare_wildfire_sam(
    wavelengths_nm,
    library: WildfireSamLibrary | None = None,
    *,
    band_stride: int = 1,
) -> PreparedSamClassifier:
    """Prepare the wildfire SAM library once for repeated runs on a wavelength grid."""

    library = library or load_wildfire_sam_library()
    return prepare_sam_classifier(
        wavelengths_nm,
        library.wavelengths_nm,
        library.reflectance,
        library.labels,
        library.target_index,
        group_labels=library.group_labels,
        fill_limit=_FILL_LIMIT,
        band_stride=band_stride,
    )


def classify_wildfire_sam(
    cube: HyperspectralCube,
    library: WildfireSamLibrary | None = None,
) -> SamResult:
    """Classify a band-last reflectance cube against the bundled wildfire library.

    The cube should already be limited to physically meaningful post-fire land pixels through
    ``cube.mask`` or caller-side filtering. Open water and unrelated surfaces are not screened here.
    """

    prepared = prepare_wildfire_sam(cube.wavelengths_nm, library)
    return prepared.classify_values(cube.values, cube.mask)


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
    *,
    band_stride: int = 1,
) -> LabeledScore:
    """Score one spectrum by angle to the burned-surface target row."""

    return prepare_wildfire_sam(wavelengths_nm, library, band_stride=band_stride).evaluate(spectrum)
