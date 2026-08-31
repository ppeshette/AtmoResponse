"""Build a fixed endmember library from a labeled scene."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LabelDerivedLibrary:
    """Per-class mean-reflectance endmembers derived from a labeled scene.

    ``endmembers`` is ``(k, n_bands)`` and ``labels`` names its rows. Feeds any
    endmember method: ``prepare_sam_classifier`` (as its ``library_wavelengths_nm``,
    ``library_reflectance``, and ``labels`` arguments), ``sam_angles`` directly, or a
    mixture model. ``pixel_counts`` maps every requested class to how many usable
    pixels it had, including any that fell below ``min_pixels`` and were left out.
    """

    endmembers: np.ndarray
    labels: tuple[str, ...]
    wavelengths_nm: np.ndarray
    pixel_counts: dict[str, int]


def endmembers_from_labels(
    reflectance,
    labels,
    wavelengths_nm,
    class_map: Mapping,
    *,
    valid=None,
    min_pixels: int = 200,
) -> LabelDerivedLibrary:
    """Build fixed endmembers as the per-class mean reflectance of a labeled scene.

    ``reflectance`` is band-last, ``(..., n_bands)``: a ``(rows, cols, n_bands)``
    cube or a ``(n_pixels, n_bands)`` list. ``labels`` is an integer class array
    shaped like ``reflectance.shape[:-1]``, for instance a land-cover map on the
    scene grid. ``class_map`` maps a label value to an endmember name and its
    insertion order is the output row order. A class with fewer than
    ``min_pixels`` usable pixels is left out of the result rather than raising,
    and its count is still reported in ``pixel_counts``.

    A pixel contributes to its class only where every band is finite and, when
    ``valid`` is given, where ``valid`` is true. ``valid`` is the caller's
    data-quality gate (cloud, nodata, a fill sentinel) and is never derived from
    ``reflectance`` here.

    The endmembers are one atmosphere's worth of spectra. In an AOD-sensitivity
    test they must be derived once and reused unchanged on both AOD sides, so the
    result measures label movement and not a moving library.
    """
    spectra = np.asarray(reflectance, dtype="f8")
    if spectra.ndim < 2:
        raise ValueError("reflectance must be band-last with shape (..., n_bands)")
    wavelengths = np.asarray(wavelengths_nm, dtype="f8")
    if wavelengths.ndim != 1 or wavelengths.size != spectra.shape[-1]:
        raise ValueError("wavelengths_nm must be 1-D and match the last axis of reflectance")
    label_grid = np.asarray(labels)
    if label_grid.shape != spectra.shape[:-1]:
        raise ValueError("labels must match reflectance.shape[:-1]")
    if not class_map:
        raise ValueError("class_map is empty")

    flat = spectra.reshape(-1, spectra.shape[-1])
    flat_labels = label_grid.reshape(-1)
    usable = np.isfinite(flat).all(axis=1)
    if valid is not None:
        valid_flat = np.asarray(valid, dtype=bool).reshape(-1)
        if valid_flat.size != usable.size:
            raise ValueError("valid must match reflectance.shape[:-1]")
        usable = usable & valid_flat

    rows, names, counts = [], [], {}
    for value, name in class_map.items():
        member = usable & (flat_labels == value)
        counts[str(name)] = int(member.sum())
        if counts[str(name)] >= min_pixels:
            rows.append(flat[member].mean(axis=0))
            names.append(str(name))

    if not rows:
        raise ValueError(f"no class reached min_pixels={min_pixels} (pixel counts {counts})")
    return LabelDerivedLibrary(
        endmembers=np.array(rows),
        labels=tuple(names),
        wavelengths_nm=wavelengths.copy(),
        pixel_counts=counts,
    )
