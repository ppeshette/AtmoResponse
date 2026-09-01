"""Mask composition helpers for scene-level AtmoResponse runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import h5py
import numpy as np

from . import lut, tanager_ortho
from .recipes import agriculture, land_mask, water


def _as_bool_mask(mask, *, name: str = "mask") -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim == 0:
        raise ValueError(f"{name} must be at least one-dimensional")
    return values.astype(bool)


def _same_shape(masks: tuple[np.ndarray, ...]) -> None:
    shape = masks[0].shape
    for mask in masks[1:]:
        if mask.shape != shape:
            raise ValueError(f"mask shapes must match: {shape} != {mask.shape}")


def _whole_scene_aoi(sr_h5: h5py.File) -> tuple[int, int, int, int]:
    rows, cols = sr_h5[tanager_ortho.GRID + "aerosol_optical_depth"].shape
    return 0, rows, 0, cols


def _default_aoi(sr_h5: h5py.File, aoi, rows, cols):
    if aoi is None and rows is None and cols is None:
        return _whole_scene_aoi(sr_h5)
    return aoi


def combine_all(*masks) -> np.ndarray:
    """Return the logical intersection of same-shaped masks."""

    if not masks:
        raise ValueError("pass at least one mask")
    arrays = tuple(_as_bool_mask(mask, name=f"mask[{i}]") for i, mask in enumerate(masks))
    _same_shape(arrays)
    return np.logical_and.reduce(arrays)


def combine_any(*masks) -> np.ndarray:
    """Return the logical union of same-shaped masks."""

    if not masks:
        raise ValueError("pass at least one mask")
    arrays = tuple(_as_bool_mask(mask, name=f"mask[{i}]") for i, mask in enumerate(masks))
    _same_shape(arrays)
    return np.logical_or.reduce(arrays)


def erode(mask, *, pixels: int = 1) -> np.ndarray:
    """Erode a 2-D boolean mask by requiring a full square neighborhood."""

    if pixels < 0:
        raise ValueError("pixels must be >= 0")
    out = _as_bool_mask(mask).copy()
    if out.ndim != 2:
        raise ValueError("erode expects a 2-D mask")
    for _ in range(pixels):
        rows, cols = out.shape
        padded = np.pad(out, 1, constant_values=False)
        neighbors = [
            padded[r : r + rows, c : c + cols]
            for r in range(3)
            for c in range(3)
        ]
        out = np.logical_and.reduce(neighbors)
    return out


def tanager_clear(sr_h5: h5py.File, aoi=None, rows=None, cols=None) -> np.ndarray:
    """Tanager cloud, cirrus, and nodata validity mask."""

    aoi = _default_aoi(sr_h5, aoi, rows, cols)
    return tanager_ortho.land_valid_mask(sr_h5, aoi=aoi, rows=rows, cols=cols)


def aod_in_lut(
    sr_h5: h5py.File,
    *,
    axes=None,
    sensor: str = "tanager",
    aoi=None,
    rows=None,
    cols=None,
) -> np.ndarray:
    """Pixels whose shipped AOD falls within the LUT AOD axis."""

    aoi = _default_aoi(sr_h5, aoi, rows, cols)
    axis_def = lut.load_axes(sensor=sensor) if axes is None else axes
    aod_nodes = lut.axis_values(axis_def, "aod")
    shipped = tanager_ortho.shipped_aod(sr_h5, aoi=aoi, rows=rows, cols=cols)
    return np.isfinite(shipped) & (shipped >= aod_nodes.min()) & (shipped <= aod_nodes.max())


def finite_tanager_inputs(
    scene_id: str,
    aoi: tuple[int, int, int, int],
    band_targets_nm: Sequence[float],
    *,
    data_dir: str | Path | None = None,
) -> np.ndarray:
    """Pixels with finite Tanager L1 radiance and geometry for a run."""

    _, l1_path = tanager_ortho.scene_paths(scene_id, data_dir)
    with h5py.File(l1_path, "r") as l1_h5:
        _, radiance = tanager_ortho.radiance_at(l1_h5, band_targets_nm, aoi=aoi)
        geometry = tanager_ortho.geometry(l1_h5, aoi=aoi)
    valid = np.isfinite(radiance).all(axis=-1)
    for values in geometry.values():
        valid &= np.isfinite(values)
    return valid


def tanager_water(
    sr_h5: h5py.File,
    *,
    aoi=None,
    threshold: float = 0.0,
    erode_pixels: int = 2,
    screen_clear: bool = True,
) -> np.ndarray:
    """MNDWI water candidates, optionally screened by Tanager quality masks."""

    aoi = _default_aoi(sr_h5, aoi, None, None)
    wavelengths, reflectance = tanager_ortho.reflectance_at(
        sr_h5,
        [water.GREEN_NM, water.SWIR_NM],
        aoi=aoi,
    )
    selected = water.water_candidate(reflectance, wavelengths, threshold=threshold)
    if screen_clear:
        selected = combine_all(selected, tanager_clear(sr_h5, aoi=aoi))
    return erode(selected, pixels=erode_pixels)


def tanager_vegetation(
    sr_h5: h5py.File,
    *,
    aoi=None,
    savi_threshold: float = land_mask.SAVI_CANOPY_MIN,
    screen_clear: bool = True,
) -> np.ndarray:
    """Canopy-presence candidates, optionally screened by Tanager quality masks.

    Wraps :func:`atmoresponse.recipes.land_mask.canopy_mask`: SAVI (Huete 1988)
    and a Dawson and Curran (1998) red edge position must agree. A corroboration
    screen for where a canopy reflectance algorithm is meaningful, not a
    land-cover product.
    """

    aoi = _default_aoi(sr_h5, aoi, None, None)
    all_wavelengths = tanager_ortho.wavelengths_nm(sr_h5, "surface_reflectance")
    lo, hi = land_mask.RED_EDGE_WINDOW_NM
    window = all_wavelengths[(all_wavelengths >= lo) & (all_wavelengths <= hi)]
    targets = np.unique(np.concatenate(
        ([land_mask.CANOPY_RED_NM, land_mask.CANOPY_NIR_NM], window)))
    wavelengths, reflectance = tanager_ortho.reflectance_at(sr_h5, targets, aoi=aoi)
    selected = land_mask.canopy_mask(reflectance, wavelengths, savi_threshold=savi_threshold)
    if screen_clear:
        selected = combine_all(selected, tanager_clear(sr_h5, aoi=aoi))
    return selected


def tanager_land(
    sr_h5: h5py.File,
    *,
    aoi=None,
    exclude_water: bool = True,
) -> np.ndarray:
    """Clear Tanager pixels, optionally excluding MNDWI water candidates."""

    aoi = _default_aoi(sr_h5, aoi, None, None)
    selected = tanager_clear(sr_h5, aoi=aoi)
    if exclude_water:
        selected = combine_all(selected, ~tanager_water(sr_h5, aoi=aoi, erode_pixels=0))
    return selected


def tanager_admissible(
    sr_h5: h5py.File,
    *,
    aoi: tuple[int, int, int, int],
    scene_id: str,
    band_targets_nm: Sequence[float],
    data_dir: str | Path | None = None,
    selector: Callable[[h5py.File, tuple[int, int, int, int]], np.ndarray] | None = None,
    axes=None,
    sensor: str = "tanager",
) -> np.ndarray:
    """Combine a recipe/domain selector with LUT and L1 input-validity gates."""

    if selector is None:
        base = np.ones_like(tanager_ortho.shipped_aod(sr_h5, aoi=aoi), dtype=bool)
    else:
        base = selector(sr_h5, aoi)
    return combine_all(
        base,
        aod_in_lut(sr_h5, aoi=aoi, axes=axes, sensor=sensor),
        finite_tanager_inputs(scene_id, aoi, band_targets_nm, data_dir=data_dir),
    )


def admissible(
    scene_id: str,
    band_targets_nm: Sequence[float],
    domain: Callable[..., np.ndarray] | None = None,
    *,
    data_dir: str | Path | None = None,
) -> Callable[[h5py.File, tuple[int, int, int, int]], np.ndarray]:
    """Build the ``mask`` argument for ``run_tanager`` in one call.

    ``domain`` is an analysis-target selector such as :func:`tanager_water` or
    :func:`tanager_vegetation` (or any ``(sr_h5, *, aoi) -> bool array``). The
    returned mask intersects it with the Tanager cloud, cirrus, and nodata
    screen, the LUT AOD-coverage check, and the finite-radiance check, so a
    section only has to name its target.
    """

    def mask(sr_h5: h5py.File, aoi: tuple[int, int, int, int]) -> np.ndarray:
        return tanager_admissible(
            sr_h5,
            aoi=aoi,
            scene_id=scene_id,
            band_targets_nm=band_targets_nm,
            data_dir=data_dir,
            selector=None if domain is None else (lambda h5, window: domain(h5, aoi=window)),
        )

    return mask
