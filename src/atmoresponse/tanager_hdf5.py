"""Tanager HDF5 extraction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import h5py
import numpy as np

from .aod import AodSummary, summarize_aod
from .bands import band_index
from .cache import CacheConfig

GRID = "HDFEOS/GRIDS/HYP/Data Fields/"


def scene_paths(scene_id: str, cache: CacheConfig | Path | str | None = None) -> tuple[Path, Path]:
    """Return the expected cached Tanager SR and radiance HDF5 paths for one scene."""

    if isinstance(cache, CacheConfig):
        root = cache.child("scenes", scene_id)
    elif cache is None:
        root = CacheConfig.default().child("scenes", scene_id)
    else:
        root = Path(cache) / "scenes" / scene_id
    return root / f"{scene_id}_ortho_sr.h5", root / f"{scene_id}_ortho_radiance.h5"


def validate_aoi(sr_h5: h5py.File, aoi: tuple[int, int, int, int]) -> None:
    """Raise ``ValueError`` if ``aoi`` does not fit inside the Tanager SR scene."""

    nrows, ncols = sr_h5[GRID + "aerosol_optical_depth"].shape
    r0, r1, c0, c1 = aoi
    if r0 < 0 or c0 < 0:
        raise ValueError(f"aoi={aoi}: negative start index")
    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"aoi={aoi}: empty or reversed")
    if r1 > nrows or c1 > ncols:
        raise ValueError(f"aoi={aoi}: exceeds scene extent rows=0..{nrows}, cols=0..{ncols}")


def _check_selector(aoi, rows, cols) -> None:
    if (aoi is None) == (rows is None and cols is None):
        raise ValueError("pass exactly one of aoi or rows+cols")
    if (rows is None) != (cols is None):
        raise ValueError("rows and cols must be given together")


def _slice_2d(dataset, aoi=None, rows=None, cols=None) -> np.ndarray:
    _check_selector(aoi, rows, cols)
    if aoi is not None:
        r0, r1, c0, c1 = aoi
        return np.asarray(dataset[r0:r1, c0:c1], dtype="f8")
    return np.asarray(dataset, dtype="f8")[np.asarray(rows), np.asarray(cols)]


def _slice_cube(dataset, band_indices, aoi=None, rows=None, cols=None) -> np.ndarray:
    _check_selector(aoi, rows, cols)
    band_indices = np.asarray(band_indices, dtype=int)
    if aoi is not None:
        r0, r1, c0, c1 = aoi
        cube = np.asarray(dataset[band_indices, r0:r1, c0:c1], dtype="f8")
        return np.moveaxis(cube, 0, -1)

    rows, cols = np.asarray(rows), np.asarray(cols)
    out = np.full((len(band_indices), rows.size), np.nan)
    for j, band in enumerate(band_indices):
        out[j] = np.asarray(dataset[int(band)], dtype="f8")[rows, cols]
    return np.moveaxis(out, 0, -1)


def wavelengths_nm(h5: h5py.File, dataset: str = "surface_reflectance") -> np.ndarray:
    """Read wavelength centers from a Tanager band cube."""

    return np.asarray(h5[GRID + dataset].attrs["wavelengths"], dtype="f8")


def geometry(l1_h5: h5py.File, aoi=None, rows=None, cols=None) -> dict[str, np.ndarray]:
    """Read Tanager sun and view geometry arrays."""

    fields = (
        ("sun_z", "sun_zenith"),
        ("sun_a", "sun_azimuth"),
        ("view_z", "sensor_zenith"),
        ("view_a", "sensor_azimuth"),
    )
    return {name: _slice_2d(l1_h5[GRID + field], aoi, rows, cols) for name, field in fields}


def land_valid_mask(sr_h5: h5py.File, aoi=None, rows=None, cols=None) -> np.ndarray:
    """Return the Tanager cloud, cirrus, and nodata validity mask for land scenes."""

    cloud = _slice_2d(sr_h5[GRID + "beta_cloud_mask"], aoi, rows, cols)
    cirrus = _slice_2d(sr_h5[GRID + "beta_cirrus_mask"], aoi, rows, cols)
    nodata = _slice_2d(sr_h5[GRID + "nodata_pixels"], aoi, rows, cols)
    return (cloud == 0) & (cirrus == 0) & (nodata == 0)


def shipped_aod(sr_h5: h5py.File, aoi=None, rows=None, cols=None) -> np.ndarray:
    """Read Tanager's delivered aerosol optical depth."""

    return _slice_2d(sr_h5[GRID + "aerosol_optical_depth"], aoi, rows, cols)


def shipped_aod_summary(
    sr_h5: h5py.File,
    aoi=None,
    rows=None,
    cols=None,
    valid_mask=None,
) -> AodSummary:
    """Summarize Tanager HDF5 delivered aerosol optical depth for selected pixels."""

    values = shipped_aod(sr_h5, aoi=aoi, rows=rows, cols=cols)
    return summarize_aod(values, valid_mask=valid_mask, detail="Tanager shipped aerosol_optical_depth")


def column_water_vapour(sr_h5: h5py.File, aoi=None, rows=None, cols=None) -> np.ndarray:
    """Read Tanager's delivered column water vapour."""

    return _slice_2d(sr_h5[GRID + "column_water_vapour"], aoi, rows, cols)


def radiance_window(
    l1_h5: h5py.File,
    wl_lo_nm: float,
    wl_hi_nm: float,
    aoi=None,
    rows=None,
    cols=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a continuous Tanager radiance band window with the band axis last."""

    wl = wavelengths_nm(l1_h5, dataset="toa_radiance")
    band_indices = np.flatnonzero((wl >= wl_lo_nm) & (wl <= wl_hi_nm))
    radiance = _slice_cube(l1_h5[GRID + "toa_radiance"], band_indices, aoi, rows, cols)
    return wl[band_indices], radiance


def radiance_at(
    l1_h5: h5py.File,
    targets_nm: Sequence[float],
    aoi=None,
    rows=None,
    cols=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read nearest-band Tanager radiance for each requested wavelength."""

    wl = wavelengths_nm(l1_h5, dataset="toa_radiance")
    band_indices = np.array([band_index(wl, target) for target in targets_nm], dtype=int)
    radiance = _slice_cube(l1_h5[GRID + "toa_radiance"], band_indices, aoi, rows, cols)
    return wl[band_indices], radiance


def reflectance_at(
    sr_h5: h5py.File,
    targets_nm: Sequence[float],
    aoi=None,
    rows=None,
    cols=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read nearest-band Tanager surface reflectance for each requested wavelength."""

    wl = wavelengths_nm(sr_h5, dataset="surface_reflectance")
    band_indices = np.array([band_index(wl, target) for target in targets_nm], dtype=int)
    reflectance = _slice_cube(sr_h5[GRID + "surface_reflectance"], band_indices, aoi, rows, cols)
    return wl[band_indices], reflectance
