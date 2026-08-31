"""EMIT NetCDF/HDF5 product readers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from .aod import AodSummary, summarize_aod
from .bands import band_index
from .cache import CacheConfig
from .cube import HyperspectralCube

RFL_DATASET = "reflectance"
RAD_DATASET = "radiance"
WAVELENGTH_DATASET = "sensor_band_parameters/wavelengths"
GOOD_WAVELENGTH_DATASET = "sensor_band_parameters/good_wavelengths"
MASK_DATASET = "mask"
MASK_BANDS_DATASET = "sensor_band_parameters/mask_bands"
OBS_DATASET = "obs"
OBS_BANDS_DATASET = "sensor_band_parameters/observation_bands"

# EMIT L1B OBS band-name prefixes for the four angles the LUT lookup needs. The
# full names carry a parenthetical unit description that has changed across
# product builds, so match on the leading phrase only.
_GEOMETRY_BAND_PREFIXES = {
    "sun_z": "to-sun zenith",
    "sun_a": "to-sun azimuth",
    "view_z": "to-sensor zenith",
    "view_a": "to-sensor azimuth",
}


def _check_selector(aoi, rows, cols) -> None:
    if (aoi is not None) and (rows is not None or cols is not None):
        raise ValueError("pass at most one of aoi or rows+cols")
    if (rows is None) != (cols is None):
        raise ValueError("rows and cols must be given together")


def _slice_2d(dataset, aoi=None, rows=None, cols=None) -> np.ndarray:
    _check_selector(aoi, rows, cols)
    if aoi is not None:
        r0, r1, c0, c1 = aoi
        return np.asarray(dataset[r0:r1, c0:c1], dtype="f8")
    if rows is not None:
        return np.asarray(dataset, dtype="f8")[np.asarray(rows), np.asarray(cols)]
    return np.asarray(dataset, dtype="f8")


def _slice_cube(dataset, aoi=None, rows=None, cols=None) -> np.ndarray:
    _check_selector(aoi, rows, cols)
    if aoi is not None:
        r0, r1, c0, c1 = aoi
        return np.asarray(dataset[r0:r1, c0:c1, :], dtype="f8")
    if rows is not None:
        return np.asarray(dataset, dtype="f8")[np.asarray(rows), np.asarray(cols), :]
    return np.asarray(dataset, dtype="f8")


def _fill_value(dataset) -> float | None:
    value = dataset.attrs.get("_FillValue")
    if value is None:
        return None
    return float(np.asarray(value).ravel()[0])


def _replace_fill(values: np.ndarray, fill_value: float | None) -> np.ndarray:
    if fill_value is None:
        return values
    values = values.copy()
    values[values == fill_value] = np.nan
    return values


def wavelengths_nm(h5: h5py.File) -> np.ndarray:
    """Read EMIT wavelength centers in nanometers."""

    return _replace_fill(np.asarray(h5[WAVELENGTH_DATASET], dtype="f8"), _fill_value(h5[WAVELENGTH_DATASET]))


def good_wavelengths(rfl_h5: h5py.File) -> np.ndarray:
    """Read EMIT's usable-wavelength flags."""

    return np.asarray(rfl_h5[GOOD_WAVELENGTH_DATASET], dtype=bool)


def mask_band_names(mask_h5: h5py.File) -> tuple[str, ...]:
    """Read EMIT mask band names."""

    names = []
    for value in mask_h5[MASK_BANDS_DATASET][:]:
        names.append(value.decode() if isinstance(value, bytes) else str(value))
    return tuple(names)


def mask_band_index(mask_h5: h5py.File, name: str) -> int:
    """Return the EMIT mask band index matching ``name``."""

    normalized = name.casefold()
    for index, band_name in enumerate(mask_band_names(mask_h5)):
        if band_name.casefold() == normalized:
            return index
    raise KeyError(f"EMIT mask band not found: {name}")


def mask_band(mask_h5: h5py.File, name: str, aoi=None, rows=None, cols=None) -> np.ndarray:
    """Read one EMIT mask band by name."""

    index = mask_band_index(mask_h5, name)
    values = _slice_2d(mask_h5[MASK_DATASET][:, :, index], aoi=aoi, rows=rows, cols=cols)
    return _replace_fill(values, _fill_value(mask_h5[MASK_DATASET]))


_PRODUCT_FILENAMES = {
    "rfl": "EMIT_L2A_RFL_001_{sid}.nc",
    "rad": "EMIT_L1B_RAD_001_{sid}.nc",
    "obs": "EMIT_L1B_OBS_001_{sid}.nc",
    "mask": "EMIT_L2A_MASK_001_{sid}.nc",
}


def scene_paths(
    scene_id: str, cache: CacheConfig | Path | str | None = None
) -> dict[str, Path]:
    """Expected cached EMIT product paths for one granule id, e.g.
    ``20250221T173656_2505212_021``.

    Returns a dict keyed ``rfl``, ``rad``, ``obs``, and ``mask``. The collection version
    is ``001`` for every EMIT product, so the names are deterministic rather than
    globbed.
    """
    if isinstance(cache, CacheConfig):
        root = cache.child("scenes", scene_id)
    elif cache is None:
        root = CacheConfig.default().child("scenes", scene_id)
    else:
        root = Path(cache) / "scenes" / scene_id
    return {key: root / name.format(sid=scene_id) for key, name in _PRODUCT_FILENAMES.items()}


def observation_band_names(obs_h5: h5py.File) -> tuple[str, ...]:
    """Read EMIT L1B OBS band names."""

    names = []
    for value in obs_h5[OBS_BANDS_DATASET][:]:
        names.append(value.decode() if isinstance(value, bytes) else str(value))
    return tuple(names)


def observation_band_index(obs_h5: h5py.File, prefix: str) -> int:
    """Return the EMIT OBS band index whose name starts with ``prefix``."""

    normalized = prefix.casefold()
    matches = [
        index
        for index, name in enumerate(observation_band_names(obs_h5))
        if name.casefold().startswith(normalized)
    ]
    if len(matches) != 1:
        raise KeyError(f"EMIT OBS band not uniquely matched: {prefix}")
    return matches[0]


def geometry(obs_h5: h5py.File, aoi=None, rows=None, cols=None) -> dict[str, np.ndarray]:
    """Read sun and view geometry (degrees) from the EMIT L1B OBS product.

    Returns a dict with keys ``sun_z``, ``sun_a``, ``view_z``, ``view_a`` (to-sun
    and to-sensor zenith and azimuth). Each value is a 2-D array over the ``aoi``
    block, a 1-D array over the ``rows`` and ``cols`` pixel list, or the full scene
    grid when no selector is given. The keys match ``tanager_ortho.geometry``.
    """

    fill_value = _fill_value(obs_h5[OBS_DATASET])
    result = {}
    for key, prefix in _GEOMETRY_BAND_PREFIXES.items():
        index = observation_band_index(obs_h5, prefix)
        values = _slice_2d(obs_h5[OBS_DATASET][:, :, index], aoi=aoi, rows=rows, cols=cols)
        result[key] = _replace_fill(values, fill_value)
    return result


def surface_reflectance_at(
    rfl_h5: h5py.File,
    targets_nm: Sequence[float],
    aoi=None,
    rows=None,
    cols=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read nearest-band EMIT surface reflectance for requested wavelengths."""

    wl = wavelengths_nm(rfl_h5)
    indices = np.array([band_index(wl, target) for target in targets_nm], dtype=int)
    cube = _slice_cube(rfl_h5[RFL_DATASET], aoi=aoi, rows=rows, cols=cols)[..., indices]
    cube = _replace_fill(cube, _fill_value(rfl_h5[RFL_DATASET]))
    return wl[indices], cube


def radiance_at(
    rad_h5: h5py.File,
    targets_nm: Sequence[float],
    aoi=None,
    rows=None,
    cols=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read nearest-band EMIT L1B radiance for requested wavelengths."""

    wl = wavelengths_nm(rad_h5)
    indices = np.array([band_index(wl, target) for target in targets_nm], dtype=int)
    cube = _slice_cube(rad_h5[RAD_DATASET], aoi=aoi, rows=rows, cols=cols)[..., indices]
    cube = _replace_fill(cube, _fill_value(rad_h5[RAD_DATASET]))
    return wl[indices], cube


def reflectance_cube(
    rfl_h5: h5py.File,
    aoi=None,
    rows=None,
    cols=None,
    valid_mask=None,
    metadata: Mapping[str, object] | None = None,
) -> HyperspectralCube:
    """Read an EMIT surface-reflectance cube."""

    values = _replace_fill(_slice_cube(rfl_h5[RFL_DATASET], aoi=aoi, rows=rows, cols=cols), _fill_value(rfl_h5[RFL_DATASET]))
    base_metadata = {
        "source": "emit",
        "quantity": "surface_reflectance",
        "good_wavelengths": good_wavelengths(rfl_h5),
    }
    if metadata is not None:
        base_metadata.update(metadata)
    return HyperspectralCube(
        values=values,
        wavelengths_nm=wavelengths_nm(rfl_h5),
        mask=valid_mask,
        metadata=base_metadata,
    )


def radiance_cube(
    rad_h5: h5py.File,
    aoi=None,
    rows=None,
    cols=None,
    metadata: Mapping[str, object] | None = None,
) -> HyperspectralCube:
    """Read an EMIT L1B radiance cube."""

    values = _replace_fill(_slice_cube(rad_h5[RAD_DATASET], aoi=aoi, rows=rows, cols=cols), _fill_value(rad_h5[RAD_DATASET]))
    base_metadata = {
        "source": "emit",
        "quantity": "radiance",
    }
    if metadata is not None:
        base_metadata.update(metadata)
    return HyperspectralCube(
        values=values,
        wavelengths_nm=wavelengths_nm(rad_h5),
        metadata=base_metadata,
    )


def aod550(mask_h5: h5py.File, aoi=None, rows=None, cols=None) -> np.ndarray:
    """Read EMIT's delivered AOD550 mask band."""

    return mask_band(mask_h5, "AOD550", aoi=aoi, rows=rows, cols=cols)


def h2o(mask_h5: h5py.File, aoi=None, rows=None, cols=None) -> np.ndarray:
    """Read EMIT's delivered water vapor mask band."""

    return mask_band(mask_h5, "H2O (g cm-2)", aoi=aoi, rows=rows, cols=cols)


def shipped_aod_summary(mask_h5: h5py.File, aoi=None, rows=None, cols=None, valid_mask=None) -> AodSummary:
    """Summarize EMIT's delivered AOD550 for selected pixels."""

    return summarize_aod(
        aod550(mask_h5, aoi=aoi, rows=rows, cols=cols),
        valid_mask=valid_mask,
        detail="EMIT shipped AOD550",
    )
