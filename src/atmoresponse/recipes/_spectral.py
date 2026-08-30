"""Shared spectral helpers for recipe modules."""

from __future__ import annotations

import numpy as np

from atmoresponse.bands import band_index

FILL_LIMIT = -900.0


def validate_spectra(reflectance, wavelengths_nm) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays after validating band-last spectra against wavelengths."""

    spectra = np.asarray(reflectance, dtype="f8")
    wavelengths = np.asarray(wavelengths_nm, dtype="f8")
    if wavelengths.ndim != 1:
        raise ValueError("wavelengths_nm must be one-dimensional")
    if spectra.ndim < 1 or spectra.shape[-1] != wavelengths.size:
        raise ValueError("reflectance must have wavelengths_nm along its final dimension")
    if not np.isfinite(wavelengths).all() or np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelengths_nm must be finite and strictly increasing")
    return spectra, wavelengths


def nearest_reflectance(reflectance, wavelengths_nm, target_nm: float) -> np.ndarray:
    """Nearest-band reflectance at ``target_nm``, with fill values replaced by NaN."""

    spectra, wavelengths = validate_spectra(reflectance, wavelengths_nm)
    values = spectra[..., band_index(wavelengths, target_nm)]
    return np.where(values <= FILL_LIMIT, np.nan, values)


def sample_linear(reflectance, wavelengths_nm, target_nm: float) -> np.ndarray:
    """Linearly sample band-last spectra at ``target_nm``."""

    spectra, wavelengths = validate_spectra(reflectance, wavelengths_nm)
    right = int(np.searchsorted(wavelengths, target_nm, side="left"))
    if right < wavelengths.size and wavelengths[right] == target_nm:
        return spectra[..., right]
    if right == 0 or right == wavelengths.size:
        raise ValueError(f"{target_nm} nm is outside the supplied wavelength range")
    left = right - 1
    fraction = (target_nm - wavelengths[left]) / (wavelengths[right] - wavelengths[left])
    return spectra[..., left] * (1.0 - fraction) + spectra[..., right] * fraction
