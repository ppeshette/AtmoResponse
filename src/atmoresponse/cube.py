"""Source-neutral hyperspectral cube container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .bands import band_index


@dataclass(frozen=True)
class HyperspectralCube:
    """Band-last hyperspectral data with wavelength and optional context arrays."""

    values: np.ndarray
    wavelengths_nm: np.ndarray
    mask: np.ndarray | None = None
    geometry: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype="f8")
        wavelengths = np.asarray(self.wavelengths_nm, dtype="f8")
        if values.ndim < 2:
            raise ValueError("values must have at least one sample dimension and one band dimension")
        if wavelengths.ndim != 1:
            raise ValueError("wavelengths_nm must be one-dimensional")
        if values.shape[-1] != wavelengths.size:
            raise ValueError(
                f"values band dimension {values.shape[-1]} does not match "
                f"{wavelengths.size} wavelengths"
            )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "wavelengths_nm", wavelengths)

        sample_shape = values.shape[:-1]
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != sample_shape:
                raise ValueError(f"mask shape {mask.shape} does not match sample shape {sample_shape}")
            object.__setattr__(self, "mask", mask)

        geometry = {name: np.asarray(array, dtype="f8") for name, array in self.geometry.items()}
        for name, array in geometry.items():
            if array.shape != sample_shape:
                raise ValueError(
                    f"geometry[{name!r}] shape {array.shape} does not match sample shape {sample_shape}"
                )
        object.__setattr__(self, "geometry", geometry)

    @property
    def sample_shape(self) -> tuple[int, ...]:
        """Shape of the non-band dimensions."""

        return self.values.shape[:-1]

    @property
    def band_count(self) -> int:
        """Number of spectral bands."""

        return int(self.values.shape[-1])

    def nearest_band(self, wavelength_nm: float) -> tuple[float, np.ndarray]:
        """Return the nearest wavelength and its data plane."""

        index = band_index(self.wavelengths_nm, wavelength_nm)
        return float(self.wavelengths_nm[index]), self.values[..., index]

    def subset_wavelengths(self, targets_nm) -> "HyperspectralCube":
        """Return a cube containing nearest bands for requested wavelengths."""

        indices = np.array([band_index(self.wavelengths_nm, target) for target in targets_nm], dtype=int)
        return HyperspectralCube(
            values=self.values[..., indices],
            wavelengths_nm=self.wavelengths_nm[indices],
            mask=self.mask,
            geometry=self.geometry,
            metadata=self.metadata,
        )
