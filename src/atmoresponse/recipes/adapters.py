"""Adapt a fixed-band recipe function to the sensitivity engine's algorithm form.

The recipe functions in this package take ``(reflectance, wavelengths_nm)`` with
reflectance band-last and vectorised over pixels.
:func:`atmoresponse.sensitivity.run_tanager` instead passes each pixel as a
``{wavelength_nm: reflectance}`` mapping. :func:`as_algorithm` bridges the two and
keeps the vectorised fast path by also exposing ``evaluate_many``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


class _RecipeAlgorithm:
    """A recipe function wrapped as a sensitivity-engine algorithm."""

    def __init__(self, recipe: Callable, result_index: int | None):
        self._recipe = recipe
        self._result_index = result_index

    def _select(self, output):
        if self._result_index is not None:
            output = output[self._result_index]
        return np.asarray(output, dtype="f8")

    def __call__(self, spectrum: Mapping[float, float]) -> float:
        wavelengths = np.fromiter(sorted(spectrum), dtype="f8")
        reflectance = np.array([spectrum[key] for key in wavelengths], dtype="f8")
        return float(self._select(self._recipe(reflectance, wavelengths)))

    def evaluate_many(self, spectra: Sequence[Mapping[float, float]]) -> list[float]:
        wavelengths = np.fromiter(sorted(spectra[0]), dtype="f8")
        block = np.array([[row[key] for key in wavelengths] for row in spectra], dtype="f8")
        return [float(value) for value in self._select(self._recipe(block, wavelengths))]


def as_algorithm(recipe: Callable, *, result_index: int | None = None) -> _RecipeAlgorithm:
    """Wrap a ``(reflectance, wavelengths_nm) -> array`` recipe as an algorithm
    that :func:`atmoresponse.sensitivity.run_tanager` accepts directly.

    ``result_index`` picks one element when the recipe returns a tuple, for
    example ``as_algorithm(cyanobacteria_index, result_index=0)`` for the index
    itself.
    """
    return _RecipeAlgorithm(recipe, result_index)
