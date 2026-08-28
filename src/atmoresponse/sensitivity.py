"""Public sensitivity API for user-supplied reflectance algorithms."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .aod import AodEstimate
from .catalog import SceneAssets

Reflectance = Mapping[float, float]


@dataclass(frozen=True)
class LabeledScore:
    """A classifier output with the numeric score used for differencing."""

    value: float
    label: str
    margin: float = float("nan")


Algorithm = Callable[[Reflectance], float | LabeledScore]


@dataclass(frozen=True)
class SensitivityResult:
    """Potential and Realized Sensitivity outputs for one scene and algorithm."""

    values_at_scene_aod: np.ndarray
    values_at_reference_aod: np.ndarray
    delta: np.ndarray
    labels_at_scene_aod: np.ndarray | None = None
    labels_at_reference_aod: np.ndarray | None = None
    class_changed: np.ndarray | None = None
    metadata: Mapping[str, object] | None = None


def evaluate_sensitivity(
    scene_assets: SceneAssets,
    algorithm: Algorithm,
    reference_aod: AodEstimate,
    wavelengths_um: Sequence[float] | None = None,
) -> SensitivityResult:
    """Evaluate a user-supplied algorithm at scene and reference AOD assumptions."""

    _ = (scene_assets, algorithm, reference_aod, wavelengths_um)
    raise NotImplementedError("Sensitivity evaluation has not been ported into AtmoResponse yet.")
