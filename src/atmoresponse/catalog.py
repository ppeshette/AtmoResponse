"""Tanager public STAC catalog interface."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .cache import CacheConfig


@dataclass(frozen=True)
class SceneQuery:
    """Inputs for a Tanager catalog search."""

    bbox: tuple[float, float, float, float] | None = None
    start: dt.datetime | None = None
    end: dt.datetime | None = None
    max_cloud_fraction: float | None = None
    collections: tuple[str, ...] = ("tanager",)


@dataclass(frozen=True)
class SceneRecord:
    """Metadata needed to choose and retrieve one Tanager scene."""

    scene_id: str
    acquired: dt.datetime
    bbox: tuple[float, float, float, float]
    cloud_fraction: float | None = None
    assets: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneAssets:
    """Resolved asset URLs or local paths for one scene."""

    scene: SceneRecord
    surface_reflectance: str | None = None
    radiance: str | None = None
    metadata: str | None = None
    auxiliary: Mapping[str, str] = field(default_factory=dict)


def search_scenes(
    query: SceneQuery,
    cache: CacheConfig | None = None,
) -> Sequence[SceneRecord]:
    """Search the public Tanager catalog.

    The implementation will be ported from the private prototype after the module inventory
    decides which STAC fields belong in the public API.
    """

    _ = (query, cache)
    raise NotImplementedError("Tanager STAC search has not been ported into AtmoResponse yet.")


def get_scene_assets(
    scene: SceneRecord,
    cache: CacheConfig | None = None,
) -> SceneAssets:
    """Resolve scene assets for downstream extraction."""

    _ = (scene, cache)
    raise NotImplementedError("Scene asset resolution has not been ported into AtmoResponse yet.")

