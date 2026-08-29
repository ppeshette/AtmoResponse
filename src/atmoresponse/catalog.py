"""Source-neutral scene catalog data structures."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class SceneQuery:
    """Inputs for a scene catalog search."""

    bbox: tuple[float, float, float, float] | None = None
    start: dt.datetime | None = None
    end: dt.datetime | None = None
    max_cloud_percent: float | None = None
    collections: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneRecord:
    """Metadata needed to choose and retrieve one scene."""

    scene_id: str
    acquired: dt.datetime
    bbox: tuple[float, float, float, float]
    source: str = ""
    collections: tuple[str, ...] = ()
    cloud_percent: float | None = None
    assets: Mapping[str, str] = field(default_factory=dict)
    properties: Mapping[str, object] = field(default_factory=dict)
    geometry: object | None = None


@dataclass(frozen=True)
class SceneAssets:
    """Resolved asset URLs or local paths for one scene."""

    scene: SceneRecord
    surface_reflectance: str | None = None
    radiance: str | None = None
    metadata: str | None = None
    auxiliary: Mapping[str, str] = field(default_factory=dict)
