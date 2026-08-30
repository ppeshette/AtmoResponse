"""Source-neutral scene asset localization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse

import requests

from .cache import CacheConfig
from .catalog import SceneAssets, SceneRecord
from .downloads import CHUNK_SIZE, DownloadResult, download_file


@dataclass(frozen=True)
class LocalSceneFiles:
    """Local file paths for scene assets."""

    scene: SceneRecord
    surface_reflectance: Path | None = None
    radiance: Path | None = None
    metadata: Path | None = None
    auxiliary: Mapping[str, Path] = field(default_factory=dict)


def _url_basename(url: str) -> str:
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    if not name:
        raise ValueError(f"cannot derive a filename from URL: {url}")
    return name


def _asset_path(
    cache: CacheConfig,
    scene_id: str,
    role: str,
    url: str,
    role_filenames: Mapping[str, str] | None,
) -> Path:
    if role_filenames is not None and role in role_filenames:
        return cache.child("scenes", scene_id, role_filenames[role])
    return cache.child("scenes", scene_id, _url_basename(url))


def cache_scene_files(
    assets: SceneAssets,
    cache: CacheConfig | None = None,
    session: requests.Session | None = None,
    include_auxiliary: bool = False,
    role_filenames: Mapping[str, str] | None = None,
) -> LocalSceneFiles:
    """Download a scene's selected assets into local cache paths."""

    cache = cache or CacheConfig.default()
    session = session or requests.Session()
    scene_id = assets.scene.scene_id

    surface_reflectance = None
    if assets.surface_reflectance is not None:
        surface_reflectance = download_file(
            assets.surface_reflectance,
            _asset_path(cache, scene_id, "surface_reflectance", assets.surface_reflectance, role_filenames),
            session=session,
        ).path

    radiance = None
    if assets.radiance is not None:
        radiance = download_file(
            assets.radiance,
            _asset_path(cache, scene_id, "radiance", assets.radiance, role_filenames),
            session=session,
        ).path

    metadata = None
    if assets.metadata is not None:
        metadata = download_file(
            assets.metadata,
            _asset_path(cache, scene_id, "metadata", assets.metadata, role_filenames),
            session=session,
        ).path

    auxiliary = {}
    if include_auxiliary:
        for name, url in assets.auxiliary.items():
            auxiliary[name] = download_file(
                url,
                _asset_path(cache, scene_id, name, url, role_filenames),
                session=session,
            ).path

    return LocalSceneFiles(
        scene=assets.scene,
        surface_reflectance=surface_reflectance,
        radiance=radiance,
        metadata=metadata,
        auxiliary=auxiliary,
    )


def local_scene_files(
    scene: SceneRecord,
    *,
    surface_reflectance: str | Path | None = None,
    radiance: str | Path | None = None,
    metadata: str | Path | None = None,
    auxiliary: Mapping[str, str | Path] | None = None,
) -> LocalSceneFiles:
    """Wrap already-local scene asset paths without downloading them."""

    return LocalSceneFiles(
        scene=scene,
        surface_reflectance=Path(surface_reflectance) if surface_reflectance is not None else None,
        radiance=Path(radiance) if radiance is not None else None,
        metadata=Path(metadata) if metadata is not None else None,
        auxiliary={name: Path(path) for name, path in (auxiliary or {}).items()},
    )
