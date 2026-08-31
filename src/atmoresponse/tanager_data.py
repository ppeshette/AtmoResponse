"""Tanager scene cache naming policy."""

from __future__ import annotations

from typing import Sequence

import requests

from .cache import CacheConfig
from .catalog import SceneAssets, SceneQuery, SceneRecord
from .data import LocalSceneFiles, cache_scene_files as _cache_scene_files

ROLE_FILENAMES = {
    "surface_reflectance": "{scene_id}_ortho_sr.h5",
    "radiance": "{scene_id}_ortho_radiance.h5",
}


def cache_scene_files(
    assets: SceneAssets,
    cache: CacheConfig | None = None,
    session: requests.Session | None = None,
    include_auxiliary: bool = False,
) -> LocalSceneFiles:
    """Resolve a Tanager scene's selected STAC assets into local cache paths."""

    scene_id = assets.scene.scene_id
    role_filenames = {
        role: template.format(scene_id=scene_id)
        for role, template in ROLE_FILENAMES.items()
    }
    return _cache_scene_files(
        assets,
        cache=cache,
        session=session,
        include_auxiliary=include_auxiliary,
        role_filenames=role_filenames,
    )


def fetch_scene(
    scene_id: str,
    cache: CacheConfig | None = None,
    *,
    records: Sequence[SceneRecord] | None = None,
    session: requests.Session | None = None,
) -> LocalSceneFiles:
    """Resolve one Tanager scene id to local surface-reflectance and radiance
    paths, downloading them into ``cache`` on first use.

    This is the one-call form of ``search_scenes`` then ``get_scene_assets`` then
    ``cache_scene_files``. Downloads are cache-first, so an already-local scene
    returns without transferring the files again.

    ``search_scenes`` walks the whole static catalog, so when fetching several
    scenes pass ``records`` from a single prior ``search_scenes`` call and the
    catalog is walked once.
    """
    from . import tanager_catalog

    if records is None:
        records = tanager_catalog.search_scenes(SceneQuery(), session=session)
    try:
        record = next(record for record in records if record.scene_id == scene_id)
    except StopIteration:
        raise KeyError(f"scene id not found in the Tanager catalog: {scene_id}") from None

    assets = tanager_catalog.get_scene_assets(record)
    return cache_scene_files(assets, cache=cache, session=session)
