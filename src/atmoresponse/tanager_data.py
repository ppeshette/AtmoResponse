"""Tanager scene cache naming policy."""

from __future__ import annotations

import requests

from .cache import CacheConfig
from .catalog import SceneAssets
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
