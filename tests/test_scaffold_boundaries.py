import datetime as dt

import pytest

from atmoresponse import (
    AodQuery,
    CacheConfig,
    SceneQuery,
    get_scene_assets,
    resolve_aod,
    search_scenes,
)


def test_cache_default_has_project_name(monkeypatch):
    monkeypatch.delenv("ATMORESPONSE_CACHE", raising=False)

    cache = CacheConfig.default()

    assert cache.root.name == "atmoresponse"


def test_cache_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ATMORESPONSE_CACHE", str(tmp_path))

    cache = CacheConfig.default()

    assert cache.root == tmp_path


def test_live_boundaries_are_explicit_placeholders():
    scene_query = SceneQuery(start=dt.datetime(2025, 1, 1), end=dt.datetime(2025, 1, 2))
    aod_query = AodQuery(latitude=34.0, longitude=-118.0, when=dt.datetime(2025, 1, 1))

    with pytest.raises(NotImplementedError, match="STAC search"):
        search_scenes(scene_query)
    with pytest.raises(NotImplementedError, match="AOD resolution"):
        resolve_aod(aod_query)
    with pytest.raises(NotImplementedError, match="asset resolution"):
        get_scene_assets(None)

