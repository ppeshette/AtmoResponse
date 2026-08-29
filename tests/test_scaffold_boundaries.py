import datetime as dt

import pytest

from atmoresponse.aod import AodQuery, resolve_aod
from atmoresponse.cache import CacheConfig


def test_cache_default_has_project_name(monkeypatch):
    monkeypatch.delenv("ATMORESPONSE_CACHE", raising=False)

    cache = CacheConfig.default()

    assert cache.root.name == "atmoresponse"


def test_cache_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ATMORESPONSE_CACHE", str(tmp_path))

    cache = CacheConfig.default()

    assert cache.root == tmp_path


def test_aod_boundary_is_explicit_placeholder():
    aod_query = AodQuery(latitude=34.0, longitude=-118.0, when=dt.datetime(2025, 1, 1))

    with pytest.raises(NotImplementedError, match="AOD resolution"):
        resolve_aod(aod_query)
