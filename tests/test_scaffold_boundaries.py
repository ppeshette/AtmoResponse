import datetime as dt

import atmoresponse.aod as aod_module
from atmoresponse.aod import AodEstimate, AodQuery, AodSource, resolve_aod
from atmoresponse.cache import CacheConfig


def test_cache_default_has_project_name(monkeypatch):
    monkeypatch.delenv("ATMORESPONSE_CACHE", raising=False)

    cache = CacheConfig.default()

    assert cache.root.name == "atmoresponse"


def test_cache_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ATMORESPONSE_CACHE", str(tmp_path))

    cache = CacheConfig.default()

    assert cache.root == tmp_path


def test_aod_boundary_uses_built_in_providers(monkeypatch):
    aod_query = AodQuery(latitude=34.0, longitude=-118.0, when=dt.datetime(2025, 1, 1))
    reference = AodEstimate(
        value=0.12,
        source=AodSource.AERONET,
        independence="measurement",
        distance_km=10.0,
        dt_minutes=0.0,
        detail="fixture",
    )
    monkeypatch.setattr(
        aod_module,
        "default_providers",
        lambda sources: {AodSource.AERONET: lambda query, cache: reference},
    )

    assert resolve_aod(aod_query) is reference
