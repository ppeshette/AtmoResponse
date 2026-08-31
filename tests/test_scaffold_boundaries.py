import datetime as dt

import atmoresponse.aod as aod_module
from atmoresponse.aod import AodEstimate, AodQuery, AodSource, resolve_aod
from atmoresponse.storage import default_data_dir


def test_data_dir_default_has_project_name(monkeypatch):
    monkeypatch.delenv("ATMORESPONSE_DATA", raising=False)

    assert default_data_dir().name == "atmoresponse_data"


def test_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("ATMORESPONSE_DATA", str(tmp_path))

    assert default_data_dir() == tmp_path


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
        lambda sources: {AodSource.AERONET: lambda query, data_dir: reference},
    )

    assert resolve_aod(aod_query) is reference
