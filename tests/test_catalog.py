import datetime as dt

from atmoresponse import SceneQuery, build_index, get_scene_assets, search_scenes
from atmoresponse.catalog import TANAGER_CATALOG_URL


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = payloads
        self.headers = {}

    def get(self, url, timeout):
        assert timeout == 60
        return FakeResponse(self.payloads[url])


def _item(scene_id, acquired, cloud_percent, lon0, lat0, lon1, lat1, assets):
    return {
        "type": "Feature",
        "id": scene_id,
        "properties": {
            "datetime": acquired,
            "cloud_percent": cloud_percent,
            "platform": "Tanager",
        },
        "assets": {name: {"href": href} for name, href in assets.items()},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [lon0, lat0],
                [lon1, lat0],
                [lon1, lat1],
                [lon0, lat1],
                [lon0, lat0],
            ]],
        },
    }


def _session():
    return FakeSession({
        TANAGER_CATALOG_URL: {
            "id": "tanager-core-imagery",
            "links": [
                {"rel": "child", "href": "fire/catalog.json"},
                {"rel": "child", "href": "water/catalog.json"},
            ],
        },
        "https://www.planet.com/data/stac/tanager-core-imagery/fire/catalog.json": {
            "id": "fire",
            "links": [
                {"rel": "item", "href": "scene-a.json"},
                {"rel": "item", "href": "scene-b.json"},
            ],
        },
        "https://www.planet.com/data/stac/tanager-core-imagery/fire/scene-a.json": _item(
            "scene-a",
            "2025-01-01T10:00:00Z",
            12.0,
            -119.0,
            34.0,
            -118.0,
            35.0,
            {
                "ortho_sr_hdf5": "https://example.test/scene-a-sr.h5",
                "ortho_radiance_hdf5": "https://example.test/scene-a-l1.h5",
                "ortho_ql_ch4": "https://example.test/scene-a-ch4.tif",
            },
        ),
        "https://www.planet.com/data/stac/tanager-core-imagery/fire/scene-b.json": _item(
            "scene-b",
            "2025-02-01T10:00:00Z",
            88.0,
            -10.0,
            10.0,
            -9.0,
            11.0,
            {"ortho_sr_hdf5": "https://example.test/scene-b-sr.h5"},
        ),
        "https://www.planet.com/data/stac/tanager-core-imagery/water/catalog.json": {
            "id": "water",
            "links": [
                {"rel": "item", "href": "../fire/scene-a.json"},
            ],
        },
    })


def test_build_index_walks_static_catalog_and_deduplicates_scenes():
    index = build_index(session=_session())

    assert list(index["id"]) == ["scene-a", "scene-b"]
    assert index.loc[0, "collections"] == ["fire", "water"]
    assert index.loc[0, "assets"]["ortho_sr_hdf5"] == "https://example.test/scene-a-sr.h5"


def test_search_scenes_filters_catalog_records():
    query = SceneQuery(
        bbox=(-120.0, 33.0, -117.0, 36.0),
        start=dt.datetime(2024, 12, 31, tzinfo=dt.timezone.utc),
        end=dt.datetime(2025, 1, 2, tzinfo=dt.timezone.utc),
        max_cloud_percent=20.0,
        collections=("water",),
    )

    scenes = search_scenes(query, session=_session())

    assert [scene.scene_id for scene in scenes] == ["scene-a"]
    assert scenes[0].cloud_percent == 12.0
    assert scenes[0].collections == ("fire", "water")


def test_get_scene_assets_maps_known_tanager_assets():
    scene = search_scenes(SceneQuery(collections=("fire",)), session=_session())[0]

    assets = get_scene_assets(scene)

    assert assets.surface_reflectance == "https://example.test/scene-a-sr.h5"
    assert assets.radiance == "https://example.test/scene-a-l1.h5"
    assert assets.auxiliary == {"ortho_ql_ch4": "https://example.test/scene-a-ch4.tif"}


def test_empty_catalog_returns_no_scenes():
    session = FakeSession({TANAGER_CATALOG_URL: {"id": "empty", "links": []}})

    assert build_index(session=session).empty
    assert search_scenes(SceneQuery(), session=session) == []
