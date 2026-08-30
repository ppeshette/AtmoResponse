import datetime as dt

import pytest

from atmoresponse.catalog import SceneQuery
from atmoresponse.emit_catalog import (
    CMR_GRANULES_URL,
    EmitProduct,
    get_scene_assets,
    search_scenes,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payload):
        self.payload = payload
        self.headers = {}
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return FakeResponse(self.payload)


def _payload(entry):
    return {"feed": {"entry": [entry]}}


def test_search_scenes_queries_cmr_for_l2a_reflectance():
    session = FakeSession(_payload({
        "id": "G1",
        "title": "EMIT_L2A_RFL_001_20250801T184645_2521312_005",
        "time_start": "2025-08-01T18:46:45.000Z",
        "boxes": ["34.0 -119.0 35.0 -118.0"],
        "links": [
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                "href": "https://example.test/EMIT_L2A_RFL_001_20250801T184645_2521312_005.nc",
            },
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                "href": "https://example.test/EMIT_L2A_MASK_001_20250801T184645_2521312_005.nc",
            },
        ],
    }))
    query = SceneQuery(
        bbox=(-120.0, 33.0, -117.0, 36.0),
        start=dt.datetime(2025, 8, 1, tzinfo=dt.timezone.utc),
        end=dt.datetime(2025, 8, 2, tzinfo=dt.timezone.utc),
    )

    scenes = search_scenes(query, session=session)

    assert session.calls == [(
        CMR_GRANULES_URL,
        {
            "short_name": ("EMITL2ARFL",),
            "page_size": 2000,
            "bounding_box": "-120.0,33.0,-117.0,36.0",
            "temporal": "2025-08-01T00:00:00Z,2025-08-02T00:00:00Z",
        },
        60,
    )]
    assert scenes[0].scene_id == "EMIT_L2A_RFL_001_20250801T184645_2521312_005"
    assert scenes[0].acquired == dt.datetime(2025, 8, 1, 18, 46, 45, tzinfo=dt.timezone.utc)
    assert scenes[0].bbox == (-119.0, 34.0, -118.0, 35.0)
    assert scenes[0].source == "emit"
    assert scenes[0].collections == ("EMITL2ARFL",)
    assert scenes[0].assets["rfl"].endswith("_RFL_001_20250801T184645_2521312_005.nc")


def test_search_scenes_can_query_emit_radiance_product():
    session = FakeSession(_payload({
        "id": "G2",
        "producer_granule_id": "EMIT_L1B_RAD_001_20250801T184645_2521312_005",
        "time_start": "2025-08-01T18:46:45.000Z",
        "boxes": ["34.0 -119.0 35.0 -118.0"],
        "links": [
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                "href": "https://example.test/EMIT_L1B_RAD_001_20250801T184645_2521312_005.nc",
            },
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                "href": "https://example.test/EMIT_L1B_LOC_001_20250801T184645_2521312_005.nc",
            },
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                "href": "https://example.test/EMIT_L1B_OBS_001_20250801T184645_2521312_005.nc",
            },
        ],
    }))

    scene = search_scenes(
        SceneQuery(),
        short_names=(EmitProduct.L1B_RADIANCE,),
        session=session,
    )[0]
    assets = get_scene_assets(scene)

    assert session.calls[0][1]["short_name"] == ("EMITL1BRAD",)
    assert scene.collections == ("EMITL1BRAD",)
    assert assets.radiance == "https://example.test/EMIT_L1B_RAD_001_20250801T184645_2521312_005.nc"
    assert assets.auxiliary == {
        "loc": "https://example.test/EMIT_L1B_LOC_001_20250801T184645_2521312_005.nc",
        "obs": "https://example.test/EMIT_L1B_OBS_001_20250801T184645_2521312_005.nc",
    }


def test_search_scenes_accepts_query_collections_as_emit_products():
    session = FakeSession(_payload({
        "id": "G2",
        "producer_granule_id": "EMIT_L1B_RAD_001_20250801T184645_2521312_005",
        "time_start": "2025-08-01T18:46:45.000Z",
        "boxes": ["34.0 -119.0 35.0 -118.0"],
        "links": [],
    }))

    scene = search_scenes(SceneQuery(collections=("EMITL1BRAD",)), session=session)[0]

    assert session.calls[0][1]["short_name"] == ("EMITL1BRAD",)
    assert scene.collections == ("EMITL1BRAD",)


def test_search_scenes_parses_cmr_polygon_lists():
    session = FakeSession(_payload({
        "id": "G4",
        "title": "EMIT_L2A_RFL_001_20250801T184645_2521312_005",
        "time_start": "2025-08-01T18:46:45.000Z",
        "polygons": [[34.0, -119.0, 34.0, -118.0, 35.0, -118.0, 35.0, -119.0, 34.0, -119.0]],
        "links": [],
    }))

    scene = search_scenes(SceneQuery(), session=session)[0]

    assert scene.bbox == (-119.0, 34.0, -118.0, 35.0)


def test_search_scenes_parses_cmr_polygon_string_lists():
    session = FakeSession(_payload({
        "id": "G5",
        "title": "EMIT_L2A_RFL_001_20250801T184645_2521312_005",
        "time_start": "2025-08-01T18:46:45.000Z",
        "polygons": [["34.0 -119.0 34.0 -118.0 35.0 -118.0 35.0 -119.0 34.0 -119.0"]],
        "links": [],
    }))

    scene = search_scenes(SceneQuery(), session=session)[0]

    assert scene.bbox == (-119.0, 34.0, -118.0, 35.0)


def test_get_scene_assets_maps_emit_l2a_roles():
    session = FakeSession(_payload({
        "id": "G3",
        "title": "EMIT_L2A_RFL_001_20250324T220953_2508314_003",
        "time_start": "2025-03-24T22:09:53.000Z",
        "boxes": ["-1.0 -2.0 1.0 2.0"],
        "links": [
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                "href": "https://example.test/EMIT_L2A_RFL_001_20250324T220953_2508314_003.nc",
            },
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                "href": "https://example.test/EMIT_L2A_RFLUNCERT_001_20250324T220953_2508314_003.nc",
            },
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                "href": "https://example.test/EMIT_L2A_MASK_001_20250324T220953_2508314_003.nc",
            },
        ],
    }))

    assets = get_scene_assets(search_scenes(SceneQuery(), session=session)[0])

    assert assets.surface_reflectance == "https://example.test/EMIT_L2A_RFL_001_20250324T220953_2508314_003.nc"
    assert assets.auxiliary == {
        "rfluncert": "https://example.test/EMIT_L2A_RFLUNCERT_001_20250324T220953_2508314_003.nc",
        "mask": "https://example.test/EMIT_L2A_MASK_001_20250324T220953_2508314_003.nc",
    }


def test_search_scenes_ignores_emit_metadata_sidecars():
    session = FakeSession(_payload({
        "id": "G6",
        "title": "EMIT_L2A_RFL_001_20250801T184645_2521312_005",
        "time_start": "2025-08-01T18:46:45.000Z",
        "boxes": ["34.0 -119.0 35.0 -118.0"],
        "links": [
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                "title": "Download EMIT_L2A_RFL_001_20250801T184645_2521312_005.nc",
                "href": "https://example.test/EMIT_L2A_RFL_001_20250801T184645_2521312_005.nc",
            },
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/metadata#",
                "title": "Download EMIT_L2A_RFL_001_20250801T184645_2521312_005.cmr.json",
                "href": "https://example.test/EMIT_L2A_RFL_001_20250801T184645_2521312_005.cmr.json",
            },
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/s3#",
                "title": "This link provides direct download access via S3 to the granule",
                "href": "s3://example-bucket/EMIT_L2A_MASK_001_20250801T184645_2521312_005.nc",
            },
        ],
    }))

    scene = search_scenes(SceneQuery(), session=session)[0]

    assert scene.assets == {
        "rfl": "https://example.test/EMIT_L2A_RFL_001_20250801T184645_2521312_005.nc",
        "mask": "s3://example-bucket/EMIT_L2A_MASK_001_20250801T184645_2521312_005.nc",
    }


def test_search_scenes_prefers_https_data_links_over_s3_links():
    session = FakeSession(_payload({
        "id": "G7",
        "title": "EMIT_L2A_RFL_001_20250801T184645_2521312_005",
        "time_start": "2025-08-01T18:46:45.000Z",
        "boxes": ["34.0 -119.0 35.0 -118.0"],
        "links": [
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/data#",
                "href": "https://example.test/EMIT_L2A_RFL_001_20250801T184645_2521312_005.nc",
            },
            {
                "rel": "http://esipfed.org/ns/fedsearch/1.1/s3#",
                "href": "s3://example-bucket/EMIT_L2A_RFL_001_20250801T184645_2521312_005.nc",
            },
        ],
    }))

    scene = search_scenes(SceneQuery(), session=session)[0]

    assert scene.assets["rfl"] == "https://example.test/EMIT_L2A_RFL_001_20250801T184645_2521312_005.nc"


def test_search_scenes_rejects_unsupported_cloud_filter():
    with pytest.raises(ValueError, match="max_cloud_percent"):
        search_scenes(SceneQuery(max_cloud_percent=10.0), session=FakeSession({"feed": {"entry": []}}))


def test_empty_cmr_search_returns_no_scenes():
    session = FakeSession({"feed": {"entry": []}})

    assert search_scenes(SceneQuery(), session=session) == []
