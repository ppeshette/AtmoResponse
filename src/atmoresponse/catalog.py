"""Tanager public STAC catalog interface."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Mapping, Sequence
from urllib.parse import urljoin

import geopandas as gpd
import requests
from shapely.geometry import box, shape

from .cache import CacheConfig

ROOT_CATALOG_URL = "https://www.planet.com/data/stac/catalog.json"
TANAGER_CATALOG_URL = "https://www.planet.com/data/stac/tanager-core-imagery/catalog.json"

_PROP_COLUMNS = (
    "datetime",
    "cloud_percent",
    "light_haze_percent",
    "quality_category",
    "gsd",
    "view:off_nadir",
    "view:azimuth",
    "view:sun_azimuth",
    "view:sun_elevation",
    "collection_mode",
    "platform",
    "location_description",
)


@dataclass(frozen=True)
class SceneQuery:
    """Inputs for a Tanager catalog search."""

    bbox: tuple[float, float, float, float] | None = None
    start: dt.datetime | None = None
    end: dt.datetime | None = None
    max_cloud_percent: float | None = None
    collections: tuple[str, ...] = ()


@dataclass(frozen=True)
class SceneRecord:
    """Metadata needed to choose and retrieve one Tanager scene."""

    scene_id: str
    acquired: dt.datetime
    bbox: tuple[float, float, float, float]
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


def _get(session: requests.Session, url: str) -> dict:
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.json()


def _link_href(base_url: str, href: str) -> str:
    return href if href.startswith(("http://", "https://")) else urljoin(base_url, href)


def _walk_item_links(session: requests.Session, node_url: str):
    """Yield ``(collection_id, item_url)`` for every item below one STAC node."""

    node = _get(session, node_url)
    node_id = node.get("id")
    for link in node.get("links", []):
        rel = link.get("rel")
        href = link.get("href")
        if not href:
            continue
        linked_url = _link_href(node_url, href)
        if rel == "item":
            yield node_id, linked_url
        elif rel == "child":
            yield from _walk_item_links(session, linked_url)


def _parse_datetime(value: str) -> dt.datetime:
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    return dt.datetime.fromisoformat(value)


def _utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _scene_id_from_url(url: str) -> str:
    return url.rstrip("/").split("/")[-1].removesuffix(".json")


def _as_record(row) -> SceneRecord:
    props = {name: row.get(name) for name in _PROP_COLUMNS if name in row and row.get(name) is not None}
    return SceneRecord(
        scene_id=row["id"],
        acquired=_parse_datetime(row["datetime"]),
        bbox=tuple(row.geometry.bounds),
        collections=tuple(row["collections"]),
        cloud_percent=row.get("cloud_percent"),
        assets=dict(row["assets"]),
        properties=props,
        geometry=row.geometry,
    )


def build_index(
    catalog_url: str = TANAGER_CATALOG_URL,
    session: requests.Session | None = None,
) -> gpd.GeoDataFrame:
    """Walk the static Tanager STAC catalog and return one row per unique scene."""

    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "atmoresponse")

    collections: dict[str, set[str]] = {}
    item_url: dict[str, str] = {}
    for collection_id, url in _walk_item_links(session, catalog_url):
        scene_id = _scene_id_from_url(url)
        collections.setdefault(scene_id, set()).add(collection_id)
        item_url.setdefault(scene_id, url)

    records = []
    for scene_id, url in item_url.items():
        item = _get(session, url)
        props = item.get("properties", {})
        records.append({
            "id": scene_id,
            "collections": sorted(collections[scene_id]),
            **{name: props.get(name) for name in _PROP_COLUMNS},
            "assets": {name: asset.get("href") for name, asset in item.get("assets", {}).items()},
            "geometry": shape(item["geometry"]),
        })

    if not records:
        return gpd.GeoDataFrame(records, geometry=[], crs="EPSG:4326")

    index = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    return index.sort_values("datetime").reset_index(drop=True)


def search_scenes(
    query: SceneQuery,
    cache: CacheConfig | None = None,
    catalog_url: str = TANAGER_CATALOG_URL,
    session: requests.Session | None = None,
) -> Sequence[SceneRecord]:
    """Search the public Tanager catalog and return matching scene records."""

    _ = cache
    index = build_index(catalog_url=catalog_url, session=session)
    if index.empty:
        return []

    mask = index["datetime"].notna()
    acquired = index["datetime"].map(lambda value: _utc(_parse_datetime(value)))
    if query.start is not None:
        mask &= acquired >= _utc(query.start)
    if query.end is not None:
        mask &= acquired <= _utc(query.end)
    if query.max_cloud_percent is not None:
        mask &= index["cloud_percent"].notna() & (index["cloud_percent"] <= query.max_cloud_percent)
    if query.collections:
        wanted = set(query.collections)
        mask &= index["collections"].map(lambda values: bool(wanted.intersection(values)))
    if query.bbox is not None:
        query_geometry = box(*query.bbox)
        mask &= index.geometry.intersects(query_geometry)

    return [_as_record(row) for _, row in index.loc[mask].iterrows()]


def get_scene_assets(
    scene: SceneRecord,
    cache: CacheConfig | None = None,
) -> SceneAssets:
    """Resolve scene assets for downstream extraction."""

    _ = cache
    auxiliary = {
        name: href
        for name, href in scene.assets.items()
        if name not in {"ortho_sr_hdf5", "ortho_radiance_hdf5", "metadata"}
    }
    return SceneAssets(
        scene=scene,
        surface_reflectance=scene.assets.get("ortho_sr_hdf5"),
        radiance=scene.assets.get("ortho_radiance_hdf5"),
        metadata=scene.assets.get("metadata"),
        auxiliary=auxiliary,
    )
