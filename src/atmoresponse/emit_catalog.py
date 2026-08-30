"""EMIT CMR catalog interface."""

from __future__ import annotations

import datetime as dt
from enum import Enum
from pathlib import PurePosixPath
from typing import Iterable, Sequence
from urllib.parse import urlparse

import requests

from .catalog import SceneAssets, SceneQuery, SceneRecord

CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
L1B_RAD_SHORT_NAME = "EMITL1BRAD"
L2A_RFL_SHORT_NAME = "EMITL2ARFL"


class EmitProduct(str, Enum):
    """CMR short names for EMIT products used by AtmoResponse."""

    L1B_RADIANCE = L1B_RAD_SHORT_NAME
    L2A_REFLECTANCE = L2A_RFL_SHORT_NAME


def _utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _format_time(value: dt.datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: str) -> dt.datetime:
    if value.endswith("Z"):
        value = f"{value[:-1]}+00:00"
    return dt.datetime.fromisoformat(value)


def _short_name(value: EmitProduct | str) -> str:
    if isinstance(value, EmitProduct):
        return value.value
    return value


def _asset_role(href: str, title: str | None = None) -> str:
    text = f"{_url_path_name(href)} {title or ''}".upper()
    if "_L2A_RFLUNCERT_" in text:
        return "rfluncert"
    if "_L2A_MASK_" in text:
        return "mask"
    if "_L2A_RFL_" in text:
        return "rfl"
    if "_L1B_RAD_" in text:
        return "rad"
    if "_L1B_OBS_" in text:
        return "obs"
    if "_L1B_LOC_" in text:
        return "loc"
    return _url_path_name(href)


def _url_path_name(href: str) -> str:
    return PurePosixPath(urlparse(href).path).name


def _is_product_data_link(link: dict) -> bool:
    href = link.get("href") or ""
    rel = link.get("rel") or ""
    name = _url_path_name(href).lower()
    return name.endswith(".nc") and rel.endswith(("/data#", "/s3#"))


def _asset_priority(link: dict) -> int:
    href = link.get("href") or ""
    rel = link.get("rel") or ""
    if rel.endswith("/data#") and href.startswith(("http://", "https://")):
        return 2
    if rel.endswith("/s3#") and href.startswith("s3://"):
        return 1
    return 0


def _assets(entry: dict) -> dict[str, str]:
    assets: dict[str, str] = {}
    priorities: dict[str, int] = {}
    for link in entry.get("links", []):
        href = link.get("href")
        if not href or link.get("inherited") or not _is_product_data_link(link):
            continue
        role = _asset_role(href, link.get("title"))
        priority = _asset_priority(link)
        if priority > priorities.get(role, -1):
            assets[role] = href
            priorities[role] = priority
    return assets


def _bbox(entry: dict) -> tuple[float, float, float, float]:
    boxes = entry.get("boxes") or []
    if boxes:
        south, west, north, east = (float(value) for value in boxes[0].split())
        return (west, south, east, north)

    polygons = entry.get("polygons") or []
    if polygons:
        values = _polygon_values(polygons[0])
        lats = values[0::2]
        lons = values[1::2]
        return (min(lons), min(lats), max(lons), max(lats))

    return (float("nan"), float("nan"), float("nan"), float("nan"))


def _polygon_values(polygon) -> list[float]:
    if isinstance(polygon, str):
        return [float(value) for value in polygon.split()]
    if polygon and isinstance(polygon[0], list):
        return [float(value) for ring in polygon for value in ring]
    if polygon and isinstance(polygon[0], str):
        return [float(value) for ring in polygon for value in ring.split()]
    return [float(value) for value in polygon]


def _cmr_params(
    query: SceneQuery,
    short_names: Iterable[EmitProduct | str],
) -> dict[str, object]:
    if query.max_cloud_percent is not None:
        raise ValueError("EMIT CMR search does not support max_cloud_percent")

    names = tuple(_short_name(name) for name in short_names)
    if not names:
        raise ValueError("at least one EMIT short name is required")

    params: dict[str, object] = {
        "short_name": names,
        "page_size": 2000,
    }
    if query.bbox is not None:
        params["bounding_box"] = ",".join(str(value) for value in query.bbox)
    if query.start is not None or query.end is not None:
        start = _format_time(query.start) if query.start is not None else ""
        end = _format_time(query.end) if query.end is not None else ""
        params["temporal"] = f"{start},{end}"
    return params


def _get(session: requests.Session, url: str, params: dict[str, object]) -> dict:
    response = session.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def _entry_short_name(entry: dict, default: str) -> str:
    scene_id = entry.get("producer_granule_id") or entry.get("title") or ""
    if scene_id.startswith("EMIT_L1B_RAD_"):
        return L1B_RAD_SHORT_NAME
    if scene_id.startswith("EMIT_L2A_RFL_"):
        return L2A_RFL_SHORT_NAME
    return default


def _as_record(entry: dict, short_name: str) -> SceneRecord:
    props = {
        key: entry[key]
        for key in ("title", "time_start", "time_end", "updated", "day_night_flag")
        if key in entry
    }
    return SceneRecord(
        scene_id=entry.get("producer_granule_id") or entry.get("title") or entry["id"],
        acquired=_parse_datetime(entry["time_start"]),
        bbox=_bbox(entry),
        source="emit",
        collections=(_entry_short_name(entry, short_name),),
        cloud_percent=None,
        assets=_assets(entry),
        properties=props,
    )


def search_scenes(
    query: SceneQuery,
    short_names: Sequence[EmitProduct | str] | None = None,
    cmr_url: str = CMR_GRANULES_URL,
    session: requests.Session | None = None,
) -> Sequence[SceneRecord]:
    """Search NASA CMR for EMIT scene records."""

    session = session or requests.Session()
    session.headers.setdefault("User-Agent", "atmoresponse")
    names = tuple(short_names) if short_names is not None else (query.collections or (EmitProduct.L2A_REFLECTANCE,))
    payload = _get(session, cmr_url, _cmr_params(query, names))
    entries = payload.get("feed", {}).get("entry", [])
    return [_as_record(entry, _short_name(names[0])) for entry in entries]


def get_scene_assets(scene: SceneRecord) -> SceneAssets:
    """Resolve EMIT scene assets for downstream extraction."""

    auxiliary = {
        name: href
        for name, href in scene.assets.items()
        if name not in {"rfl", "rad", "metadata"}
    }
    return SceneAssets(
        scene=scene,
        surface_reflectance=scene.assets.get("rfl"),
        radiance=scene.assets.get("rad"),
        metadata=scene.assets.get("metadata"),
        auxiliary=auxiliary,
    )
