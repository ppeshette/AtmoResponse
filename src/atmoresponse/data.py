"""Cache-backed scene file access."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping
from urllib.parse import unquote, urlparse

import requests

from .cache import CacheConfig
from .catalog import SceneAssets, SceneRecord

CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class DownloadResult:
    """Result of resolving one URL into the local cache."""

    path: Path
    downloaded: bool
    size_bytes: int | None = None


@dataclass(frozen=True)
class CachedSceneFiles:
    """Local file paths for the scene assets used by AtmoResponse."""

    scene: SceneRecord
    surface_reflectance: Path | None = None
    radiance: Path | None = None
    metadata: Path | None = None
    auxiliary: Mapping[str, Path] = field(default_factory=dict)


def _content_length(session: requests.Session, url: str) -> int | None:
    response = session.head(url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    value = response.headers.get("Content-Length")
    return int(value) if value is not None else None


def _is_complete(path: Path, expected_size: int | None) -> bool:
    if not path.exists():
        return False
    return expected_size is None or path.stat().st_size == expected_size


def _stream_download(
    session: requests.Session,
    url: str,
    destination: Path,
    chunk_size: int,
) -> int:
    bytes_written = 0
    temporary = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.part")
    with session.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                bytes_written += len(chunk)
    temporary.replace(destination)
    return bytes_written


def download_file(
    url: str,
    destination: str | Path,
    session: requests.Session | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> DownloadResult:
    """Download ``url`` to ``destination`` unless a complete cached file exists."""

    session = session or requests.Session()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    expected_size = _content_length(session, url)
    if _is_complete(destination, expected_size):
        return DownloadResult(destination, downloaded=False, size_bytes=expected_size)

    bytes_written = _stream_download(session, url, destination, chunk_size)
    if expected_size is not None and bytes_written != expected_size:
        raise IOError(f"downloaded {bytes_written} bytes from {url}, expected {expected_size}")
    return DownloadResult(destination, downloaded=True, size_bytes=bytes_written)


def _url_basename(url: str) -> str:
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    if not name:
        raise ValueError(f"cannot derive a filename from URL: {url}")
    return name


def _scene_asset_path(cache: CacheConfig, scene_id: str, role: str, url: str) -> Path:
    names = {
        "surface_reflectance": f"{scene_id}_ortho_sr.h5",
        "radiance": f"{scene_id}_ortho_radiance.h5",
    }
    return cache.child("scenes", scene_id, names.get(role, _url_basename(url)))


def cache_scene_files(
    assets: SceneAssets,
    cache: CacheConfig | None = None,
    session: requests.Session | None = None,
    include_auxiliary: bool = False,
) -> CachedSceneFiles:
    """Resolve a scene's selected STAC assets into local cache paths."""

    cache = cache or CacheConfig.default()
    session = session or requests.Session()
    scene_id = assets.scene.scene_id

    surface_reflectance = None
    if assets.surface_reflectance is not None:
        surface_reflectance = download_file(
            assets.surface_reflectance,
            _scene_asset_path(cache, scene_id, "surface_reflectance", assets.surface_reflectance),
            session=session,
        ).path

    radiance = None
    if assets.radiance is not None:
        radiance = download_file(
            assets.radiance,
            _scene_asset_path(cache, scene_id, "radiance", assets.radiance),
            session=session,
        ).path

    metadata = None
    if assets.metadata is not None:
        metadata = download_file(
            assets.metadata,
            _scene_asset_path(cache, scene_id, "metadata", assets.metadata),
            session=session,
        ).path

    auxiliary = {}
    if include_auxiliary:
        for name, url in assets.auxiliary.items():
            auxiliary[name] = download_file(
                url,
                _scene_asset_path(cache, scene_id, name, url),
                session=session,
            ).path

    return CachedSceneFiles(
        scene=assets.scene,
        surface_reflectance=surface_reflectance,
        radiance=radiance,
        metadata=metadata,
        auxiliary=auxiliary,
    )

