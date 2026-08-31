"""Generic cache-aware file downloads."""

from __future__ import annotations

import hashlib
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import requests

from .cache import CacheConfig

CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class DownloadResult:
    """Result of resolving one URL into the local cache."""

    path: Path
    downloaded: bool
    size_bytes: int | None = None


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


@dataclass(frozen=True)
class LutArchive:
    """A published per-sensor LUT archive. The archive is a zip whose top level
    holds ``shards/`` (and the sensor's stripped ``axes_<sensor>.json``)."""

    url: str | None
    sha256: str | None


# One Zenodo record per sensor. A ``None`` entry is not published yet, and
# ``download_lut`` then needs an explicit ``url=``. A new LUT version is a new
# Zenodo record, so its URL and checksum are updated here together.
LUT_ARCHIVES: dict[str, LutArchive] = {
    "tanager": LutArchive(
        url="https://zenodo.org/records/22210933/files/atmoresponse_lut_tanager.zip?download=1",
        sha256="f457621d75e1a7f0eee33e6f6b0ef37cb56219f748dfbc1af5d117c55fd402e9",
    ),
    "emit": LutArchive(
        url="https://zenodo.org/records/22210726/files/atmoresponse_lut_emit.zip?download=1",
        sha256="7ffb2ab6f807e2f33bdb4a172c30b50f864f02b64fd3fa2f18c7c6497fdc1a3c",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_archive(url: str, destination: Path, session: requests.Session | None) -> None:
    """Fetch ``url`` to ``destination``. An http(s) URL goes through
    ``download_file``; a ``file://`` URL or a bare local path is copied directly,
    which is what the tests and a Zenodo-sandbox dry run use."""
    if url.startswith(("http://", "https://")):
        download_file(url, destination, session=session)
        return
    source = Path(url2pathname(urlparse(url).path)) if url.startswith("file://") else Path(url)
    destination.write_bytes(source.read_bytes())


def _safe_extract(archive: zipfile.ZipFile, target: Path) -> None:
    """Extract every member under ``target``, rejecting any path that escapes it."""
    target = target.resolve()
    for member in archive.namelist():
        resolved = (target / member).resolve()
        if target not in resolved.parents and resolved != target:
            raise IOError(f"unsafe path in LUT archive: {member}")
    archive.extractall(target)


def download_lut(
    sensor: str,
    dest: str | Path | None = None,
    *,
    cache: CacheConfig | None = None,
    url: str | None = None,
    sha256: str | None = None,
    session: requests.Session | None = None,
    force: bool = False,
) -> Path:
    """Fetch and unpack the per-sensor LUT coefficient archive.

    Returns the store directory, the one that contains ``shards/``. Point
    ``LUT_STORE_TANAGER`` or ``LUT_STORE_EMIT`` at it, or pass it to
    ``run_tanager`` or ``run_emit`` as ``lut=``.

    ``sensor`` is ``"tanager"`` or ``"emit"``. With no ``url`` the published
    archive for that sensor is used. Pass ``url`` (an https URL, a ``file://``
    URL, or a bare local path) to fetch a specific archive such as a Zenodo
    sandbox draft or a mirror; ``sha256`` overrides the expected checksum.

    The unpack is idempotent: an existing populated ``shards/`` is reused unless
    ``force=True``.
    """
    sensor = sensor.lower()
    if sensor not in LUT_ARCHIVES:
        raise ValueError(f"unknown sensor {sensor!r}, expected one of {sorted(LUT_ARCHIVES)}")
    archive_spec = LUT_ARCHIVES[sensor]
    if url is None:
        # Fall back to the published record. Its checksum only pairs with its own
        # URL, so an explicit url= keeps only an explicit sha256.
        url = archive_spec.url
        expected_sha = sha256 or archive_spec.sha256
        if url is None:
            raise ValueError(
                f"no published LUT archive URL for {sensor!r} yet. Pass url= with a Zenodo "
                f"record file URL or a local path.")
    else:
        expected_sha = sha256

    if dest is not None:
        store = Path(dest)
    else:
        cache = cache or CacheConfig.default()
        store = cache.child("lut", f"lut_store_{sensor}")

    if (store / "shards").is_dir() and any((store / "shards").glob("shard_*.npz")) and not force:
        return store

    store.mkdir(parents=True, exist_ok=True)
    archive_path = store / f"lut_{sensor}.zip"
    _fetch_archive(url, archive_path, session)

    if expected_sha is not None:
        actual = _sha256(archive_path)
        if actual != expected_sha:
            archive_path.unlink()
            raise IOError(
                f"LUT archive checksum mismatch for {sensor!r}: got {actual}, expected "
                f"{expected_sha}")

    with zipfile.ZipFile(archive_path) as handle:
        _safe_extract(handle, store)
    archive_path.unlink()

    if not any((store / "shards").glob("shard_*.npz")):
        raise IOError(f"unpacked LUT archive for {sensor!r} has no shards under {store}")
    return store
