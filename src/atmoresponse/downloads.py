"""Generic cache-aware file downloads."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import requests

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
