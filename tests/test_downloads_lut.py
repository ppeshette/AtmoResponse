"""Offline tests for ``downloads.download_lut``.

These build a tiny synthetic LUT archive on disk and fetch it through a local
path, so they need no network and no published Zenodo record.
"""
from __future__ import annotations

import hashlib
import zipfile

import pytest

from atmoresponse import downloads
from atmoresponse.downloads import download_lut


def _make_archive(path, members):
    """Write ``members`` (name -> bytes) into a zip whose top level is ``shards/``
    plus an axes file, matching the published archive layout."""
    with zipfile.ZipFile(path, "w") as handle:
        for name, data in members.items():
            handle.writestr(name, data)
    return path


_MEMBERS = {
    "shards/shard_000000000.npz": b"not a real npz, but download_lut only globs the name",
    "shards/shard_000000001.npz": b"second shard",
    "axes_tanager.json": b'{"axes": {}}',
}


class _NoNetworkSession:
    """Any HTTP call is a test failure: the archive is fetched from a local path."""

    def head(self, *a, **k):  # pragma: no cover - only hit on a bug
        raise AssertionError("download_lut made a network call")

    def get(self, *a, **k):  # pragma: no cover - only hit on a bug
        raise AssertionError("download_lut made a network call")


def test_unpacks_local_archive_into_the_store(tmp_path):
    archive = _make_archive(tmp_path / "lut_tanager.zip", _MEMBERS)
    store = tmp_path / "store"

    result = download_lut("tanager", dest=store, url=str(archive), session=_NoNetworkSession())

    assert result == store
    assert sorted(p.name for p in (store / "shards").glob("shard_*.npz")) == [
        "shard_000000000.npz",
        "shard_000000001.npz",
    ]
    assert (store / "axes_tanager.json").is_file()
    assert not (store / "lut_tanager.zip").exists()  # removed after a successful unpack


def test_reuse_is_idempotent_and_does_not_refetch(tmp_path):
    archive = _make_archive(tmp_path / "lut_tanager.zip", _MEMBERS)
    store = tmp_path / "store"
    download_lut("tanager", dest=store, url=str(archive))

    # A second call with a session that refuses I/O must still succeed from disk.
    again = download_lut("tanager", dest=store, url=str(archive), session=_NoNetworkSession())
    assert again == store


def test_force_reextracts(tmp_path):
    archive = _make_archive(tmp_path / "lut_tanager.zip", _MEMBERS)
    store = tmp_path / "store"
    download_lut("tanager", dest=store, url=str(archive))
    (store / "shards" / "shard_000000000.npz").unlink()

    download_lut("tanager", dest=store, url=str(archive), force=True)
    assert (store / "shards" / "shard_000000000.npz").is_file()


def test_checksum_mismatch_raises_and_drops_the_archive(tmp_path):
    archive = _make_archive(tmp_path / "lut_tanager.zip", _MEMBERS)
    store = tmp_path / "store"

    with pytest.raises(IOError, match="checksum mismatch"):
        download_lut("tanager", dest=store, url=str(archive), sha256="0" * 64)
    assert not (store / "lut_tanager.zip").exists()
    assert not (store / "shards").exists()


def test_explicit_url_ignores_the_registry_checksum(tmp_path):
    # A caller-supplied archive is a different file, so the published record's
    # checksum must not be applied to it.
    archive = _make_archive(tmp_path / "lut_tanager.zip", _MEMBERS)
    store = tmp_path / "store"

    result = download_lut("tanager", dest=store, url=str(archive))  # no sha256=
    assert (result / "shards" / "shard_000000000.npz").is_file()


def test_checksum_match_is_accepted(tmp_path):
    archive = _make_archive(tmp_path / "lut_tanager.zip", _MEMBERS)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    store = tmp_path / "store"

    download_lut("tanager", dest=store, url=str(archive), sha256=digest)
    assert (store / "shards" / "shard_000000000.npz").is_file()


def test_unknown_sensor_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown sensor"):
        download_lut("landsat", dest=tmp_path, url=str(tmp_path / "x.zip"))


def test_unpublished_archive_raises(tmp_path, monkeypatch):
    monkeypatch.setitem(downloads.LUT_ARCHIVES, "tanager", downloads.LutArchive(None, None))
    with pytest.raises(ValueError, match="no published LUT archive URL"):
        download_lut("tanager", dest=tmp_path)


def test_shipped_archives_are_published_zenodo_records():
    for sensor, archive in downloads.LUT_ARCHIVES.items():
        assert archive.url and "zenodo.org" in archive.url, sensor
        assert archive.sha256 and len(archive.sha256) == 64, sensor


def test_archive_without_shards_raises(tmp_path):
    archive = _make_archive(tmp_path / "lut_tanager.zip", {"axes_tanager.json": b"{}"})
    with pytest.raises(IOError, match="no shards"):
        download_lut("tanager", dest=tmp_path / "store", url=str(archive))


def test_zip_slip_member_is_rejected(tmp_path):
    archive = tmp_path / "traversal.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("shards/shard_000000000.npz", b"ok")
        handle.writestr("../escape.txt", b"out of the store")
    with pytest.raises(IOError, match="unsafe path"):
        download_lut("tanager", dest=tmp_path / "store", url=str(archive))
