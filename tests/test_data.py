import datetime as dt

from atmoresponse import (
    CacheConfig,
    SceneAssets,
    SceneRecord,
    cache_scene_files,
    download_file,
)


class FakeHeadResponse:
    def __init__(self, size):
        self.headers = {}
        if size is not None:
            self.headers["Content-Length"] = str(size)

    def raise_for_status(self):
        pass


class FakeGetResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start:start + chunk_size]


class FakeDownloadSession:
    def __init__(self, bodies):
        self.bodies = bodies
        self.head_calls = []
        self.get_calls = []

    def head(self, url, timeout, allow_redirects):
        assert timeout == 60
        assert allow_redirects is True
        self.head_calls.append(url)
        return FakeHeadResponse(len(self.bodies[url]))

    def get(self, url, stream, timeout):
        assert stream is True
        assert timeout == 600
        self.get_calls.append(url)
        return FakeGetResponse(self.bodies[url])


def _scene_record():
    return SceneRecord(
        scene_id="scene-a",
        acquired=dt.datetime(2025, 1, 1),
        bbox=(-119.0, 34.0, -118.0, 35.0),
    )


def test_download_file_writes_complete_file(tmp_path):
    url = "https://example.test/scene-a-sr.h5"
    session = FakeDownloadSession({url: b"abcdef"})
    destination = tmp_path / "scene-a-sr.h5"

    result = download_file(url, destination, session=session, chunk_size=2)

    assert result.downloaded is True
    assert result.size_bytes == 6
    assert destination.read_bytes() == b"abcdef"
    assert session.get_calls == [url]
    assert list(tmp_path.glob("*.part")) == []


def test_download_file_skips_complete_file(tmp_path):
    url = "https://example.test/scene-a-sr.h5"
    session = FakeDownloadSession({url: b"abcdef"})
    destination = tmp_path / "scene-a-sr.h5"
    destination.write_bytes(b"abcdef")

    result = download_file(url, destination, session=session)

    assert result.downloaded is False
    assert result.size_bytes == 6
    assert session.get_calls == []


def test_cache_scene_files_downloads_sr_and_radiance(tmp_path):
    urls = {
        "https://example.test/source-name-sr.h5": b"sr",
        "https://example.test/source-name-l1.h5": b"l1",
        "https://example.test/quicklook.tif": b"ql",
    }
    session = FakeDownloadSession(urls)
    assets = SceneAssets(
        scene=_scene_record(),
        surface_reflectance="https://example.test/source-name-sr.h5",
        radiance="https://example.test/source-name-l1.h5",
        auxiliary={"quicklook": "https://example.test/quicklook.tif"},
    )

    files = cache_scene_files(assets, cache=CacheConfig(tmp_path), session=session)

    assert files.surface_reflectance == tmp_path / "scenes" / "scene-a" / "scene-a_ortho_sr.h5"
    assert files.radiance == tmp_path / "scenes" / "scene-a" / "scene-a_ortho_radiance.h5"
    assert files.surface_reflectance.read_bytes() == b"sr"
    assert files.radiance.read_bytes() == b"l1"
    assert files.auxiliary == {}
    assert "https://example.test/quicklook.tif" not in session.get_calls


def test_cache_scene_files_can_include_auxiliary_assets(tmp_path):
    urls = {
        "https://example.test/source-name-sr.h5": b"sr",
        "https://example.test/quicklook.tif": b"ql",
    }
    session = FakeDownloadSession(urls)
    assets = SceneAssets(
        scene=_scene_record(),
        surface_reflectance="https://example.test/source-name-sr.h5",
        auxiliary={"quicklook": "https://example.test/quicklook.tif"},
    )

    files = cache_scene_files(
        assets,
        cache=CacheConfig(tmp_path),
        session=session,
        include_auxiliary=True,
    )

    assert files.auxiliary == {"quicklook": tmp_path / "scenes" / "scene-a" / "quicklook.tif"}
    assert files.auxiliary["quicklook"].read_bytes() == b"ql"
