"""Tests for the web viewer's server."""

from collections.abc import Iterator
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs
from urllib.request import Request, urlopen

import pytest

from ecephys_mipmap_zarr import serve
from ecephys_mipmap_zarr.serve import ViewerServer, proxy_base


@pytest.fixture
def store(tmp_path: Path) -> Path:
    path = tmp_path / "rec.zarr"
    (path / "raw" / "c").mkdir(parents=True)
    (path / "zarr.json").write_text('{"zarr_format": 3}')
    (path / "raw" / "c" / "0").write_bytes(bytes(range(256)))
    (tmp_path / "secret.txt").write_text("outside")
    return path


@pytest.fixture(autouse=True)
def static(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in for the built bundle, which editable installs lack."""
    root = tmp_path / "static"
    root.mkdir()
    (root / "index.html").write_text("<mipmap-viewer></mipmap-viewer>")
    (root / "viewer.js").write_text("export {};")
    monkeypatch.setattr(serve, "_static_root", lambda: root)
    return root


@pytest.fixture
def server(store: Path) -> Iterator[ViewerServer]:
    srv = ViewerServer(str(store), markers="https://example.org/m.zarr", params={"channels": "0,1"})
    srv.start()
    yield srv
    srv.close()


def _get(server: ViewerServer, path: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    request = Request(f"http://127.0.0.1:{server.port}/{path}", headers=headers or {})
    try:
        with urlopen(request) as response:
            return response.status, dict(response.headers), response.read()
    except HTTPError as err:
        return err.code, dict(err.headers), b""


def test_page_query_names_local_mounts_and_passes_urls_through(server: ViewerServer) -> None:
    query = parse_qs(server.query)
    assert query == {"store": ["data/0/rec.zarr"], "markers": ["https://example.org/m.zarr"], "channels": ["0,1"]}
    assert server.url("/user/me/proxy/5006/").startswith("/user/me/proxy/5006/?store=data/0/rec.zarr")


def test_serves_the_page_and_its_script(server: ViewerServer) -> None:
    status, headers, body = _get(server, "")
    assert status == 200
    assert headers["Content-Type"] == "text/html"
    assert b"<mipmap-viewer" in body
    status, headers, _ = _get(server, "viewer.js")
    assert (status, headers["Content-Type"]) == (200, "text/javascript")


def test_serves_store_files_whole_or_by_range(server: ViewerServer) -> None:
    status, _, body = _get(server, "data/0/rec.zarr/raw/c/0")
    assert (status, body) == (200, bytes(range(256)))
    status, headers, body = _get(server, "data/0/rec.zarr/raw/c/0", {"Range": "bytes=10-19"})
    assert (status, body, headers["Content-Range"]) == (206, bytes(range(10, 20)), "bytes 10-19/256")
    status, _, body = _get(server, "data/0/rec.zarr/raw/c/0", {"Range": "bytes=-4"})
    assert (status, body) == (206, bytes(range(252, 256)))
    status, _, _ = _get(server, "data/0/rec.zarr/raw/c/0", {"Range": "bytes=300-"})
    assert status == 416


def test_missing_keys_and_paths_outside_a_mount_are_404(server: ViewerServer) -> None:
    assert _get(server, "data/0/rec.zarr/raw/c/9")[0] == 404
    assert _get(server, "data/0/rec.zarr/../secret.txt")[0] == 404
    assert _get(server, "data/0/rec.zarr/%2e%2e/secret.txt")[0] == 404
    assert _get(server, "../../pyproject.toml")[0] == 404


def test_refuses_a_local_path_that_is_not_a_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ViewerServer(str(tmp_path / "missing.zarr"))


def test_proxy_base_prefers_the_given_prefix_then_jupyterhub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JUPYTERHUB_SERVICE_PREFIX", raising=False)
    assert proxy_base(5006, None) is None
    assert proxy_base(5006, "/x/") == "/x/"
    monkeypatch.setenv("JUPYTERHUB_SERVICE_PREFIX", "/user/me/")
    assert proxy_base(5006, None) == "/user/me/proxy/5006/"
