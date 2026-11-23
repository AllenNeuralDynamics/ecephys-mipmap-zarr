"""Serve the web viewer and local stores to a browser, with the ``Range`` requests zarr's sharded reads make.

The page is the standalone viewer built from ``viewer-web`` into ``static/viewer``. Each local store or event layer
is mounted read-only at ``data/<n>/<name>``; remote URLs go to the page unchanged, for the browser to fetch. Every
URL the page uses is relative, so it also works under a reverse proxy's path prefix.
"""

from __future__ import annotations

import os
import re
import threading
import webbrowser
from collections.abc import Mapping
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, urlencode, urlsplit

__all__ = ["DEFAULT_REMOTE_PORT", "ViewerServer", "proxy_base", "viewer_query"]

#: Port served on behind a proxy when none is given; the Bokeh viewer's, so proxy setups carry over.
DEFAULT_REMOTE_PORT = 5006

#: Bytes copied per read when streaming a file.
_COPY_BYTES = 1 << 20
_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
#: The page's own types, fixed: system tables (the Windows registry among them) can map .js to text/plain, which
#: browsers refuse to run as a module.
_TYPES = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".wasm": "application/wasm",
}


def _static_root() -> Path:
    root = Path(str(files("ecephys_mipmap_zarr") / "static" / "viewer"))
    if not (root / "index.html").is_file():
        raise FileNotFoundError(f"the web viewer is not built: run `npm run build:app` in viewer-web ({root})")
    return root


def _is_url(location: str) -> bool:
    return "://" in location


@dataclass
class _Mounts:
    """URL prefixes (no leading slash) and the local directories they serve."""

    roots: dict[str, Path] = field(default_factory=dict)

    def add(self, location: str) -> str:
        """Mount a local directory and return its URL relative to the page, or a remote URL unchanged."""
        if _is_url(location):
            return location
        path = Path(location).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"{location} is not a directory")
        prefix = f"data/{len(self.roots)}/{path.name}"
        self.roots[prefix] = path
        return prefix

    def resolve(self, url_path: str) -> Path | None:
        """Return the local file `url_path` names, or None when it is outside every mount."""
        for prefix, root in self.roots.items():
            if url_path == prefix or url_path.startswith(prefix + "/"):
                target = (root / url_path[len(prefix) + 1 :]).resolve()
                return target if target.is_relative_to(root) else None
        return None


def viewer_query(params: Mapping[str, str | None]) -> str:
    """Encode the page's parameters (``store``, ``events``, ``markers``, ...) as its query string."""
    return urlencode({k: v for k, v in params.items() if v is not None}, safe="/:,")


def proxy_base(port: int, url_prefix: str | None) -> str | None:
    """Return the URL path a proxy serves `port` under: `url_prefix`, or JupyterHub's ``<prefix>/proxy/<port>``."""
    if url_prefix is not None:
        return url_prefix
    hub = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
    return f"{hub.rstrip('/')}/proxy/{port}/" if hub else None


def _byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Inclusive ``(start, end)`` of a single-range ``Range`` header, or None to send the whole file."""
    match = _RANGE.match(header or "")
    if match is None or match.group(1) == match.group(2) == "":
        return None
    first, last = match.groups()
    if first == "":
        return max(0, size - int(last)), size - 1
    return int(first), size - 1 if last == "" else min(size - 1, int(last))


class _Handler(BaseHTTPRequestHandler):
    # Keep-alive: a view fetches tiles in bursts, and a new connection per tile stalls them.
    protocol_version = "HTTP/1.1"
    # Headers and body go out as separate writes; with Nagle on, each response then waits on a delayed ACK.
    disable_nagle_algorithm = True

    def do_GET(self) -> None:
        """Send a static file or a mounted store's file, honouring ``Range``."""
        self._send(body=True)

    def do_HEAD(self) -> None:
        """Send headers only."""
        self._send(body=False)

    def log_message(self, format: str, *args: Any) -> None:
        """Stay quiet: a view fetches thousands of chunks."""

    def _send(self, *, body: bool) -> None:
        server = cast("_Server", self.server)
        url_path = unquote(urlsplit(self.path).path).lstrip("/")
        path = server.mounts.resolve(url_path)
        if path is None:
            static = server.static
            path = (static / (url_path or "index.html")).resolve()
            if not path.is_relative_to(static):
                path = None
        if path is None or not path.is_file():
            # zarr reads a missing key's 404 as absent.
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        stat = path.stat()
        size = stat.st_size
        # Stores can be rewritten in place, so the browser revalidates each time; an unchanged file costs a 304.
        etag = f'"{stat.st_mtime_ns:x}-{size:x}"'
        if etag in (self.headers.get("If-None-Match") or "").split(", "):
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return
        span = _byte_range(self.headers.get("Range"), size)
        if span is not None and span[0] >= size:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        start, end = span if span is not None else (0, size - 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if span is not None else HTTPStatus.OK)
        self.send_header("Content-Type", _TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        if span is not None:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if not body:
            return
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    chunk = f.read(min(_COPY_BYTES, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                # The viewer cancels requests for tiles that left the view.
                pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # The default backlog of 5 drops connections from a burst of tile requests, which then wait out SYN retries.
    request_queue_size = 128

    def __init__(self, address: tuple[str, int], static: Path, mounts: _Mounts) -> None:
        super().__init__(address, _Handler)
        self.static = static
        self.mounts = mounts


class ViewerServer:
    """Serve the web viewer opened on `store`, with optional event and marker layers.

    Parameters
    ----------
    store : str
        A local mipmap store directory, or its URL.
    events, markers : str, optional
        Event layers, local or URLs: one colouring the traces, one ticking the time axis.
    params : mapping, optional
        Further page parameters, such as ``channels`` (``"0-63"``), ``order``, ``time`` (``"10,11"``) or ``theme``.
    host : str, optional
        Interface to bind.
    port : int, optional
        Port to bind; 0 picks a free one.
    """

    def __init__(
        self,
        store: str,
        *,
        events: str | None = None,
        markers: str | None = None,
        params: Mapping[str, str] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        mounts = _Mounts()
        layers = {"store": store, "events": events, "markers": markers}
        query = {k: None if v is None else mounts.add(v) for k, v in layers.items()}
        self.query = viewer_query({**query, **(params or {})})
        self._server = _Server((host, port), _static_root(), mounts)
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """The bound port."""
        return int(self._server.server_address[1])

    def url(self, base: str | None = None) -> str:
        """Return the page's URL under `base`, by default this machine's address and port."""
        root = base if base is not None else f"http://localhost:{self.port}/"
        return f"{root.rstrip('/')}/?{self.query}"

    def start(self) -> None:
        """Serve in a background thread."""
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def serve_forever(self, *, open_browser: bool = False) -> None:
        """Serve until interrupted, opening the page in a browser first if asked."""
        if open_browser:
            threading.Timer(0.2, webbrowser.open, args=(self.url(),)).start()
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self) -> None:
        """Stop serving and release the port."""
        if self._thread is not None:
            self._server.shutdown()
            self._thread = None
        self._server.server_close()
