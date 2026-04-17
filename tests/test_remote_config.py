"""Unit tests for viewer.RemoteConfig (cloud/proxy serve wiring).

These tests exercise the kwarg-mapping and env-var auto-derivation
without actually booting a Panel server — no Bokeh runtime needed
beyond the import at module load.
"""

from __future__ import annotations

import pytest

from ecephys_mipmap_zarr.viewer import DEFAULT_BOKEH_PORT, RemoteConfig


class TestToServeKwargs:
    def test_defaults_emit_wildcard_origin_and_warn(self, caplog):
        rc = RemoteConfig()
        with caplog.at_level("WARNING"):
            kwargs = rc.to_serve_kwargs()
        assert kwargs["address"] == "0.0.0.0"
        assert kwargs["show"] is False
        assert kwargs["allow_websocket_origin"] == ["*"]
        assert "port" not in kwargs
        assert "prefix" not in kwargs
        assert any("wildcard" in rec.message.lower() for rec in caplog.records)

    def test_explicit_port_and_prefix(self):
        rc = RemoteConfig(port=5010, url_prefix="/proxy/5010", allow_origin=["example.com"])
        kwargs = rc.to_serve_kwargs()
        assert kwargs["port"] == 5010
        assert kwargs["prefix"] == "/proxy/5010"
        assert kwargs["allow_websocket_origin"] == ["example.com"]

    def test_explicit_wildcard_list_passes_through(self):
        rc = RemoteConfig(allow_origin=["*"])
        kwargs = rc.to_serve_kwargs()
        assert kwargs["allow_websocket_origin"] == ["*"]


class TestAutoFromEnv:
    def test_jupyterhub_prefix_derives_url_prefix(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("JUPYTERHUB_SERVICE_PREFIX", "/user/me/")
        rc = RemoteConfig.auto(port=5006)
        assert rc.url_prefix == "/user/me/proxy/5006"
        assert rc.port == 5006

    def test_jupyterhub_prefix_defaults_port(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("JUPYTERHUB_SERVICE_PREFIX", "/user/me")
        rc = RemoteConfig.auto()
        # When no port given, auto-derived prefix uses the Bokeh default
        assert rc.url_prefix == f"/user/me/proxy/{DEFAULT_BOKEH_PORT}"
        assert rc.port is None

    def test_explicit_prefix_wins_over_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("JUPYTERHUB_SERVICE_PREFIX", "/user/me")
        rc = RemoteConfig.auto(url_prefix="/custom")
        assert rc.url_prefix == "/custom"

    def test_no_env_no_prefix(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("JUPYTERHUB_SERVICE_PREFIX", raising=False)
        rc = RemoteConfig.auto(port=5006)
        assert rc.url_prefix is None


class TestDescribeUrl:
    def test_describe_proxy(self):
        rc = RemoteConfig(port=5010, url_prefix="/user/me/proxy/5010")
        desc = rc.describe_url()
        assert "/user/me/proxy/5010" in desc
        assert "5010" in desc

    def test_describe_direct(self):
        rc = RemoteConfig(port=5010)
        desc = rc.describe_url()
        assert "5010" in desc
        assert "proxy" not in desc
