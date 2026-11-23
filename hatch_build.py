"""Build the web viewer into the package for ``ecephys-mipmap view`` to serve."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

BUNDLE = "src/ecephys_mipmap_zarr/static/viewer"


class ViewerBuildHook(BuildHookInterface):  # type: ignore[type-arg]
    """Run ``npm ci`` and ``npm run build:app`` in ``viewer-web`` before a wheel or sdist is built."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Build the bundle, or require one when the web sources are absent (a wheel built from the sdist)."""
        # Editable installs skip it, so Python-only work needs no node; `serve` says how to build it.
        if version == "editable":
            return
        root = Path(self.root)
        web = root / "viewer-web"
        if (web / "package.json").is_file():
            npm = shutil.which("npm")
            if npm is None:
                raise RuntimeError("building the package needs npm (node >= 22.12) to build the web viewer")
            subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=web, check=True)
            subprocess.run([npm, "run", "build:app"], cwd=web, check=True)
        if not (root / BUNDLE / "index.html").is_file():
            raise RuntimeError(f"{BUNDLE} is missing and there are no web sources to build it from")
