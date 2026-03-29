r"""CLI entry point using pydantic-settings with subcommands.

Requires the ``cli`` extra::

    pip install ecephys-mipmap-zarr[cli]

Usage::

    ecephys-mipmap build --input raw.dat --output out.zarr --sampling-rate 30000 \
        --n-samples 1000000 --n-channels 384 --dtype float32
    ecephys-mipmap add-pyramid --raw-store raw.zarr --output pyr.zarr --sampling-rate 30000
    ecephys-mipmap info store.zarr
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings, CliApp, CliSubCommand
from rich.console import Console
from rich.table import Table

from ecephys_mipmap_zarr.build import ProgressCallback, add_pyramid, build_mipmap
from ecephys_mipmap_zarr.metadata import BuildConfig, QuantizationConfig
from ecephys_mipmap_zarr.read import open_mipmap

console = Console()


@contextmanager
def _progress_bar() -> Generator[ProgressCallback, None, None]:
    """Context manager that yields a rich progress callback."""
    from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )
    task_id = progress.add_task("Processing", total=1)

    def callback(processed: int, total: int) -> None:
        progress.update(task_id, completed=processed, total=total)

    with progress:
        yield callback


def _is_zarr_store(path: str) -> bool:
    """Check whether *path* looks like a Zarr store or remote URI."""
    if "://" in path:
        return True  # Remote URIs are always zarr stores for our purposes
    p = Path(path)
    return p.suffix == ".zarr" or (p / "zarr.json").exists()


def _open_cbin(path: str, sampling_frequency: float | None) -> tuple[Any, str, float]:
    """Open a .cbin file via mtscomp and read params.py if available.

    Parameters
    ----------
    path : str
        Path to the ``.cbin`` file.
    sampling_frequency : float or None
        Explicit sampling rate.  If ``None``, read from ``params.py``
        next to the ``.cbin`` file.

    Returns
    -------
    source : mtscomp.Reader
        Array-like reader with ``.shape`` and slice indexing.
    dtype : str
        Data type string.
    sampling_frequency : float
        Sampling rate in Hz.
    """
    try:
        from mtscomp import Reader
    except ImportError as e:
        msg = "mtscomp is required for .cbin files: pip install ecephys-mipmap-zarr[cbin]"
        raise ImportError(msg) from e

    reader = Reader()
    reader.open(path)

    dtype = str(reader.dtype)

    # Try to read sampling rate from params.py if not provided
    if sampling_frequency is None:
        params_path = Path(path).parent / "params.py"
        if params_path.exists():
            params: dict[str, Any] = {}
            exec(params_path.read_text(), {}, params)
            sampling_frequency = float(params.get("sample_rate", 0))
        if sampling_frequency is None or sampling_frequency <= 0:
            console.print("[red]Error:[/red] --sampling-rate required (could not read from params.py)")
            raise SystemExit(1)

    return reader, dtype, sampling_frequency


class BuildSettings(BaseSettings):
    """Build a mipmap zarr from a flat binary, cbin, or zarr source.

    For flat binary input (``.dat``), provide ``--n-samples``,
    ``--n-channels``, and ``--dtype``.  For ``.cbin`` and zarr input
    these are read from the source automatically.
    """

    input: str = Field(description="Input path: .dat (flat binary), .cbin (mtscomp), or .zarr")
    output: str = Field(description="Output mipmap zarr store path")
    sampling_frequency: float | None = Field(default=None, description="Sampling rate in Hz (auto-detected for .cbin)")
    units: str = Field(default="uV", description="Physical units")
    n_samples: int | None = Field(default=None, description="Number of samples (required for .dat)")
    n_channels: int | None = Field(default=None, description="Number of channels (required for .dat)")
    dtype: str = Field(default="int16", description="Data type (required for .dat)")
    raw_path: str = Field(default="raw", description="Path to raw array (zarr input only)")
    quantize: bool = Field(default=False, description="Quantize raw and pyramids (smart: skips raw if already integer)")
    quantize_pyramids: bool | None = Field(
        default=None, description="Quantize pyramids to uint8 (overrides --quantize for pyramids)"
    )
    compressor: str = Field(default="blosc", description="Compressor preset: blosc, gzip, zstd")
    metadata_from: str | None = Field(
        default=None, description="Zarr store (local or S3) to copy probe/channel metadata from"
    )
    lsb_shift_raw: int | str = Field(
        default="auto",
        description="LSB right-shift for raw data: 'auto' (detect from noise), 0 (lossless), or an integer",
    )
    lsb_shift_pyramid: int | str = Field(
        default="auto",
        description="LSB right-shift for pyramids: 'auto' (detect, +2 vs raw), 0 (lossless), or an integer",
    )
    noise_floor: float | None = Field(
        default=None,
        description="Explicit noise floor in ADC RMS units (overrides auto-detection)",
    )
    clip_outliers: bool = Field(default=True, description="Clip rare outliers (<0.01%) to fit int8 when LSB-shifting")

    def cli_cmd(self) -> None:
        """Run the build command."""
        source: Any
        input_path = Path(self.input)

        if _is_zarr_store(self.input):
            from ecephys_mipmap_zarr._store import _open_store

            console.print(f"Reading zarr store: {self.input}")
            root = _open_store(self.input, mode="r")
            raw_array = root[self.raw_path]
            source = raw_array
            dtype = str(raw_array.dtype)  # type: ignore[union-attr]
            n_samples, n_channels = raw_array.shape  # type: ignore[union-attr]
            sampling_frequency = self.sampling_frequency
            if sampling_frequency is None:
                console.print("[red]Error:[/red] --sampling-rate is required for zarr input")
                raise SystemExit(1)
            console.print(f"  shape: ({n_samples}, {n_channels}), dtype: {dtype}")

        elif input_path.suffix == ".cbin":
            source, dtype, sampling_frequency = _open_cbin(self.input, self.sampling_frequency)
            n_samples, n_channels = source.shape
            console.print(
                f"Reading cbin: {self.input} ({n_samples:,} samples, {n_channels} channels, "
                f"{dtype}, {sampling_frequency:.0f} Hz)"
            )

        else:
            if self.n_samples is None or self.n_channels is None:
                console.print("[red]Error:[/red] --n-samples and --n-channels are required for .dat input")
                raise SystemExit(1)
            if self.sampling_frequency is None:
                console.print("[red]Error:[/red] --sampling-rate is required for .dat input")
                raise SystemExit(1)
            dtype = self.dtype
            sampling_frequency = self.sampling_frequency
            n_samples = self.n_samples
            n_channels = self.n_channels
            console.print(f"Reading binary: {self.input} ({n_samples:,} samples, {n_channels} channels, {dtype})")
            source = np.memmap(self.input, dtype=dtype, mode="r", shape=(n_samples, n_channels))

        console.print(f"Building mipmap → {self.output}")
        build_config = BuildConfig(
            compressor=self.compressor,
            lsb_shift_raw=self.lsb_shift_raw,
            lsb_shift_pyramid=self.lsb_shift_pyramid,
            noise_floor=self.noise_floor,
            clip_outliers=self.clip_outliers,
        )
        quant_config = QuantizationConfig(
            quantize=self.quantize,
            quantize_pyramids=self.quantize_pyramids,
        )
        with _progress_bar() as progress_fn:
            build_mipmap(
                source,
                self.output,
                sampling_frequency=sampling_frequency,
                units=self.units,
                dtype=dtype,
                config=build_config,
                quantization=quant_config,
                metadata_from=self.metadata_from,
                progress=progress_fn,
            )
        console.print("[green]Done.[/green]")


class AddPyramidSettings(BaseSettings):
    """Add pyramid levels to an existing zarr, writing to a separate output store."""

    raw_store: str = Field(description="Path or S3 URI to zarr store with raw data")
    output: str = Field(description="Output pyramid store path")
    sampling_frequency: float | None = Field(
        default=None, description="Sampling rate in Hz (auto-detected from zarr attrs if omitted)"
    )
    units: str = Field(default="uV", description="Physical units")
    raw_path: str = Field(default="raw", description="Path to raw array in store")
    quantize: bool = Field(default=False, description="Quantize pyramids to uint8")
    compressor: str = Field(default="blosc", description="Compressor preset: blosc, gzip, zstd")
    lsb_shift_pyramid: int | str = Field(
        default="auto",
        description="LSB right-shift for pyramids: 'auto' (detect, +2 vs raw), 0 (lossless), or an integer",
    )
    noise_floor: float | None = Field(
        default=None,
        description="Explicit noise floor in ADC RMS units (overrides auto-detection)",
    )
    clip_outliers: bool = Field(default=True, description="Clip rare outliers (<0.01%) to fit int8 when LSB-shifting")

    def cli_cmd(self) -> None:
        """Run the add-pyramid command."""
        console.print(f"Reading raw from: {self.raw_store}")
        console.print(f"Building pyramids → {self.output}")
        build_config = BuildConfig(
            compressor=self.compressor,
            lsb_shift_pyramid=self.lsb_shift_pyramid,
            noise_floor=self.noise_floor,
            clip_outliers=self.clip_outliers,
        )
        quant_config = QuantizationConfig(
            quantize=self.quantize,
        )
        with _progress_bar() as progress_fn:
            add_pyramid(
                self.raw_store,
                self.output,
                sampling_frequency=self.sampling_frequency,
                units=self.units,
                raw_path=self.raw_path,
                config=build_config,
                quantization=quant_config,
                progress=progress_fn,
            )
        console.print("[green]Done.[/green]")


class InfoSettings(BaseSettings):
    """Print metadata for a mipmap zarr store."""

    store_path: str = Field(description="Path to mipmap zarr store")

    def cli_cmd(self) -> None:
        """Run the info command."""
        store = open_mipmap(self.store_path)

        table = Table(title=f"Mipmap Store: {self.store_path}")
        table.add_column("Field", style="bold")
        table.add_column("Value")

        table.add_row("Sampling rate", f"{store.sampling_frequency:,.1f} Hz")
        table.add_row("Samples", f"{store.n_samples:,}")
        table.add_row("Channels", str(store.n_channels))
        table.add_row("Duration", f"{store.n_samples / store.sampling_frequency:.2f} s")
        table.add_row("Dtype", store._attrs.dtype)
        table.add_row("Strides", ", ".join(str(s) for s in store.strides))
        table.add_row("Pyramid levels", str(len(store.strides)))

        if store._attrs.raw_source is not None:
            table.add_row("Raw source", store._attrs.raw_source)

        if store._raw_quantization is not None:
            table.add_row("Raw quantization", f"→ {store._raw_quantization.quantized_dtype}")

        if store._level_quantization is not None:
            table.add_row("Pyramid quantization", f"→ {store._level_quantization.quantized_dtype}")

        if store._raw_lsb_shift > 0:
            table.add_row("Raw LSB shift", f">>{store._raw_lsb_shift}")
        if store._pyr_lsb_shift > 0:
            table.add_row("Pyramid LSB shift", f">>{store._pyr_lsb_shift}")

        ch_ids = store.channel_ids
        if ch_ids is not None:
            table.add_row("Channel IDs", f"{ch_ids[0]}..{ch_ids[-1]} ({len(ch_ids)} total)")

        if store.source_metadata is not None:
            sm = store.source_metadata
            if "probe" in sm:
                probes = sm["probe"].get("probes", [])
                table.add_row("Probe", f"{len(probes)} probe(s)")
            if "annotations" in sm:
                exp = sm["annotations"].get("experiment_name", "")
                if exp:
                    table.add_row("Experiment", exp)

        console.print(table)


class ViewSettings(BaseSettings):
    """Launch the interactive trace viewer in a browser tab."""

    store_path: str = Field(description="Path to mipmap zarr store")
    channels: list[int] | None = Field(default=None, description="Channel indices to display")
    n_channels: int = Field(default=32, description="Number of channels to display")
    width: int = Field(default=1200, description="Figure width in pixels")
    height: int = Field(default=800, description="Figure height in pixels")

    def cli_cmd(self) -> None:
        """Run the view command."""
        from ecephys_mipmap_zarr.read import open_mipmap
        from ecephys_mipmap_zarr.viewer import TraceViewer

        store = open_mipmap(self.store_path)
        console.print(f"Opening viewer: {self.store_path}")
        console.print(f"  {store.n_samples:,} samples, {store.n_channels} channels, {store.sampling_frequency:.0f} Hz")
        viewer = TraceViewer(
            store,
            channels=self.channels,
            n_channels=self.n_channels,
            width=self.width,
            height=self.height,
        )
        viewer.show()


class CLI(BaseSettings):
    """ecephys-mipmap: Build and inspect multi-resolution mipmap zarr stores."""

    model_config = ConfigDict(cli_kebab_case=True)  # type: ignore[assignment,typeddict-unknown-key]

    build: CliSubCommand[BuildSettings]
    add_pyramid: CliSubCommand[AddPyramidSettings]
    info: CliSubCommand[InfoSettings]
    view: CliSubCommand[ViewSettings]


def main() -> None:
    """CLI entry point."""
    try:
        model = CliApp.run(CLI)
        CliApp.run_subcommand(model)
    except SystemExit:
        raise
    except Exception:
        console.print_exception()
        raise SystemExit(1) from None
