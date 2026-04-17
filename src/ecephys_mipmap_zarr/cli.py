r"""CLI entry point using tyro with subcommands.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Union

import numpy as np
import tyro
from rich.console import Console
from rich.table import Table

from ecephys_mipmap_zarr.build import ProgressCallback, add_pyramid, build_mipmap
from ecephys_mipmap_zarr.metadata import BuildConfig, QuantizationConfig
from ecephys_mipmap_zarr.read import open_mipmap

console = Console()
PositionalStr = Annotated[str, tyro.conf.Positional]


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


@dataclass
class BuildSettings:
    """Build a mipmap zarr from a flat binary, cbin, or zarr source.

    For flat binary input (``.dat``), provide ``--n-samples``,
    ``--n-channels``, and ``--dtype``.  For ``.cbin`` and zarr input
    these are read from the source automatically.
    """

    input: str
    """Input path: .dat (flat binary), .cbin (mtscomp), .zarr, or SortingAnalyzer .zarr (with --from-analyzer)."""
    output: str
    """Output mipmap zarr store path."""
    from_analyzer: bool = False
    """Treat --input as a SortingAnalyzer zarr, stream the filtered chain via SpikeInterface."""
    raw_path_override: str | None = None
    """When --from-analyzer: URI/path to the raw zarr (required; substituted at the chain leaf)."""
    keep_bad_channels: bool = False
    """When --from-analyzer: peel off the outer ChannelSlice wrapper to keep dead+noise channels."""
    start_seconds: float = 0.0
    """When --from-analyzer: offset into the recording at which to start the mipmap (seconds)."""
    duration_seconds: float | None = None
    """When --from-analyzer: length of the window to include (seconds); default = rest of recording."""
    sampling_frequency: float | None = None
    """Sampling rate in Hz (auto-detected for .cbin)."""
    units: str = "uV"
    """Physical units."""
    n_samples: int | None = None
    """Number of samples (required for .dat)."""
    n_channels: int | None = None
    """Number of channels (required for .dat)."""
    dtype: str = "int16"
    """Data type (required for .dat)."""
    raw_path: str = "raw"
    """Path to raw array (zarr input only)."""
    quantize: bool = False
    """Quantize raw and pyramids (smart: skips raw if already integer)."""
    quantize_pyramids: bool | None = None
    """Quantize pyramids to uint8 (overrides --quantize for pyramids)."""
    compressor: str = "blosc"
    """Compressor preset: blosc, gzip, zstd."""
    metadata_from: str | None = None
    """Zarr store (local or S3) to copy probe/channel metadata from."""
    lsb_shift_raw: int | str = "auto"
    """LSB right-shift for raw data: 'auto' (detect from noise), 0 (lossless), or an integer."""
    lsb_shift_pyramid: int | str = "auto"
    """LSB right-shift for pyramids: 'auto' (detect, +2 vs raw), 0 (lossless), or an integer."""
    noise_floor: float | None = None
    """Explicit noise floor in ADC RMS units (overrides auto-detection)."""
    clip_outliers: bool = True
    """Clip rare outliers (<0.01%%) to fit int8 when LSB-shifting."""
    prefetch_chunks: int = 0
    """Prefetch N source chunks in background threads (0=disabled, 2-3 for S3/WavPack)."""

    def cli_cmd(self) -> None:
        """Run the build command."""
        source: Any
        input_path = Path(self.input)
        built_source_metadata: dict[str, Any] | None = None

        if self.from_analyzer:
            if self.raw_path_override is None:
                console.print("[red]Error:[/red] --raw-path-override is required with --from-analyzer")
                raise SystemExit(1)
            from ecephys_mipmap_zarr.preprocess import (
                SIRecordingSource,
                build_source_metadata,
                load_preprocessed_recording,
            )

            console.print(f"Loading analyzer: {self.input}")
            console.print(f"  raw path: {self.raw_path_override}")
            if self.keep_bad_channels:
                console.print("  keep-bad-channels: unwrapping outer ChannelSlice")
            rec = load_preprocessed_recording(
                analyzer_path=self.input,
                raw_path=self.raw_path_override,
                keep_bad_channels=self.keep_bad_channels,
                analyzer_storage_options={"anon": True} if self.input.startswith("s3://") else None,
                raw_storage_options={"anon": True} if self.raw_path_override.startswith("s3://") else None,
            )
            sampling_frequency = float(rec.get_sampling_frequency())
            full_samples = int(rec.get_num_samples())
            start_sample = int(self.start_seconds * sampling_frequency)
            capped_n_samples: int | None = (
                int(self.duration_seconds * sampling_frequency) if self.duration_seconds is not None else None
            )
            source = SIRecordingSource(rec, start_sample=start_sample, n_samples=capped_n_samples)
            built_source_metadata = build_source_metadata(rec, self.raw_path_override)
            assert built_source_metadata is not None
            if start_sample > 0:
                t0: float = built_source_metadata.get("t_start") or 0.0
                built_source_metadata["t_start"] = t0 + start_sample / sampling_frequency
            dtype = str(rec.get_dtype())
            n_samples, n_channels = source.shape
            clip_note = (
                f" (from full {full_samples:,}, offset={start_sample:,})"
                if n_samples < full_samples or start_sample > 0
                else ""
            )
            console.print(
                f"  filtered chain: {n_samples:,} samples{clip_note}, {n_channels} channels, "
                f"{dtype}, {sampling_frequency:.0f} Hz"
            )

        elif _is_zarr_store(self.input):
            from ecephys_mipmap_zarr._store import _open_store

            if self.sampling_frequency is None:
                console.print("[red]Error:[/red] --sampling-rate is required for zarr input")
                raise SystemExit(1)
            console.print(f"Reading zarr store: {self.input}")
            root = _open_store(self.input, mode="r")
            raw_array = root[self.raw_path]
            source = raw_array
            dtype = str(raw_array.dtype)  # type: ignore[union-attr]
            n_samples, n_channels = raw_array.shape  # type: ignore[union-attr]
            sampling_frequency = self.sampling_frequency
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
            prefetch_chunks=self.prefetch_chunks,
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
                metadata_from=None if built_source_metadata is not None else self.metadata_from,
                source_metadata=built_source_metadata,
                progress=progress_fn,
            )
        console.print("[green]Done.[/green]")


@dataclass
class AddPyramidSettings:
    """Add pyramid levels to an existing zarr, writing to a separate output store."""

    raw_store: str
    """Path or S3 URI to zarr store with raw data."""
    output: str
    """Output pyramid store path."""
    sampling_frequency: float | None = None
    """Sampling rate in Hz (auto-detected from zarr attrs if omitted)."""
    units: str = "uV"
    """Physical units."""
    raw_path: str = "raw"
    """Path to raw array in store."""
    quantize: bool = False
    """Quantize pyramids to uint8."""
    compressor: str = "blosc"
    """Compressor preset: blosc, gzip, zstd."""
    lsb_shift_pyramid: int | str = "auto"
    """LSB right-shift for pyramids: 'auto' (detect, +2 vs raw), 0 (lossless), or an integer."""
    noise_floor: float | None = None
    """Explicit noise floor in ADC RMS units (overrides auto-detection)."""
    clip_outliers: bool = True
    """Clip rare outliers (<0.01%%) to fit int8 when LSB-shifting."""
    prefetch_chunks: int = 0
    """Prefetch N source chunks in background threads (0=disabled, 2-3 for S3/WavPack)."""

    def cli_cmd(self) -> None:
        """Run the add-pyramid command."""
        console.print(f"Reading raw from: {self.raw_store}")
        console.print(f"Building pyramids → {self.output}")
        build_config = BuildConfig(
            compressor=self.compressor,
            lsb_shift_pyramid=self.lsb_shift_pyramid,
            noise_floor=self.noise_floor,
            clip_outliers=self.clip_outliers,
            prefetch_chunks=self.prefetch_chunks,
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


@dataclass
class InfoSettings:
    """Print metadata for a mipmap zarr store."""

    store_path: PositionalStr
    """Path to mipmap zarr store."""

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


@dataclass
class ViewSettings:
    """Launch the interactive trace viewer in a browser tab."""

    store_path: PositionalStr
    """Path to mipmap zarr store."""
    channels: list[int] | None = None
    """Channel indices to display."""
    n_channels: int = 32
    """Number of channels to display."""
    channel_order: str | None = "depth"
    """Channel display order: 'depth' (default, by probe position), 'index' (by channel number)."""
    width: int = 1200
    """Figure width in pixels."""
    height: int = 800
    """Figure height in pixels."""
    cache_dir: str | None = None
    """Cache directory for remote stores (auto-detected if omitted)."""
    cache_max_gb: float = 10
    """Maximum cache size in GB."""
    spikes_from: str | None = None
    """Path to phy/Kilosort output directory for spike overlay."""
    units: list[int] | None = None
    """Unit IDs to display (default: all). Channels auto-selected from spike templates."""

    def cli_cmd(self) -> None:
        """Run the view command."""
        from ecephys_mipmap_zarr.read import open_mipmap
        from ecephys_mipmap_zarr.viewer import TraceViewer, load_phy_spikes

        store = open_mipmap(self.store_path, cache_dir=self.cache_dir, cache_max_gb=self.cache_max_gb)
        console.print(f"Opening viewer: {self.store_path}")
        console.print(f"  {store.n_samples:,} samples, {store.n_channels} channels, {store.sampling_frequency:.0f} Hz")

        spike_data = None
        initial_channels = None
        if self.spikes_from is not None:
            console.print(f"Loading spikes from: {self.spikes_from}")
            spike_data, template_channels = load_phy_spikes(
                self.spikes_from,
                sampling_frequency=store.sampling_frequency,
                unit_ids=self.units,
                t_start=store.t_start,
            )
            n_spikes = sum(len(t) for t in spike_data.spike_times.values())
            console.print(f"  {len(spike_data.spike_times)} units, {n_spikes:,} spikes")

            # Auto-select initial view from templates (doesn't restrict panning)
            if self.channels is None and self.units is not None:
                all_chs: set[int] = set()
                for chs in template_channels.values():
                    all_chs.update(chs)
                initial_channels = sorted(all_chs)
                console.print(f"  Starting view on {len(initial_channels)} template channels")

        viewer = TraceViewer(
            store,
            channels=self.channels,
            n_channels=self.n_channels,
            channel_order=self.channel_order,
            initial_channels=initial_channels,
            width=self.width,
            height=self.height,
            spike_data=spike_data,
        )
        viewer.show()


CLICommand = Union[  # noqa: UP007 — Union required for tyro.cli() type resolution
    Annotated[BuildSettings, tyro.conf.subcommand(name="build")],
    Annotated[AddPyramidSettings, tyro.conf.subcommand(name="add-pyramid")],
    Annotated[InfoSettings, tyro.conf.subcommand(name="info")],
    Annotated[ViewSettings, tyro.conf.subcommand(name="view")],
]


def _suppress_s3fs_shutdown_noise() -> None:
    """Silence s3fs/aiohttp async teardown tracebacks at interpreter exit.

    s3fs registers a weakref finalizer that tries to close the aiohttp
    session, but at shutdown the event loop is already gone — producing
    a harmless but noisy ``RuntimeError`` traceback on stderr.
    """
    import atexit
    import os
    import sys

    def _quiet_exit() -> None:
        # Only suppress if s3fs was actually imported
        if "s3fs" not in sys.modules:
            return
        # Redirect stderr to devnull for the remaining weakref finalizers
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
        except OSError:
            pass

    atexit.register(_quiet_exit)


def main() -> None:
    """CLI entry point."""
    import os

    if os.environ.get("ECEPHYS_MIPMAP_DEBUG"):
        import logging

        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")

    _suppress_s3fs_shutdown_noise()

    try:
        command = tyro.cli(
            CLICommand,
            prog="ecephys-mipmap",
            config=(tyro.conf.ConsolidateSubcommandArgs,),
        )  # type: ignore[call-overload]
        command.cli_cmd()
    except SystemExit:
        raise
    except Exception:
        console.print_exception()
        raise SystemExit(1) from None
