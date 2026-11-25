r"""CLI entry point using tyro with subcommands.

Requires the ``cli`` extra::

    pip install ecephys-mipmap-zarr[cli]

Usage::

    ecephys-mipmap build --input raw.dat --output out.zarr --sampling-frequency 30000 \
        --n-samples 1000000 --n-channels 384 --dtype float32
    ecephys-mipmap add-pyramid --raw-store raw.zarr --output pyr.zarr
    ecephys-mipmap info store.zarr
    ecephys-mipmap add-stats store.zarr
    ecephys-mipmap events-from-sorting curated/ analyzer.zarr spikes.events.zarr
"""

from __future__ import annotations

import dataclasses
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import numpy as np
import tyro
from rich.console import Console
from rich.table import Table

from ecephys_mipmap_zarr.build import ProgressCallback, add_pyramid, build_mipmap
from ecephys_mipmap_zarr.metadata import BuildConfig, EncodingConfig
from ecephys_mipmap_zarr.read import open_mipmap
from ecephys_mipmap_zarr.stats import STATS_ATTR, write_channel_stats

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


def _parse_shift(value: int | str) -> Any:
    """Turn a ``--*-shift`` value into ``"auto"`` or an int; tyro passes digits through as str."""
    if isinstance(value, str) and value != "auto":
        try:
            return int(value)
        except ValueError:
            msg = f"shift must be 'auto' or an integer, got {value!r}"
            raise SystemExit(msg) from None
    return value


def _resolve_prefetch_chunks(value: int, source_path: str) -> int:
    """Resolve ``prefetch_chunks``, treating ``-1`` as auto.

    Auto expands to 3 for remote URIs (``"://"`` in *source_path*) and 0
    for local paths.  Overlapping a couple of WavPack/S3 decodes is worth
    the extra in-flight memory when the bottleneck is network+decode; it
    only adds scheduling overhead on a local nvme.
    """
    if value >= 0:
        return value
    return 3 if "://" in source_path else 0


@contextmanager
def _scipy_fft_workers(n: int) -> Generator[None, None, None]:
    """Scope ``scipy.fft.set_workers(n)`` around a build.

    ``n == 0`` leaves the scipy default alone (no-op).  Any other value
    is passed through to ``scipy.fft.set_workers`` — ``-1`` means "all
    available cores", which is the right choice for bulk `--from-analyzer`
    builds dominated by PhaseShift FFTs.

    The setting rides a :class:`contextvars.ContextVar`; `_chunk_iter`
    copies the context when submitting to its prefetch pool so the value
    actually reaches the worker threads.
    """
    if n == 0:
        yield
        return
    from scipy.fft import set_workers

    with set_workers(n):
        yield


def _resolve_window(
    start_seconds: float,
    duration_seconds: float | None,
    sampling_frequency: float,
    full_samples: int,
) -> tuple[int, int | None]:
    """Translate --start-seconds / --duration-seconds into sample indices.

    Returns ``(start_sample, n_samples)`` where *n_samples* is ``None``
    when the user didn't specify a duration (meaning "to the end").
    """
    start_sample = max(0, int(start_seconds * sampling_frequency))
    if start_sample >= full_samples:
        msg = (
            f"--start-seconds {start_seconds:.3f} (sample {start_sample:,}) "
            f"is past the end of the recording ({full_samples:,} samples)"
        )
        raise ValueError(msg)
    if duration_seconds is None:
        return start_sample, None
    n_samples = int(duration_seconds * sampling_frequency)
    n_samples = min(n_samples, full_samples - start_sample)
    return start_sample, n_samples


def _maybe_window_source(
    source: Any,
    start_seconds: float,
    duration_seconds: float | None,
    sampling_frequency: float,
    full_samples: int,
) -> tuple[Any, int]:
    """Wrap *source* in :class:`_WindowedSource` if a time window is requested.

    Returns ``(windowed_source, n_samples)``.  When no window is given
    (offset zero and duration unset), *source* is returned unchanged
    with ``n_samples = full_samples``.
    """
    start_sample, capped = _resolve_window(start_seconds, duration_seconds, sampling_frequency, full_samples)
    if start_sample == 0 and capped is None:
        return source, full_samples
    end_sample = full_samples if capped is None else start_sample + capped
    return _WindowedSource(source, start_sample, end_sample), end_sample - start_sample


class _WindowedSource:
    """Array-like wrapper restricting a (T, C) source to a sample range.

    Implements the same ``shape`` / ``dtype`` / 2D slice interface the
    build engine uses (see ``_engine._write_raw_and_compute_finest``).
    Access is lazy — the underlying source is indexed on each read
    with ``start`` added to the time-axis slice — so this works with
    zarr arrays, numpy memmaps, and mtscomp readers alike without
    pulling the whole window into memory up front.
    """

    def __init__(self, source: Any, start_sample: int, end_sample: int) -> None:
        full_n, n_channels = source.shape
        if start_sample < 0 or start_sample >= full_n:
            msg = f"start_sample={start_sample} out of range [0, {full_n})"
            raise ValueError(msg)
        if end_sample < start_sample or end_sample > full_n:
            msg = f"end_sample={end_sample} out of range [{start_sample}, {full_n}]"
            raise ValueError(msg)
        self._source = source
        self._start = int(start_sample)
        self._n_samples = int(end_sample - start_sample)
        self._n_channels = int(n_channels)
        self._dtype: np.dtype[Any] = np.dtype(source.dtype)

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_samples, n_channels)`` of the restricted window."""
        return (self._n_samples, self._n_channels)

    @property
    def dtype(self) -> np.dtype[Any]:
        """Underlying source dtype."""
        return self._dtype

    def __getitem__(self, key: Any) -> np.ndarray:
        """Support ``src[t0:t1, :]`` — indices are window-relative."""
        if isinstance(key, tuple) and len(key) == 2:
            t_key, ch_key = key
            if isinstance(t_key, slice):
                start = (t_key.start or 0) + self._start
                stop = (t_key.stop if t_key.stop is not None else self._n_samples) + self._start
                return np.asarray(self._source[start:stop, ch_key])
        if isinstance(key, slice):
            start = (key.start or 0) + self._start
            stop = (key.stop if key.stop is not None else self._n_samples) + self._start
            return np.asarray(self._source[start:stop])
        msg = f"_WindowedSource only supports 1D/2D slicing, got {type(key).__name__}"
        raise TypeError(msg)


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
            console.print("[red]Error:[/red] --sampling-frequency required (could not read from params.py)")
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
    """Offset into the source recording at which to start the mipmap, in seconds."""
    duration_seconds: float | None = None
    """Length of the source window to include, in seconds; default = rest of recording."""
    sampling_frequency: float | None = None
    """Sampling rate in Hz (default: read from a zarr's root attrs or a .cbin's params.py)."""
    units: str | None = None
    """Units the values decode to (default: uV with an ADC gain, counts for integer sources without one)."""
    n_samples: int | None = None
    """Number of samples (required for .dat)."""
    n_channels: int | None = None
    """Number of channels (required for .dat)."""
    dtype: str = "int16"
    """Data type (required for .dat)."""
    raw_path: str = "raw"
    """Path to raw array (zarr input only)."""
    raw_encoding: Literal["shift", "range", "none"] | None = None
    """Raw encoding: shift (drop noise bits, int8 when it fits), range (per-channel uint16) or none.
    Default: shift for integer sources, range for float."""
    pyramid_encoding: Literal["shift", "range", "none"] | None = None
    """Pyramid encoding: shift, range (per-channel uint8) or none. Default as for --raw-encoding."""
    compressor: str = "blosc"
    """Compressor preset: blosc, gzip, zstd."""
    metadata_from: str | None = None
    """Zarr store (local or S3) to copy probe/channel metadata from."""
    raw_shift: int | str = "auto"
    """Bits the raw shift drops: 'auto' (chosen from the noise floor) or an integer."""
    pyramid_shift: int | str = "auto"
    """Bits the pyramid shift drops: 'auto' (2 more than the noise allows for raw) or an integer."""
    noise_floor: float | None = None
    """Noise floor in source units RMS for 'auto' shifts (default: 10-14 kHz, before any --from-analyzer chain)."""
    clip_outliers: bool = True
    """After shifting, clip values outside int8 when under 0.01%% of them, rather than keep int16."""
    prefetch_chunks: int = -1
    """Prefetch N source chunks in background threads (-1=auto: 3 remote / 0 local; 0=disabled)."""
    wavpack_decode_threads: int = 4
    """Threads for WavPack decode on --from-analyzer sources (1=single-threaded)."""
    fft_workers: int = -1
    """scipy.fft workers for PhaseShift FFTs (-1=all cores, 1=single, 0=leave scipy default)."""

    def cli_cmd(self) -> None:
        """Run the build command."""
        source: Any
        input_path = Path(self.input)
        built_source_metadata: dict[str, Any] | None = None
        chain: Any = None
        chain_start = 0

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
                wavpack_decode_threads=self.wavpack_decode_threads,
            )
            sampling_frequency = float(rec.get_sampling_frequency())
            full_samples = int(rec.get_num_samples())
            start_sample, capped_n_samples = _resolve_window(
                self.start_seconds,
                self.duration_seconds,
                sampling_frequency,
                full_samples,
            )
            source = SIRecordingSource(rec, start_sample=start_sample, n_samples=capped_n_samples)
            chain, chain_start = rec, start_sample
            built_source_metadata = build_source_metadata(rec)
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

            console.print(f"Reading zarr store: {self.input}")
            root = _open_store(self.input, mode="r")
            sampling_frequency = self.sampling_frequency or float(root.attrs.get("sampling_frequency") or 0)  # type: ignore[arg-type]
            if sampling_frequency <= 0:
                console.print(
                    "[red]Error:[/red] --sampling-frequency is required: the store has no sampling_frequency attr"
                )
                raise SystemExit(1)
            raw_array = root[self.raw_path]
            source = raw_array
            dtype = str(raw_array.dtype)  # type: ignore[union-attr]
            full_samples, n_channels = raw_array.shape  # type: ignore[union-attr]
            console.print(f"  shape: ({full_samples}, {n_channels}), dtype: {dtype}")
            source, n_samples = _maybe_window_source(
                source,
                self.start_seconds,
                self.duration_seconds,
                sampling_frequency,
                full_samples,
            )

        elif input_path.suffix == ".cbin":
            source, dtype, sampling_frequency = _open_cbin(self.input, self.sampling_frequency)
            full_samples, n_channels = source.shape
            console.print(
                f"Reading cbin: {self.input} ({full_samples:,} samples, {n_channels} channels, "
                f"{dtype}, {sampling_frequency:.0f} Hz)"
            )
            source, n_samples = _maybe_window_source(
                source,
                self.start_seconds,
                self.duration_seconds,
                sampling_frequency,
                full_samples,
            )

        else:
            if self.n_samples is None or self.n_channels is None:
                console.print("[red]Error:[/red] --n-samples and --n-channels are required for .dat input")
                raise SystemExit(1)
            if self.sampling_frequency is None:
                console.print("[red]Error:[/red] --sampling-frequency is required for .dat input")
                raise SystemExit(1)
            dtype = self.dtype
            sampling_frequency = self.sampling_frequency
            full_samples = self.n_samples
            n_channels = self.n_channels
            console.print(f"Reading binary: {self.input} ({full_samples:,} samples, {n_channels} channels, {dtype})")
            source = np.memmap(self.input, dtype=dtype, mode="r", shape=(full_samples, n_channels))
            source, n_samples = _maybe_window_source(
                source,
                self.start_seconds,
                self.duration_seconds,
                sampling_frequency,
                full_samples,
            )

        if n_samples < full_samples:
            console.print(
                f"  windowed: {n_samples:,} samples (offset {self.start_seconds:.3f}s, "
                f"duration {n_samples / sampling_frequency:.3f}s)"
            )

        prefetch_chunks = _resolve_prefetch_chunks(self.prefetch_chunks, self.input)
        console.print(f"Building mipmap → {self.output}")
        build_config = BuildConfig(
            compressor=self.compressor,
            prefetch_chunks=prefetch_chunks,
        )
        encoding = EncodingConfig(
            raw=self.raw_encoding,
            pyramids=self.pyramid_encoding,
            raw_shift=_parse_shift(self.raw_shift),
            pyramid_shift=_parse_shift(self.pyramid_shift),
            noise_floor=self.noise_floor,
            clip_outliers=self.clip_outliers,
        )
        if chain is not None and encoding.measures_noise(dtype):
            from ecephys_mipmap_zarr.preprocess import chain_noise_floor

            encoding = dataclasses.replace(encoding, noise_floor=chain_noise_floor(chain, chain_start))
            console.print(f"  noise floor {encoding.noise_floor:.3f} RMS, measured before the chain")
        with _progress_bar() as progress_fn, _scipy_fft_workers(self.fft_workers):
            build_mipmap(
                source,
                self.output,
                sampling_frequency=sampling_frequency,
                units=self.units,
                dtype=dtype,
                config=build_config,
                encoding=encoding,
                metadata_from=None if built_source_metadata is not None else self.metadata_from,
                source_metadata=built_source_metadata,
                t_start_offset=float(self.start_seconds),
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
    units: str | None = None
    """Units the values decode to (default: counts for integer raw; required for float)."""
    raw_path: str = "raw"
    """Path to raw array in store."""
    pyramid_encoding: Literal["shift", "range", "none"] | None = None
    """Pyramid encoding: shift, range (per-channel uint8) or none. Default as for --raw-encoding."""
    compressor: str = "blosc"
    """Compressor preset: blosc, gzip, zstd."""
    pyramid_shift: int | str = "auto"
    """Bits the pyramid shift drops: 'auto' (2 more than the noise allows for raw) or an integer."""
    noise_floor: float | None = None
    """Noise floor in source units RMS for 'auto' shifts (default: measured at 10-14 kHz)."""
    clip_outliers: bool = True
    """After shifting, clip values outside int8 when under 0.01%% of them, rather than keep int16."""
    prefetch_chunks: int = -1
    """Prefetch N source chunks in background threads (-1=auto: 3 remote / 0 local; 0=disabled)."""

    def cli_cmd(self) -> None:
        """Run the add-pyramid command."""
        prefetch_chunks = _resolve_prefetch_chunks(self.prefetch_chunks, self.raw_store)
        console.print(f"Reading raw from: {self.raw_store}")
        console.print(f"Building pyramids → {self.output}")
        build_config = BuildConfig(
            compressor=self.compressor,
            prefetch_chunks=prefetch_chunks,
        )
        encoding = EncodingConfig(
            pyramids=self.pyramid_encoding,
            pyramid_shift=_parse_shift(self.pyramid_shift),
            noise_floor=self.noise_floor,
            clip_outliers=self.clip_outliers,
        )
        with _progress_bar() as progress_fn:
            add_pyramid(
                self.raw_store,
                self.output,
                sampling_frequency=self.sampling_frequency,
                units=self.units,
                raw_path=self.raw_path,
                config=build_config,
                encoding=encoding,
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
class AddStatsSettings:
    """Store per-channel display statistics in a mipmap store built before builds wrote them."""

    store_path: PositionalStr
    """Path to a local mipmap zarr store."""

    def cli_cmd(self) -> None:
        """Run the add-stats command."""
        if not Path(self.store_path).is_dir():
            console.print(f"[red]Error:[/red] {self.store_path} is not a local directory")
            raise SystemExit(1)
        stats = write_channel_stats(self.store_path)
        fs = open_mipmap(self.store_path).sampling_frequency
        start, stop = (s / fs for s in stats.samples)
        channels = len(stats.centre)
        console.print(f"Wrote {STATS_ATTR} for {channels} channels, noise measured over {start:.2f}-{stop:.2f} s")


@dataclass
class ViewSettings:
    """Open a store in the web viewer, served from this machine; --bokeh opens the older Panel viewer."""

    store_path: PositionalStr
    """Mipmap store: a local directory or a URL."""
    events: str | None = None
    """Event layer whose groups colour the traces and fill the raster: a local directory or a URL."""
    markers: str | None = None
    """Event layer whose events tick the time axis, to step between: a local directory or a URL."""
    filter: str | None = None
    """Draw only the event groups whose properties match, e.g. 'KSLabel:good,decoder_label:sua|mua' (all of the
    comma-joined properties, any of the bar-joined values)."""
    channels: list[int] | None = None
    """Channel indices to display."""
    channel_order: str | None = None
    """Channel display order: 'depth' (by probe position; the default when the store has locations) or 'index'."""
    remote: bool = False
    """Serve for access through a reverse proxy (cloud IDE / JupyterHub); binds on 0.0.0.0, no auto browser."""
    port: int | None = None
    """Fixed bind port (default: a free one locally, 5006 with --remote)."""
    url_prefix: str | None = None
    """External URL prefix the proxy mounts the viewer under, for the printed link (only with --remote)."""
    open_browser: bool = True
    """Open the page in a browser (not with --remote)."""
    bokeh: bool = False
    """Use the Panel + Bokeh viewer; the options below apply to it only."""
    n_channels: int = 32
    """Number of channels to display (--bokeh)."""
    width: int = 1200
    """Figure width in pixels (--bokeh)."""
    height: int = 800
    """Figure height in pixels (--bokeh)."""
    cache_dir: str | None = None
    """Cache directory for remote stores, auto-detected if omitted (--bokeh)."""
    cache_max_gb: float = 10
    """Maximum cache size in GB (--bokeh)."""
    spikes_from: str | None = None
    """Path to phy/Kilosort output directory for spike overlay (--bokeh)."""
    units: list[int] | None = None
    """Unit IDs to display, default all; channels auto-selected from spike templates (--bokeh)."""
    allow_origin: list[str] | None = None
    """Allowed websocket Origin hostnames, default wildcard with warning (--bokeh with --remote)."""

    def cli_cmd(self) -> None:
        """Run the view command."""
        if self.bokeh:
            self._bokeh()
        else:
            self._web()

    def _web(self) -> None:
        from ecephys_mipmap_zarr.serve import DEFAULT_REMOTE_PORT, ViewerServer, proxy_base

        ignored = [
            flag
            for flag, value in (
                ("--spikes-from", self.spikes_from),
                ("--units", self.units),
                ("--cache-dir", self.cache_dir),
                ("--allow-origin", self.allow_origin),
            )
            if value is not None
        ]
        if ignored:
            console.print(f"[yellow]Warning:[/yellow] {', '.join(ignored)} apply to --bokeh only; ignoring.")
        if not self.remote and self.url_prefix is not None:
            console.print("[yellow]Warning:[/yellow] --url-prefix requires --remote; ignoring.")
        params: dict[str, str] = {}
        if self.channels is not None:
            params["channels"] = ",".join(str(c) for c in self.channels)
        if self.channel_order is not None:
            params["order"] = self.channel_order
        if self.filter is not None:
            params["filter"] = self.filter
        port = self.port if self.port is not None else DEFAULT_REMOTE_PORT if self.remote else 0
        try:
            server = ViewerServer(
                self.store_path,
                events=self.events,
                markers=self.markers,
                params=params,
                host="0.0.0.0" if self.remote else "127.0.0.1",
                port=port,
            )
        except FileNotFoundError as err:
            # A missing store or an unbuilt bundle: the message says which.
            console.print(f"[red]Error:[/red] {err}", soft_wrap=True)
            raise SystemExit(1) from None
        base = proxy_base(server.port, self.url_prefix) if self.remote else None
        console.print(f"Serving the viewer at {server.url(base)}", soft_wrap=True)
        console.print("Press Ctrl+C to stop.")
        server.serve_forever(open_browser=self.open_browser and not self.remote)

    def _bokeh(self) -> None:
        from ecephys_mipmap_zarr.read import open_mipmap
        from ecephys_mipmap_zarr.viewer import RemoteConfig, TraceViewer, load_phy_spikes

        if self.events is not None or self.markers is not None:
            console.print("[yellow]Warning:[/yellow] --events/--markers apply to the web viewer only; ignoring.")
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
            channel_order=self.channel_order or "depth",
            initial_channels=initial_channels,
            width=self.width,
            height=self.height,
            spike_data=spike_data,
        )

        if self.remote:
            remote_config = RemoteConfig.auto(
                port=self.port,
                url_prefix=self.url_prefix,
                allow_origin=self.allow_origin,
            )
            console.print(f"Serving remotely: {remote_config.describe_url()}")
            viewer.show(remote=remote_config)
        else:
            if self.port is not None or self.url_prefix is not None or self.allow_origin is not None:
                console.print(
                    "[yellow]Warning:[/yellow] --port/--url-prefix/--allow-origin require --remote; ignoring."
                )
            viewer.show()


@dataclass
class EventsFromSortingSettings:
    """Write an event layer of sorted spikes, with curation applied, for the web viewer to colour."""

    sorting: PositionalStr
    """Numpy-folder sorting with ``spikes.npy`` and ``curation.json`` (an AIND sorted asset's ``curated/``)."""

    analyzer: PositionalStr
    """SpikeInterface zarr analyzer of the same sorting (path or ``s3://`` URI), for templates and amplitudes."""

    output: PositionalStr
    """Event layer to write."""

    anon: bool = False
    """Read a remote analyzer without credentials, as public data needs."""

    overwrite: bool = False
    """Replace an existing output."""

    def cli_cmd(self) -> None:
        """Run the events-from-sorting command."""
        from ecephys_mipmap_zarr.events import events_from_sorting, read_analyzer, write_events

        analyzer = read_analyzer(self.analyzer, {"anon": True} if self.anon else None)
        args = events_from_sorting(self.sorting, **analyzer)
        write_events(self.output, **args, overwrite=self.overwrite)
        console.print(f"{args['sample_index'].size:,} spikes in {len(args['groups'].ids)} units → {self.output}")


@dataclass
class EventsFromTableSettings:
    """Write an event layer from a tab-separated table of detections, one row each."""

    table: PositionalStr
    """TSV with ``time_s`` and optionally ``peak_channel``, ``peak_uv``, ``width_ms``, ``half_top_um``,
    ``half_bottom_um``."""

    store: PositionalStr
    """Mipmap store the times refer to, for its sampling rate, start and channel ids."""

    output: PositionalStr
    """Event layer to write."""

    group_by: str | None = None
    """Column whose values become groups."""

    overwrite: bool = False
    """Replace an existing output."""

    def cli_cmd(self) -> None:
        """Run the events-from-table command."""
        import csv

        from ecephys_mipmap_zarr.events import events_from_detections, write_events

        with open(self.table, newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        numeric = {"time_s", "peak_channel", "peak_uv", "width_ms", "half_top_um", "half_bottom_um"}
        columns: dict[str, Any] = {}
        for name in rows[0].keys() if rows else []:
            values = [r[name] for r in rows]
            columns[name] = np.asarray(values, float) if name in numeric else np.asarray(values)
        store = open_mipmap(self.store)
        ids = store.channel_ids
        args = events_from_detections(
            columns,
            sampling_frequency=store.sampling_frequency,
            channel_ids=[str(c) for c in ids] if ids is not None else [str(c) for c in range(store.n_channels)],
            group_by=self.group_by,
            t_start=store.t_start,
        )
        write_events(self.output, **args, overwrite=self.overwrite)
        console.print(f"{len(rows):,} events in {len(args['groups'].ids)} groups → {self.output}")


CLICommand = Union[  # noqa: UP007 — Union required for tyro.cli() type resolution
    Annotated[BuildSettings, tyro.conf.subcommand(name="build")],
    Annotated[AddPyramidSettings, tyro.conf.subcommand(name="add-pyramid")],
    Annotated[InfoSettings, tyro.conf.subcommand(name="info")],
    Annotated[AddStatsSettings, tyro.conf.subcommand(name="add-stats")],
    Annotated[ViewSettings, tyro.conf.subcommand(name="view")],
    Annotated[EventsFromSortingSettings, tyro.conf.subcommand(name="events-from-sorting")],
    Annotated[EventsFromTableSettings, tyro.conf.subcommand(name="events-from-table")],
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
        )
        command.cli_cmd()
    except SystemExit:
        raise
    except Exception:
        console.print_exception()
        raise SystemExit(1) from None
