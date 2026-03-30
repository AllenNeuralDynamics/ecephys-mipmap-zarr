"""Interactive multi-channel trace viewer backed by mipmap Zarr stores.

Requires the ``viewer`` extra::

    pip install ecephys-mipmap-zarr[viewer]

Usage::

    from ecephys_mipmap_zarr.viewer import view
    view("mipmap.zarr")  # opens browser tab
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import panel as pn
from bokeh.models import ColumnDataSource, FixedTicker, Range1d
from bokeh.plotting import figure as bokeh_figure

from ecephys_mipmap_zarr.read import MipmapStore, WindowResult, open_mipmap

log = logging.getLogger(__name__)

__all__ = [
    "SpikeData",
    "SpikeOverlay",
    "TraceViewer",
    "view",
]


@dataclass
class SpikeData:
    """Spike times and unit-to-channel mapping for overlay display.

    Parameters
    ----------
    spike_times : dict[str, np.ndarray]
        Mapping of unit ID to sorted spike times in seconds.
    unit_channels : dict[str, int]
        Mapping of unit ID to channel index.
    colors : dict[str, str] or None
        Per-unit colors.  Auto-assigned from palette if ``None``.
    spike_window_ms : float
        Half-window in ms for trace coloring in raw mode (default 1.0).
    box_height_fraction : float
        Fraction of channel spacing used for box height (default 0.3).
    sparse_threshold : float
        Spikes per pixel below which individual ticks are shown (default 2.0).
    """

    spike_times: dict[str, npt.NDArray[Any]]
    unit_channels: dict[str, int]
    colors: dict[str, str] | None = None
    spike_window_ms: float = 1.0
    box_height_fraction: float = 0.3
    sparse_threshold: float = 2.0


class SpikeOverlay:
    """Render spike times on a Bokeh figure with adaptive display modes.

    Zoomed out: colored boxes per unit (density-based opacity).
    Zoomed in: colored trace segments around each spike time.

    Parameters
    ----------
    figure : bokeh Figure
        The figure to add glyphs to.
    spike_data : SpikeData
        Spike times and configuration.
    channel_offsets : np.ndarray
        Y-offsets for each displayed channel.
    channel_spacing : float
        Spacing between channels.
    channels : list[int]
        Which channel indices are displayed.
    """

    def __init__(
        self,
        figure: Any,
        spike_data: SpikeData,
        channel_offsets: npt.NDArray[Any],
        channel_spacing: float,
        channels: list[int],
    ) -> None:
        self._fig = figure
        self.spike_data = spike_data
        self._offsets = channel_offsets
        self._spacing = channel_spacing
        self._channels = channels

        # Build channel index → display index mapping
        self._ch_to_idx: dict[int, int] = {ch: i for i, ch in enumerate(channels)}

        # Assign colors if not provided
        if spike_data.colors is None:
            from bokeh.palettes import Category10_10

            units = list(spike_data.spike_times.keys())
            spike_data.colors = {u: Category10_10[i % 10] for i, u in enumerate(units)}

        # Create sources
        self._box_source = ColumnDataSource(
            data={"left": [], "right": [], "top": [], "bottom": [], "fill_color": [], "fill_alpha": []}
        )
        self._box_glyph = figure.quad(
            left="left",
            right="right",
            top="top",
            bottom="bottom",
            fill_color="fill_color",
            fill_alpha="fill_alpha",
            line_color=None,
            source=self._box_source,
        )

        self._seg_source = ColumnDataSource(data={"xs": [], "ys": [], "line_color": []})
        self._seg_glyph = figure.multi_line(
            xs="xs",
            ys="ys",
            line_color="line_color",
            line_width=1.5,
            source=self._seg_source,
        )

        self._box_glyph.visible = False
        self._seg_glyph.visible = False

    def update(
        self,
        x_start: float,
        x_end: float,
        pixel_width: int,
        is_raw: bool,
        raw_result: WindowResult | None = None,
        fs: float = 30000.0,
    ) -> None:
        """Update the spike display for the current viewport.

        Parameters
        ----------
        x_start : float
            Viewport start in seconds.
        x_end : float
            Viewport end in seconds.
        pixel_width : int
            Figure width in pixels.
        is_raw : bool
            Whether the traces are showing raw data.
        raw_result : WindowResult or None
            The raw trace data (needed for colored segments in raw mode).
        fs : float
            Sampling frequency in Hz.
        """
        if is_raw and raw_result is not None:
            self._update_segments(x_start, x_end, raw_result, fs)
        else:
            self._update_boxes(x_start, x_end, pixel_width)

    def _update_boxes(self, x_start: float, x_end: float, pixel_width: int) -> None:
        """Compute and display spike boxes for the current viewport."""
        sd = self.spike_data
        bin_width = (x_end - x_start) / max(pixel_width, 1)
        half_h = self._spacing * sd.box_height_fraction

        lefts: list[float] = []
        rights: list[float] = []
        tops: list[float] = []
        bottoms: list[float] = []
        colors: list[str] = []
        alphas: list[float] = []

        assert sd.colors is not None
        for unit_id, times in sd.spike_times.items():
            ch = sd.unit_channels.get(unit_id)
            if ch is None or ch not in self._ch_to_idx:
                continue
            idx = self._ch_to_idx[ch]
            y_center = float(self._offsets[idx])
            color = sd.colors[unit_id]

            visible = times[(times >= x_start) & (times < x_end)]
            if len(visible) == 0:
                continue

            spikes_per_pixel = len(visible) / max(pixel_width, 1)

            if spikes_per_pixel < sd.sparse_threshold:
                # Sparse: individual ticks
                for t in visible:
                    lefts.append(t - bin_width / 4)
                    rights.append(t + bin_width / 4)
                    tops.append(y_center + half_h)
                    bottoms.append(y_center - half_h)
                    colors.append(color)
                    alphas.append(0.7)
            else:
                # Dense: histogram bins
                bins = np.arange(x_start, x_end + bin_width, bin_width)
                counts, edges = np.histogram(visible, bins=bins)
                max_count = max(counts.max(), 1)
                for j in range(len(counts)):
                    if counts[j] > 0:
                        alpha = 0.3 + 0.7 * counts[j] / max_count
                        lefts.append(float(edges[j]))
                        rights.append(float(edges[j + 1]))
                        tops.append(y_center + half_h)
                        bottoms.append(y_center - half_h)
                        colors.append(color)
                        alphas.append(float(alpha))

        self._box_source.data = {
            "left": lefts,
            "right": rights,
            "top": tops,
            "bottom": bottoms,
            "fill_color": colors,
            "fill_alpha": alphas,
        }
        self._box_glyph.visible = True
        self._seg_glyph.visible = False

    def _update_segments(self, x_start: float, x_end: float, result: WindowResult, fs: float) -> None:
        """Compute and display colored trace segments around spike times."""
        sd = self.spike_data
        window_samples = int(sd.spike_window_ms * fs / 1000)

        all_xs: list[npt.NDArray[Any]] = []
        all_ys: list[npt.NDArray[Any]] = []
        all_colors: list[str] = []

        assert sd.colors is not None
        for unit_id, times in sd.spike_times.items():
            ch = sd.unit_channels.get(unit_id)
            if ch is None or ch not in self._ch_to_idx:
                continue
            ch_idx = self._ch_to_idx[ch]
            offset = float(self._offsets[ch_idx])
            color = sd.colors[unit_id]

            visible = times[(times >= x_start) & (times < x_end)]
            for t in visible:
                t_sample = int(t * fs)
                s0 = max(t_sample - window_samples, result.t0_actual)
                s1 = min(t_sample + window_samples, result.t1_actual)
                idx0 = s0 - result.t0_actual
                idx1 = s1 - result.t0_actual
                if idx1 <= idx0:
                    continue
                xs = np.linspace(s0 / fs, s1 / fs, idx1 - idx0)
                ys = result.data[ch_idx, idx0:idx1] + offset
                all_xs.append(xs)
                all_ys.append(ys)
                all_colors.append(color)

        self._seg_source.data = {"xs": all_xs, "ys": all_ys, "line_color": all_colors}
        self._seg_glyph.visible = True
        self._box_glyph.visible = False


class TraceViewer:
    """Interactive multi-channel trace viewer backed by a MipmapStore.

    Parameters
    ----------
    store : MipmapStore
        The data backend.
    channels : list[int] or None, optional
        Channel indices to display.  ``None`` selects the first
        *n_channels* channels.
    n_channels : int, optional
        Number of channels to display when *channels* is ``None``,
        by default 32.
    channel_spacing : float, optional
        Y-offset between channels.  If 0 (default), auto-computed
        from the data.
    channel_order : str or list[int] or None, optional
        Display order for channels.  ``"index"`` (default) uses the
        natural order.  ``"depth"`` sorts by probe y-position (requires
        channel locations in the store).  A ``list[int]`` specifies an
        explicit ordering of channel indices.
    width : int, optional
        Figure width in pixels, by default 1200.
    height : int, optional
        Figure height in pixels, by default 800.
    spike_data : SpikeData or None, optional
        Spike times for overlay display.
    """

    def __init__(
        self,
        store: MipmapStore,
        channels: list[int] | None = None,
        n_channels: int = 32,
        channel_spacing: float = 0.0,
        channel_order: str | list[int] | None = None,
        width: int = 1200,
        height: int = 800,
        spike_data: SpikeData | None = None,
    ) -> None:
        self.store = store
        self.width = width
        self.height = height
        self._gain = 1.0

        # Resolve channels
        if channels is None:
            n_ch = min(n_channels, store.n_channels)
            channels_resolved = list(range(n_ch))
        else:
            channels_resolved = list(channels)

        # Apply channel ordering
        self.channels = _order_channels(channels_resolved, channel_order, store)

        # Compute spacing
        if channel_spacing <= 0:
            self.channel_spacing = self._compute_channel_spacing()
        else:
            self.channel_spacing = channel_spacing

        # Viewport state
        self._offsets = self._compute_channel_offsets()
        self._last_x_start = 0.0
        self._last_x_end = 0.0
        self._last_was_raw: bool | None = None

        # Background fetch executor — non-blocking data loading
        self._fetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fetch")
        self._fetch_future: Future[Any] | None = None
        self._fetch_generation: int = 0  # incremented on each viewport change
        self._pending_viewport: tuple[float, float] | None = None

        # Prefetch for remote stores — warm adjacent shards in background
        self._prefetch_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch") if store.is_remote else None
        )
        self._prefetch_futures: list[Future[Any]] = []

        # Build figure + Panel layout (figure pane is swappable)
        self._spike_data = spike_data
        self._build_figure()
        self._fig_pane = pn.pane.Bokeh(self._fig)
        self._controls = self._build_controls()
        self._layout = pn.Row(self._fig_pane, self._controls)

        self._initial_render()

    def _display_to_samples(self, x_start: float, x_end: float) -> tuple[int, int]:
        """Convert display-time range to clamped sample indices."""
        fs = self.store.sampling_frequency
        t0 = max(0, int((x_start - self.store.t_start) * fs))
        t1 = min(self.store.n_samples, int((x_end - self.store.t_start) * fs))
        return t0, t1

    def _samples_to_display(self, sample: int) -> float:
        """Convert a sample index to display time."""
        return sample / self.store.sampling_frequency + self.store.t_start

    def _compute_channel_spacing(self) -> float:
        """Auto-compute channel spacing from a data sample."""
        n_samples = min(10000, self.store.n_samples)
        result = self.store.get_window(0, n_samples, channels=self.channels, num_points=n_samples)
        if result.is_raw:
            # data shape: (C, T)
            ranges = result.data.max(axis=1) - result.data.min(axis=1)
        else:
            # data shape: (C, 2, B) — axis 1 is [min, max]
            ranges = result.data[:, 1, :].max(axis=1) - result.data[:, 0, :].min(axis=1)
        median_range = float(np.median(ranges))
        return median_range * 1.5 if median_range > 0 else 1.0

    def _compute_channel_offsets(self) -> npt.NDArray[Any]:
        """Compute y-offsets for each channel (top-to-bottom)."""
        return np.arange(len(self.channels), dtype=np.float64) * -self.channel_spacing * self._gain

    def _build_figure(self) -> None:
        """Create the Bokeh figure with glyphs and sources."""
        t_start = self._samples_to_display(0)
        t_end = self._samples_to_display(self.store.n_samples)

        self._fig = bokeh_figure(
            width=self.width,
            height=self.height,
            x_range=Range1d(t_start, t_end),
            y_range=Range1d(
                float(self._offsets[-1] - self.channel_spacing),
                float(self._offsets[0] + self.channel_spacing),
            ),
            tools="xpan,xwheel_zoom,reset",
            active_scroll="xwheel_zoom",
            active_drag="xpan",
        )
        self._fig.xaxis.axis_label = "Time (s)"

        # Channel labels on y-axis — use channel_ids if available
        ch_ids = self.store.channel_ids
        self._fig.yaxis.ticker = FixedTicker(ticks=self._offsets.tolist())
        self._fig.yaxis.major_label_overrides = {
            float(self._offsets[i]): (str(ch_ids[self.channels[i]]) if ch_ids is not None else str(self.channels[i]))
            for i in range(len(self.channels))
        }

        # Single source per display mode — one websocket push per render
        n_ch = len(self.channels)
        colors = _channel_colors(n_ch)

        # Envelope source: columns x, y1_0, y2_0, y1_1, y2_1, ...
        env_data: dict[str, list[Any]] = {"x": []}
        for i in range(n_ch):
            env_data[f"y1_{i}"] = []
            env_data[f"y2_{i}"] = []
        self._envelope_source = ColumnDataSource(data=env_data)

        self._envelope_glyphs: list[Any] = []
        for i in range(n_ch):
            g = self._fig.varea(
                x="x",
                y1=f"y1_{i}",
                y2=f"y2_{i}",
                source=self._envelope_source,
                fill_alpha=0.7,
                color=colors[i],
            )
            g.visible = False
            self._envelope_glyphs.append(g)

        # Line source: columns x, y_0, y_1, ...
        line_data: dict[str, list[Any]] = {"x": []}
        for i in range(n_ch):
            line_data[f"y_{i}"] = []
        self._line_source = ColumnDataSource(data=line_data)

        self._line_glyphs: list[Any] = []
        for i in range(n_ch):
            g = self._fig.line(
                x="x",
                y=f"y_{i}",
                source=self._line_source,
                line_width=0.5,
                color=colors[i],
            )
            g.visible = False
            self._line_glyphs.append(g)

        # Register range change callback
        self._fig.x_range.on_change("start", self._on_range_change)  # type: ignore[arg-type]
        self._fig.x_range.on_change("end", self._on_range_change)  # type: ignore[arg-type]

        # Create spike overlay if spike data provided
        self._spike_overlay: SpikeOverlay | None = None
        if self._spike_data is not None:
            self._spike_overlay = SpikeOverlay(
                self._fig, self._spike_data, self._offsets, self.channel_spacing, self.channels
            )

    def _initial_render(self) -> None:
        """Render the initial view (full recording)."""
        t_start = self._samples_to_display(0)
        t_end = self._samples_to_display(self.store.n_samples)
        self._update_data(t_start, t_end)

    def _on_range_change(self, attr: str, old: Any, new: Any) -> None:
        """Handle Bokeh x_range change events.

        Submits a non-blocking fetch.  If a fetch is already in flight,
        the viewport is saved and picked up when the current fetch
        completes — naturally throttling to one fetch at a time while
        keeping panning smooth.
        """
        x_start = float(self._fig.x_range.start)  # type: ignore[arg-type]
        x_end = float(self._fig.x_range.end)  # type: ignore[arg-type]
        log.debug("range_change: x=[%.2f, %.2f]", x_start, x_end)

        # Threshold skip: ignore if < 1% change
        width = x_end - x_start
        center = (x_start + x_end) / 2.0
        last_width = self._last_x_end - self._last_x_start
        last_center = (self._last_x_start + self._last_x_end) / 2.0
        if last_width > 0:
            width_change = abs(width - last_width) / last_width
            center_change = abs(center - last_center) / last_width
            if width_change < 0.01 and center_change < 0.01:
                return

        self._request_update(x_start, x_end)

    def _request_update(self, x_start: float, x_end: float) -> None:
        """Submit a non-blocking data fetch for the given viewport.

        The fetch runs in a background thread.  When it completes, the
        render is scheduled back on the Bokeh document's event loop via
        ``add_next_tick_callback``.  Stale fetches (superseded by a
        newer viewport change) are discarded on completion.

        If a fetch is already in flight, it is not cancelled but the
        desired viewport is saved.  When the in-flight fetch completes
        (and finds its generation stale), a new fetch is automatically
        submitted for the latest desired viewport.
        """
        t0, t1 = self._display_to_samples(x_start, x_end)
        if t0 >= t1:
            log.debug("request_update: empty window t0=%d t1=%d, skipping", t0, t1)
            return

        # Skip if a fetch is already running — save the desired viewport
        # so we can re-submit when the current one finishes.
        if self._fetch_future is not None and not self._fetch_future.done():
            log.debug("request_update: fetch in flight, saving pending viewport")
            self._pending_viewport = (x_start, x_end)
            return
        self._pending_viewport = None

        # Increment generation only when actually submitting a fetch
        self._fetch_generation += 1
        gen = self._fetch_generation
        log.debug("request_update: submitting fetch gen=%d t0=%d t1=%d", gen, t0, t1)

        def _fetch_and_render() -> None:
            import time

            t_start_fetch = time.perf_counter()
            result = self.store.get_window(t0, t1, channels=self.channels, num_points=self.width)
            t_fetched = time.perf_counter()
            log.debug(
                "fetch gen=%d: %.1fms, %s shape=%s",
                gen,
                (t_fetched - t_start_fetch) * 1000,
                "raw" if result.is_raw else f"s{result.stride}",
                result.data.shape,
            )

            # Mark fetch as done before checking pending, so
            # _request_update won't see us as "in flight".
            self._fetch_future = None

            # If user panned while we were fetching, skip render and
            # fetch the latest viewport instead.
            pending = self._pending_viewport
            if pending is not None:
                log.debug("fetch gen=%d: pending viewport, skipping", gen)
                self._pending_viewport = None
                self._request_update(*pending)
                return

            # Prepare display data on this thread (downsampling + dict building)
            # so _apply only does the source assignment on the event loop.
            display = self._prepare_display_data(result)
            t_prepared = time.perf_counter()
            log.debug(
                "fetch gen=%d: prepare %.1fms, %d points",
                gen,
                (t_prepared - t_fetched) * 1000,
                len(display["data"]["x"]),
            )

            def _apply() -> None:
                import time as _time

                if gen != self._fetch_generation:
                    log.debug("apply gen=%d: stale (current=%d), skipping", gen, self._fetch_generation)
                    return

                t_apply_start = _time.perf_counter()
                self._last_x_start = x_start
                self._last_x_end = x_end

                # Apply pre-computed display data (single source assignment)
                use_envelope = display["mode"] == "envelope"
                if self._last_was_raw is not (not use_envelope):
                    for i in range(len(self.channels)):
                        self._envelope_glyphs[i].visible = use_envelope
                        self._line_glyphs[i].visible = not use_envelope

                if use_envelope:
                    self._envelope_source.data = display["data"]
                else:
                    self._line_source.data = display["data"]
                self._last_was_raw = not use_envelope
                log.debug(
                    "apply gen=%d: %.1fms, mode=%s", gen, (_time.perf_counter() - t_apply_start) * 1000, display["mode"]
                )

                # Update spike overlay
                if self._spike_overlay is not None:
                    self._spike_overlay.update(
                        x_start,
                        x_end,
                        self.width,
                        result.is_raw,
                        raw_result=result if result.is_raw else None,
                        fs=self.store.sampling_frequency,
                    )

                # Prefetch adjacent shard for remote stores
                if self._prefetch_executor is not None:
                    self._prefetch_adjacent(x_start, x_end)

            # Schedule render on Bokeh's event loop (thread-safe)
            doc = self._fig.document
            if doc is not None:
                log.debug("fetch gen=%d: scheduling _apply via add_next_tick_callback", gen)
                doc.add_next_tick_callback(_apply)
            else:
                log.debug("fetch gen=%d: no document, calling _apply directly", gen)
                _apply()

        self._fetch_future = self._fetch_executor.submit(_fetch_and_render)

    def _update_data(self, x_start: float, x_end: float) -> None:
        """Fetch data and update all sources (blocking).

        Used for the initial render before the Bokeh document is
        available.  After that, :meth:`_request_update` is used for
        non-blocking updates.
        """
        t0, t1 = self._display_to_samples(x_start, x_end)
        if t0 >= t1:
            return

        result = self.store.get_window(t0, t1, channels=self.channels, num_points=self.width)
        display = self._prepare_display_data(result)

        self._last_x_start = x_start
        self._last_x_end = x_end

        use_envelope = display["mode"] == "envelope"
        if self._last_was_raw is not (not use_envelope):
            for i in range(len(self.channels)):
                self._envelope_glyphs[i].visible = use_envelope
                self._line_glyphs[i].visible = not use_envelope

        if use_envelope:
            self._envelope_source.data = display["data"]
        else:
            self._line_source.data = display["data"]
        self._last_was_raw = not use_envelope

        # Update spike overlay
        if self._spike_overlay is not None:
            self._spike_overlay.update(
                x_start,
                x_end,
                self.width,
                result.is_raw,
                raw_result=result if result.is_raw else None,
                fs=self.store.sampling_frequency,
            )

        # Prefetch adjacent shard for remote stores
        if self._prefetch_executor is not None:
            self._prefetch_adjacent(x_start, x_end)

    def _prefetch_adjacent(self, x_start: float, x_end: float) -> None:
        """Warm the fsspec cache for the shard adjacent to the pan direction.

        Submits a minimal read (1 element) from the next shard in the
        pan direction.  With fsspec ``filecache``, this triggers the
        full shard file download, making subsequent reads local.
        """
        assert self._prefetch_executor is not None

        # Cancel stale prefetches
        for f in self._prefetch_futures:
            f.cancel()
        self._prefetch_futures.clear()

        width = x_end - x_start

        # Detect pan direction from center shift
        last_center = (self._last_x_start + self._last_x_end) / 2
        cur_center = (x_start + x_end) / 2
        if cur_center >= last_center:
            adj_start = x_end  # panning right — prefetch right
        else:
            adj_start = x_start - width  # panning left — prefetch left

        adj_t0, adj_t1 = self._display_to_samples(adj_start, adj_start + width)
        if adj_t0 >= adj_t1:
            return

        # Plan adjacent viewport to find which level/blocks
        try:
            plan = self.store.plan_window(adj_t0, adj_t1, num_points=self.width)
        except (ValueError, RuntimeError):
            return  # out of bounds or raw unavailable
        if plan.stride is None or plan.block_start is None:
            return  # raw level — shards too large to prefetch

        # Check if adjacent falls in a different shard
        mm_array = self.store._minmax_arrays[plan.stride]
        if mm_array.shards is None:
            return  # no sharding — can't prefetch by shard
        shard_blocks = mm_array.shards[2]  # block-axis shard size
        cur_t0, _ = self._display_to_samples(x_start, x_end)
        cur_shard = (cur_t0 // plan.stride) // shard_blocks
        adj_shard = plan.block_start // shard_blocks

        if adj_shard == cur_shard:
            return  # same shard — already cached

        # Minimal read to trigger shard download
        touch_block = min(adj_shard * shard_blocks, mm_array.shape[2] - 1)

        def _warm() -> None:
            _ = np.asarray(mm_array[:1, :1, touch_block : touch_block + 1])

        self._prefetch_futures.append(self._prefetch_executor.submit(_warm))

    def _prepare_display_data(self, result: WindowResult) -> dict[str, Any]:
        """Prepare downsampled display data dict on the fetch thread.

        Returns a dict with ``mode`` (``"envelope"`` or ``"line"``) and
        ``data`` (the ColumnDataSource dict ready for assignment).
        """
        if result.is_raw:
            return self._prepare_raw_display(result)
        return self._prepare_envelope_display(result)

    def _prepare_envelope_display(self, result: WindowResult) -> dict[str, Any]:
        """Prepare envelope display data, downsampling if needed."""
        stride = result.stride
        assert stride is not None
        data = result.data  # (C, 2, B)
        n_blocks = data.shape[2]

        # Downsample to ~2x pixel width by re-bucketing min/max
        max_blocks = self.width * 2
        effective_stride = stride
        if n_blocks > max_blocks:
            bucket_size = n_blocks // max_blocks
            n_full = (n_blocks // bucket_size) * bucket_size
            full = data[:, :, :n_full].reshape(data.shape[0], 2, -1, bucket_size)
            mins = full[:, 0, :, :].min(axis=2)
            maxs = full[:, 1, :, :].max(axis=2)
            if n_full < n_blocks:
                tail = data[:, :, n_full:]
                tail_min = tail[:, 0, :].min(axis=1, keepdims=True)
                tail_max = tail[:, 1, :].max(axis=1, keepdims=True)
                mins = np.concatenate([mins, tail_min], axis=1)
                maxs = np.concatenate([maxs, tail_max], axis=1)
            data = np.stack([mins, maxs], axis=1)
            n_blocks = data.shape[2]
            effective_stride = stride * bucket_size

        sample_indices = np.arange(n_blocks) * effective_stride + result.t0_actual
        x = sample_indices / self.store.sampling_frequency + self.store.t_start

        new_data: dict[str, Any] = {"x": x}
        for i in range(len(self.channels)):
            new_data[f"y1_{i}"] = data[i, 0, :] + self._offsets[i]
            new_data[f"y2_{i}"] = data[i, 1, :] + self._offsets[i]
        return {"mode": "envelope", "data": new_data}

    def _prepare_raw_display(self, result: WindowResult) -> dict[str, Any]:
        """Prepare raw display data, switching to envelope if too many points."""
        data = result.data
        n_samples = data.shape[1]
        max_points = self.width * 2

        if n_samples <= max_points:
            x = np.linspace(
                self._samples_to_display(result.t0_actual),
                self._samples_to_display(result.t1_actual),
                n_samples,
            )
            new_data: dict[str, Any] = {"x": x}
            for i in range(len(self.channels)):
                new_data[f"y_{i}"] = data[i, :] + self._offsets[i]
            return {"mode": "line", "data": new_data}

        # Too many points — min/max bucket into envelope
        bucket_size = n_samples // self.width
        n_full = (n_samples // bucket_size) * bucket_size
        full = data[:, :n_full].reshape(data.shape[0], -1, bucket_size)
        mins = full.min(axis=2)
        maxs = full.max(axis=2)
        n_buckets = mins.shape[1]

        x = np.linspace(
            self._samples_to_display(result.t0_actual),
            self._samples_to_display(result.t0_actual + n_full),
            n_buckets,
        )
        new_data = {"x": x}
        for i in range(len(self.channels)):
            new_data[f"y1_{i}"] = mins[i, :] + self._offsets[i]
            new_data[f"y2_{i}"] = maxs[i, :] + self._offsets[i]
        return {"mode": "envelope", "data": new_data}

    def _update_layout(self) -> None:
        """Recompute offsets and y-range after spacing/gain change."""
        self._offsets = self._compute_channel_offsets()
        self._fig.y_range.start = float(self._offsets[-1] - self.channel_spacing * self._gain)  # type: ignore[assignment]
        self._fig.y_range.end = float(self._offsets[0] + self.channel_spacing * self._gain)  # type: ignore[assignment]
        self._fig.yaxis.ticker = FixedTicker(ticks=self._offsets.tolist())  # type: ignore[assignment]
        self._fig.yaxis.major_label_overrides = {
            float(self._offsets[i]): str(self.channels[i]) for i in range(len(self.channels))
        }
        # Re-render with current viewport
        self._update_data(
            float(self._fig.x_range.start),  # type: ignore[arg-type]
            float(self._fig.x_range.end),  # type: ignore[arg-type]
        )

    def set_channels(self, channels: list[int]) -> None:
        """Change which channels are displayed.

        Rebuilds the Bokeh figure and swaps it into the Panel pane,
        preserving the current time viewport.

        Parameters
        ----------
        channels : list[int]
            New channel indices.
        """
        # Invalidate any in-flight fetches (they have the wrong channel count)
        self._fetch_generation += 1
        self._pending_viewport = None

        # Preserve current viewport
        x_start = float(self._fig.x_range.start)  # type: ignore[arg-type]
        x_end = float(self._fig.x_range.end)  # type: ignore[arg-type]

        self.channels = list(channels)
        self._offsets = self._compute_channel_offsets()
        self._last_was_raw = None  # force visibility reset
        self._build_figure()

        # Restore viewport
        self._fig.x_range.start = x_start  # type: ignore[assignment]
        self._fig.x_range.end = x_end  # type: ignore[assignment]

        # Swap figure into the Panel pane
        self._fig_pane.object = self._fig

        # Render with preserved viewport (blocking — need correct data before display)
        self._update_data(x_start, x_end)

    def set_gain(self, gain: float) -> None:
        """Change the amplitude gain (channel spacing multiplier).

        Parameters
        ----------
        gain : float
            Gain multiplier (1.0 = default spacing).
        """
        self._gain = gain
        self._update_layout()

    def _build_controls(self) -> pn.Column:
        """Build the control panel widgets."""
        max_ch = self.store.n_channels - 1

        channel_start = pn.widgets.IntSlider(
            name="Channel start",
            start=0,
            end=max_ch,
            value=self.channels[0] if self.channels else 0,
            step=1,
        )
        n_visible = pn.widgets.IntSlider(
            name="Visible channels",
            start=1,
            end=min(128, self.store.n_channels),
            value=len(self.channels),
            step=1,
        )
        gain_slider = pn.widgets.FloatSlider(
            name="Gain",
            start=0.1,
            end=10.0,
            value=1.0,
            step=0.1,
        )

        def _on_channel_change(event: Any) -> None:
            start = channel_start.value
            n = n_visible.value
            end = min(start + n, self.store.n_channels)
            self.set_channels(list(range(start, end)))

        def _on_gain_change(event: Any) -> None:
            self.set_gain(gain_slider.value)

        channel_start.param.watch(_on_channel_change, "value")
        n_visible.param.watch(_on_channel_change, "value")
        gain_slider.param.watch(_on_gain_change, "value")

        return pn.Column(channel_start, n_visible, gain_slider, width=250)

    def panel(self) -> pn.Row:
        """Return the Panel layout with the figure and controls.

        Returns
        -------
        pn.Row
        """
        return self._layout

    def show(self, **kwargs: Any) -> None:
        """Launch the viewer in a browser tab.

        Parameters
        ----------
        **kwargs
            Passed to ``pn.serve()``.
        """
        pn.serve(self.panel(), show=True, **kwargs)


def _order_channels(
    channels: list[int],
    order: str | list[int] | None,
    store: MipmapStore,
) -> list[int]:
    """Sort channel indices according to the requested order.

    Parameters
    ----------
    channels : list[int]
        Channel indices to sort.
    order : str, list[int], or None
        ``"index"`` or ``None``: natural order (no change).
        ``"depth"``: sort by probe y-position (deepest first, requires
        channel locations in the store).
        ``list[int]``: explicit ordering — only channels present in both
        *channels* and *order* are shown, in the order specified.
    store : MipmapStore
        Used to read channel locations for ``"depth"`` ordering.

    Returns
    -------
    list[int]
        Sorted channel indices.
    """
    if order is None or order == "index":
        return channels

    if isinstance(order, list):
        # Explicit order — filter to channels that are in the requested set
        ch_set = set(channels)
        return [ch for ch in order if ch in ch_set]

    if order == "depth":
        locations = store.channel_locations
        if locations is None:
            return channels  # no location data, fall back to index order
        # Sort by y-position (column 1). Deepest (smallest y) first = top of display.
        ch_set = set(channels)
        ch_with_depth = [(ch, float(locations[ch, 1])) for ch in channels if ch < len(locations)]
        ch_with_depth.sort(key=lambda x: x[1])
        return [ch for ch, _ in ch_with_depth]

    msg = f"Unknown channel_order: {order!r}. Use 'index', 'depth', or a list of channel indices."
    raise ValueError(msg)


def _channel_colors(n: int) -> list[str]:
    """Generate a list of N channel colors from Bokeh's Category10 palette."""
    from bokeh.palettes import Category10_10

    return [Category10_10[i % len(Category10_10)] for i in range(n)]


def view(
    store_or_path: MipmapStore | str | Path,
    channels: list[int] | None = None,
    n_channels: int = 32,
    channel_order: str | list[int] | None = None,
    spike_data: SpikeData | None = None,
    cache_dir: str | Path | None = None,
    cache_max_gb: float = 10,
    **kwargs: Any,
) -> TraceViewer:
    """Open an interactive trace viewer.

    Parameters
    ----------
    store_or_path : MipmapStore, str, or Path
        A :class:`MipmapStore` or path to a mipmap Zarr store.
    channels : list[int] or None, optional
        Channel indices to display.
    n_channels : int, optional
        Number of channels when *channels* is ``None``, by default 32.
    channel_order : str, list[int], or None, optional
        Display order: ``"index"`` (default), ``"depth"`` (by probe
        y-position), or explicit ``list[int]``.
    cache_dir : str, Path, or None, optional
        Cache directory for remote stores.  ``None`` auto-detects.
    cache_max_gb : float, optional
        Maximum cache size in GB.  Default 10.
    **kwargs
        Passed to :meth:`TraceViewer.show`.

    Returns
    -------
    TraceViewer
    """
    if not isinstance(store_or_path, MipmapStore):
        store = open_mipmap(str(store_or_path), cache_dir=cache_dir, cache_max_gb=cache_max_gb)
    else:
        store = store_or_path
    viewer = TraceViewer(
        store,
        channels=channels,
        n_channels=n_channels,
        channel_order=channel_order,
        spike_data=spike_data,
    )
    viewer.show(**kwargs)
    return viewer
