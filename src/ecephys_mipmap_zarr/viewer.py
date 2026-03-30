"""Interactive multi-channel trace viewer backed by mipmap Zarr stores.

Requires the ``viewer`` extra::

    pip install ecephys-mipmap-zarr[viewer]

Usage::

    from ecephys_mipmap_zarr.viewer import view
    view("mipmap.zarr")  # opens browser tab
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import panel as pn
from bokeh.models import ColumnDataSource, FixedTicker, Range1d
from bokeh.plotting import figure as bokeh_figure

from ecephys_mipmap_zarr.read import MipmapStore, WindowResult, open_mipmap

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

        # Build Bokeh figure
        self._offsets = self._compute_channel_offsets()
        self._last_x_start = 0.0
        self._last_x_end = 0.0
        self._last_was_raw: bool | None = None

        self._build_figure()

        # Create spike overlay if spike data provided
        self._spike_overlay: SpikeOverlay | None = None
        if spike_data is not None:
            self._spike_overlay = SpikeOverlay(
                self._fig, spike_data, self._offsets, self.channel_spacing, self.channels
            )

        self._initial_render()

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
        fs = self.store.sampling_frequency
        duration = self.store.n_samples / fs

        self._fig = bokeh_figure(
            width=self.width,
            height=self.height,
            x_range=Range1d(0, duration),
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

        # Create sources and glyphs for each channel
        self._envelope_sources: list[ColumnDataSource] = []
        self._line_sources: list[ColumnDataSource] = []
        self._envelope_glyphs: list[Any] = []
        self._line_glyphs: list[Any] = []

        colors = _channel_colors(len(self.channels))

        for i in range(len(self.channels)):
            # Envelope source (varea)
            env_src = ColumnDataSource(data={"x": [], "y1": [], "y2": []})
            env_glyph = self._fig.varea(x="x", y1="y1", y2="y2", source=env_src, fill_alpha=0.7, color=colors[i])
            env_glyph.visible = False
            self._envelope_sources.append(env_src)
            self._envelope_glyphs.append(env_glyph)

            # Line source
            line_src = ColumnDataSource(data={"x": [], "y": []})
            line_glyph = self._fig.line(x="x", y="y", source=line_src, line_width=0.5, color=colors[i])
            line_glyph.visible = False
            self._line_sources.append(line_src)
            self._line_glyphs.append(line_glyph)

        # Register range change callback
        self._fig.x_range.on_change("start", self._on_range_change)  # type: ignore[arg-type]
        self._fig.x_range.on_change("end", self._on_range_change)  # type: ignore[arg-type]

    def _initial_render(self) -> None:
        """Render the initial view (full recording)."""
        self._update_data(0.0, self.store.n_samples / self.store.sampling_frequency)

    def _on_range_change(self, attr: str, old: Any, new: Any) -> None:
        """Handle Bokeh x_range change events."""
        x_start = float(self._fig.x_range.start)  # type: ignore[arg-type]
        x_end = float(self._fig.x_range.end)  # type: ignore[arg-type]

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

        self._update_data(x_start, x_end)

    def _update_data(self, x_start: float, x_end: float) -> None:
        """Fetch data and update all sources."""
        fs = self.store.sampling_frequency
        t0 = max(0, int(x_start * fs))
        t1 = min(self.store.n_samples, int(x_end * fs))
        if t0 >= t1:
            return

        result = self.store.get_window(t0, t1, channels=self.channels, num_points=self.width)

        self._last_x_start = x_start
        self._last_x_end = x_end

        if result.is_raw:
            self._render_raw(result)
        else:
            self._render_envelope(result)

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

    def _render_envelope(self, result: WindowResult) -> None:
        """Update sources for envelope (minmax) display."""
        fs = self.store.sampling_frequency
        stride = result.stride
        assert stride is not None
        n_blocks = result.data.shape[2]  # (C, 2, B)
        x = (np.arange(n_blocks) * stride + result.t0_actual) / fs

        for i in range(len(self.channels)):
            offset = self._offsets[i]
            self._envelope_sources[i].data = {
                "x": x,
                "y1": result.data[i, 0, :] + offset,
                "y2": result.data[i, 1, :] + offset,
            }
            self._envelope_glyphs[i].visible = True
            self._line_glyphs[i].visible = False

        self._last_was_raw = False

    def _render_raw(self, result: WindowResult) -> None:
        """Update sources for raw trace display."""
        fs = self.store.sampling_frequency
        n_samples = result.data.shape[1]  # (C, T)
        x = np.linspace(result.t0_actual / fs, result.t1_actual / fs, n_samples)

        for i in range(len(self.channels)):
            offset = self._offsets[i]
            self._line_sources[i].data = {
                "x": x,
                "y": result.data[i, :] + offset,
            }
            self._line_glyphs[i].visible = True
            self._envelope_glyphs[i].visible = False

        self._last_was_raw = True

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

        Parameters
        ----------
        channels : list[int]
            New channel indices.
        """
        self.channels = list(channels)
        # Rebuild figure with new channels
        self._build_figure()
        self._offsets = self._compute_channel_offsets()
        self._initial_render()

    def set_gain(self, gain: float) -> None:
        """Change the amplitude gain (channel spacing multiplier).

        Parameters
        ----------
        gain : float
            Gain multiplier (1.0 = default spacing).
        """
        self._gain = gain
        self._update_layout()

    def panel(self) -> pn.Column:
        """Return a Panel layout with the figure and controls.

        Returns
        -------
        pn.Column
        """
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

        controls = pn.Column(channel_start, n_visible, gain_slider, width=250)
        return pn.Row(self._fig, controls)

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
