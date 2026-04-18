"""Interactive multi-channel trace viewer backed by mipmap Zarr stores.

Requires the ``viewer`` extra::

    pip install ecephys-mipmap-zarr[viewer]

Usage::

    from ecephys_mipmap_zarr.viewer import view
    view("mipmap.zarr")  # opens browser tab
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import panel as pn
from bokeh.models import ColumnDataSource, FixedTicker, Label, Range1d
from bokeh.plotting import figure as bokeh_figure

from ecephys_mipmap_zarr.read import MipmapStore, WindowResult, open_mipmap

log = logging.getLogger(__name__)


DEFAULT_BOKEH_PORT = 5006


@dataclass
class RemoteConfig:
    """Configuration for running the viewer behind a reverse proxy.

    Plain ``pn.serve(show=True)`` is fine for local use but breaks under
    the proxies used by cloud IDEs (Code Ocean, JupyterHub with
    ``jupyter-server-proxy``, etc.) because Bokeh rejects websocket
    upgrades whose ``Origin`` is not in ``allow_websocket_origin`` and
    generates asset URLs that assume a root-level mount.

    Use :meth:`auto` to build a config that infers sensible defaults
    from the environment (notably ``JUPYTERHUB_SERVICE_PREFIX``).

    Parameters
    ----------
    port : int or None
        Bind port.  ``None`` lets Panel pick a free port.
    url_prefix : str or None
        External URL prefix (e.g. ``/user/me/proxy/5006``) passed to
        Panel as ``prefix``.  Needed when the app is mounted under a
        path prefix by ``jupyter-server-proxy``.
    allow_origin : list[str] or None
        Origins permitted to open a websocket.  ``None`` expands to
        ``["*"]`` (wildcard — wide-open, log warns).  Pass ``["*"]``
        explicitly to silence the warning.  For Code Ocean code-server
        the bound URL's Origin is unpredictable, so wildcard is usually
        the right choice.
    """

    port: int | None = None
    url_prefix: str | None = None
    allow_origin: list[str] | None = None

    @classmethod
    def auto(
        cls,
        *,
        port: int | None = None,
        url_prefix: str | None = None,
        allow_origin: list[str] | None = None,
    ) -> RemoteConfig:
        """Construct a :class:`RemoteConfig` with environment-derived defaults.

        - When ``url_prefix`` is not given and ``JUPYTERHUB_SERVICE_PREFIX``
          is set, derive ``<hub_prefix>/proxy/<port>`` so Panel's asset
          URLs resolve through ``jupyter-server-proxy``.
        - Other fields pass through unchanged.
        """
        hub_prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")
        if url_prefix is None and hub_prefix:
            resolved_port = port if port is not None else DEFAULT_BOKEH_PORT
            url_prefix = f"{hub_prefix.rstrip('/')}/proxy/{resolved_port}"
            log.info(
                "auto-derived url_prefix from JUPYTERHUB_SERVICE_PREFIX: %s",
                url_prefix,
            )
        return cls(port=port, url_prefix=url_prefix, allow_origin=allow_origin)

    def to_serve_kwargs(self) -> dict[str, Any]:
        """Return kwargs for :func:`panel.serve`.

        When ``allow_origin`` is unset, default to ``["*"]`` (wide open)
        and warn via the module logger.  Bokeh's ``allow_websocket_origin``
        requires a list — a bare ``"*"`` string fails validation.
        """
        if self.allow_origin is None:
            log.warning(
                "RemoteConfig: allow_origin unset — using wildcard websocket origin (world-readable). "
                "Pass a concrete hostname via allow_origin to tighten.",
            )
            origin = ["*"]
        else:
            origin = self.allow_origin

        kwargs: dict[str, Any] = {
            "address": "0.0.0.0",
            "show": False,
            "allow_websocket_origin": origin,
        }
        if self.port is not None:
            kwargs["port"] = self.port
        if self.url_prefix is not None:
            kwargs["prefix"] = self.url_prefix
        return kwargs

    def describe_url(self) -> str:
        """Return a human-readable description of the URL the app will be at."""
        port = self.port if self.port is not None else DEFAULT_BOKEH_PORT
        prefix = self.url_prefix or ""
        if self.url_prefix:
            return f"(via proxy) <host>{prefix} (bound on 0.0.0.0:{port})"
        return f"http://<host>:{port}{prefix}"


__all__ = [
    "SpikeData",
    "SpikeOverlay",
    "TraceViewer",
    "load_phy_spikes",
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
        t_start: float = 0.0,
    ) -> None:
        self._fig = figure
        self.spike_data = spike_data
        self._offsets = channel_offsets
        self._spacing = channel_spacing
        self._channels = channels
        self._t_start = t_start

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
                # Convert display time → sample index (subtract t_start)
                t_sample = int((t - self._t_start) * fs)
                s0 = max(t_sample - window_samples, result.t0_actual)
                s1 = min(t_sample + window_samples, result.t1_actual)
                idx0 = s0 - result.t0_actual
                idx1 = s1 - result.t0_actual
                if idx1 <= idx0:
                    continue
                # Convert sample indices back to display time for x coords
                xs = np.linspace(s0 / fs + self._t_start, s1 / fs + self._t_start, idx1 - idx0)
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
        channel_order: str | list[int] | None = "depth",
        initial_channels: list[int] | None = None,
        width: int = 1200,
        height: int = 800,
        spike_data: SpikeData | None = None,
    ) -> None:
        self.store = store
        self.width = width
        self.height = height
        self._gain = 1.0

        # Cache channel_ids to avoid repeated zarr reads during panning
        self._channel_ids = store.channel_ids

        # Resolve channels:
        # _all_channels = scrollable set (all store channels, or --channels subset)
        # self.channels = currently visible subset
        # initial_channels = starting view position (doesn't restrict scrolling)
        if channels is None:
            all_ch = list(range(store.n_channels))
        else:
            all_ch = list(channels)
        self._all_channels = _order_channels(all_ch, channel_order, store)
        self._ch_to_all_idx = {ch: i for i, ch in enumerate(self._all_channels)}

        if initial_channels is not None:
            # Start viewing these channels, but allow panning to all
            valid = [ch for ch in initial_channels if ch in self._ch_to_all_idx]
            self.channels = valid if valid else list(self._all_channels[:n_channels])
        else:
            n_visible = min(n_channels, len(self._all_channels))
            self.channels = list(self._all_channels[:n_visible])

        # Compute spacing
        if channel_spacing <= 0:
            self.channel_spacing = self._compute_channel_spacing()
        else:
            self.channel_spacing = channel_spacing

        # All-channel y-positions (for mapping y_range to channel indices)
        self._all_offsets = np.arange(len(self._all_channels), dtype=np.float64) * -self.channel_spacing

        # Viewport state
        self._offsets = self._compute_channel_offsets()
        self._last_x_start = 0.0
        self._last_x_end = 0.0
        self._last_was_raw: bool | None = None
        self._last_result: WindowResult | None = None  # cached for gain re-render

        # Background fetch executor — non-blocking data loading
        self._fetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fetch")
        self._fetch_future: Future[Any] | None = None
        self._fetch_generation: int = 0  # incremented on each viewport change
        self._pending_viewport: tuple[float, float] | None = None

        # Debounce timer for grouping x+y range changes into a single update.
        # During diagonal panning, Bokeh fires 4 separate callbacks (x start,
        # x end, y start, y end).  We schedule a single deferred update that
        # reads the final state after all callbacks have fired.
        self._update_scheduled: bool = False

        # Prefetch for remote stores — warm adjacent shards in background
        self._prefetch_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch") if store.is_remote else None
        )
        self._prefetch_futures: list[Future[Any]] = []

        # Build figure + Panel layout (figure pane is swappable)
        self._spike_data = spike_data
        self._build_figure()
        self._fig_pane = pn.pane.Bokeh(self._fig)
        self._scale_pane = pn.pane.Bokeh(self._scale_fig)
        self._controls = self._build_controls()
        self._layout = pn.Row(self._fig_pane, self._scale_pane, self._controls)

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
        """Compute y-offsets for the currently visible channels.

        Looks up each visible channel's position in ``_all_offsets`` so
        that y-positions are consistent when panning through channels.
        """
        indices = [self._ch_to_all_idx[ch] for ch in self.channels if ch in self._ch_to_all_idx]
        return self._all_offsets[indices]

    def _compute_scale_bar_value(self) -> float:
        """Pick a nice round value for the scale bar (~75% of channel spacing)."""
        target = self.channel_spacing * 0.75 / max(self._gain, 0.01)
        # Round down to nearest nice value (multiples of 1, 2, 5 × powers of 10)
        nice_values = [
            0.1,
            0.2,
            0.5,
            1,
            2,
            5,
            10,
            20,
            50,
            100,
            200,
            500,
            1000,
            2000,
            5000,
            10000,
            20000,
            50000,
        ]
        # Pick largest nice value that doesn't exceed the target
        result = nice_values[0]
        for v in nice_values:
            if v <= target:
                result = v
            else:
                break
        return result

    @staticmethod
    def _format_scale_label(value: float, units: str) -> str:
        """Format a scale bar label with appropriate unit prefix."""
        if units.lower() in ("uv", "µv") and value >= 1000:
            return f"{value / 1000:g} mV"
        if value == int(value):
            return f"{int(value)} {units}"
        return f"{value:g} {units}"

    def _scale_units(self) -> str:
        """Return the effective display units for the scale bar.

        When the store has ``RawQuantization`` (ADC → µV conversion),
        the data is in physical units.  Otherwise (only ``lsb_shift``
        or raw int), the data is in ADC units.
        """
        if self.store._raw_quantization is not None:
            return self.store._attrs.units
        return "ADC"

    def _build_scale_bar(self) -> None:
        """Build a narrow figure to the right of the main plot for the scale bar.

        Shares ``y_range`` with the main figure so the bar stays
        proportional on vertical pan and gain changes automatically.
        """
        bar_value = self._compute_scale_bar_value()
        units = self._scale_units()

        self._scale_fig = bokeh_figure(
            width=80,
            height=self.height,
            x_range=Range1d(0, 1),
            y_range=self._fig.y_range,  # linked to main figure
            tools="",
            toolbar_location=None,
            outline_line_color=None,
        )
        self._scale_fig.axis.visible = False
        self._scale_fig.grid.visible = False
        self._scale_fig.min_border_left = 0
        self._scale_fig.min_border_right = 0

        # Position bar near bottom of visible y-range
        y_bottom = float(self._fig.y_range.start) + self.channel_spacing * 0.2  # type: ignore[arg-type]
        y_top = y_bottom + bar_value * self._gain

        self._scale_bar_source = ColumnDataSource(data={"x0": [0.5], "y0": [y_bottom], "x1": [0.5], "y1": [y_top]})
        self._scale_fig.segment(
            x0="x0",
            y0="y0",
            x1="x1",
            y1="y1",
            source=self._scale_bar_source,
            line_width=3,
            line_color="black",
        )
        self._scale_bar_label = Label(
            x=0.55,
            y=(y_bottom + y_top) / 2,
            text=self._format_scale_label(bar_value, units),
            text_font_size="10px",
        )
        self._scale_fig.add_layout(self._scale_bar_label)

    def _update_scale_bar(self) -> None:
        """Recompute scale bar value and position for current gain."""
        bar_value = self._compute_scale_bar_value()
        units = self._scale_units()

        y_bottom = float(self._fig.y_range.start) + self.channel_spacing * 0.2  # type: ignore[arg-type]
        y_top = y_bottom + bar_value * self._gain

        self._scale_bar_source.data = {
            "x0": [0.5],
            "y0": [y_bottom],
            "x1": [0.5],
            "y1": [y_top],
        }
        self._scale_bar_label.y = (y_bottom + y_top) / 2
        self._scale_bar_label.text = self._format_scale_label(bar_value, units)

    def _add_envelope_slot(self) -> None:
        """Append a new envelope slot: ``varea`` plus min/max outline lines.

        The outlines guarantee ≥1 px of ink per pixel-column even when the
        min/max spread is too small for the fill to be visible at the
        current zoom (e.g. fully-zoomed-out views).
        """
        i = len(self._envelope_glyphs)
        g = self._fig.varea(
            x="x",
            y1=f"y1_{i}",
            y2=f"y2_{i}",
            source=self._envelope_source,
            fill_alpha=0.7,
            color="black",
        )
        g.visible = False
        self._envelope_glyphs.append(g)
        for col, target in (
            (f"y2_{i}", self._envelope_top_lines),
            (f"y1_{i}", self._envelope_bottom_lines),
        ):
            line = self._fig.line(
                x="x",
                y=col,
                source=self._envelope_source,
                line_width=1.0,
                color="black",
            )
            line.visible = False
            target.append(line)

    def _add_line_slot(self) -> None:
        """Append a new raw-mode line glyph slot."""
        i = len(self._line_glyphs)
        g = self._fig.line(
            x="x",
            y=f"y_{i}",
            source=self._line_source,
            line_width=0.5,
            color="black",
        )
        g.visible = False
        self._line_glyphs.append(g)

    def _set_envelope_visible(self, i: int, visible: bool) -> None:
        """Toggle visibility of the varea and its two outline lines together."""
        self._envelope_glyphs[i].visible = visible
        self._envelope_top_lines[i].visible = visible
        self._envelope_bottom_lines[i].visible = visible

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
                bounds=(
                    float(self._all_offsets[-1] - self.channel_spacing),
                    float(self._all_offsets[0] + self.channel_spacing),
                ),
            ),
            tools="pan,xwheel_zoom,reset",
            active_scroll="xwheel_zoom",
            active_drag="pan",
        )
        self._fig.xaxis.axis_label = "Time (s)"

        # Channel labels on y-axis — use cached channel_ids if available
        ch_ids = self._channel_ids
        self._fig.yaxis.ticker = FixedTicker(ticks=self._offsets.tolist())
        self._fig.yaxis.major_label_overrides = {
            float(self._offsets[i]): (str(ch_ids[self.channels[i]]) if ch_ids is not None else str(self.channels[i]))
            for i in range(len(self.channels))
        }

        # Glyphs allocated on demand — start with initial channel count
        self._max_glyphs = len(self.channels)

        # Envelope source: columns x, y1_0, y2_0, y1_1, y2_1, ...
        env_data: dict[str, list[Any]] = {"x": []}
        for i in range(self._max_glyphs):
            env_data[f"y1_{i}"] = []
            env_data[f"y2_{i}"] = []
        self._envelope_source = ColumnDataSource(data=env_data)

        self._envelope_glyphs: list[Any] = []
        self._envelope_top_lines: list[Any] = []
        self._envelope_bottom_lines: list[Any] = []
        for _ in range(self._max_glyphs):
            self._add_envelope_slot()

        # Line source: columns x, y_0, y_1, ...
        line_data: dict[str, list[Any]] = {"x": []}
        for i in range(self._max_glyphs):
            line_data[f"y_{i}"] = []
        self._line_source = ColumnDataSource(data=line_data)

        self._line_glyphs: list[Any] = []
        for _ in range(self._max_glyphs):
            self._add_line_slot()

        # Register range change callbacks
        self._fig.x_range.on_change("start", self._on_range_change)  # type: ignore[arg-type]
        self._fig.x_range.on_change("end", self._on_range_change)  # type: ignore[arg-type]
        self._fig.y_range.on_change("start", self._on_y_range_change)  # type: ignore[arg-type]
        self._fig.y_range.on_change("end", self._on_y_range_change)  # type: ignore[arg-type]

        # Scale bar figure — shares y_range with main figure
        self._build_scale_bar()

        # Create spike overlay if spike data provided
        self._spike_overlay: SpikeOverlay | None = None
        if self._spike_data is not None:
            self._spike_overlay = SpikeOverlay(
                self._fig,
                self._spike_data,
                self._offsets,
                self.channel_spacing,
                self.channels,
                t_start=self.store.t_start,
            )

    def _initial_render(self) -> None:
        """Render the initial view (full recording)."""
        t_start = self._samples_to_display(0)
        t_end = self._samples_to_display(self.store.n_samples)
        self._update_data(t_start, t_end)

    def _on_y_range_change(self, attr: str, old: Any, new: Any) -> None:
        """Handle Bokeh y_range change events for channel panning."""
        self._schedule_viewport_update()

    def _on_range_change(self, attr: str, old: Any, new: Any) -> None:
        """Handle Bokeh x_range change events."""
        self._schedule_viewport_update()

    def _schedule_viewport_update(self) -> None:
        """Schedule a deferred viewport update, grouping rapid x+y changes.

        During diagonal panning, Bokeh fires 4 separate callbacks (x start,
        x end, y start, y end).  Instead of triggering a fetch for each one,
        we mark the viewport as dirty and schedule a single callback on the
        next tick.  By the time the callback runs, all range properties have
        their final values.
        """
        if self._update_scheduled:
            return  # already have a pending update
        self._update_scheduled = True

        doc = self._fig.document
        if doc is not None:
            doc.add_next_tick_callback(self._deferred_viewport_update)
        else:
            # No document yet (shouldn't happen for range callbacks, but be safe)
            self._deferred_viewport_update()

    def _deferred_viewport_update(self) -> None:
        """Run the actual viewport update after all range changes are batched."""
        self._update_scheduled = False

        x_start = float(self._fig.x_range.start)  # type: ignore[arg-type]
        x_end = float(self._fig.x_range.end)  # type: ignore[arg-type]
        y_start = float(self._fig.y_range.start)  # type: ignore[arg-type]
        y_end = float(self._fig.y_range.end)  # type: ignore[arg-type]

        # Rapid wheel-zoom can transiently leave a range at NaN/inf before
        # Bokeh settles on valid endpoints.  Skip — the next range event
        # will reschedule us with good values.
        if not all(math.isfinite(v) for v in (x_start, x_end, y_start, y_end)):
            log.debug("viewport_update: non-finite range, skipping")
            return

        visible_mask = (self._all_offsets >= y_start) & (self._all_offsets <= y_end)
        visible_indices = np.where(visible_mask)[0]

        channels_changed = False
        if len(visible_indices) > 0:
            new_channels = [self._all_channels[i] for i in visible_indices]
            if new_channels != self.channels:
                channels_changed = True
                log.debug(
                    "viewport_update: y=[%.1f, %.1f], %d channels visible",
                    y_start,
                    y_end,
                    len(new_channels),
                )
                self.channels = new_channels
                self._offsets = self._all_offsets[visible_indices]
                self._last_was_raw = None  # force visibility reset

                # Update y-axis labels
                n_ch = len(self.channels)
                self._fig.yaxis.ticker = FixedTicker(ticks=self._offsets.tolist())  # type: ignore[assignment]
                self._fig.yaxis.major_label_overrides = {  # type: ignore[attr-defined]
                    float(self._offsets[i]): (
                        str(self._channel_ids[self.channels[i]])
                        if self._channel_ids is not None
                        else str(self.channels[i])
                    )
                    for i in range(n_ch)
                }

                # Grow glyph pool if needed
                if n_ch > self._max_glyphs:
                    for _ in range(self._max_glyphs, n_ch):
                        self._add_envelope_slot()
                        self._add_line_slot()
                    self._max_glyphs = n_ch

                # Hide excess glyphs
                for i in range(n_ch, self._max_glyphs):
                    self._set_envelope_visible(i, False)
                    self._line_glyphs[i].visible = False

        # Reposition scale bar to stay at bottom of viewport
        self._update_scale_bar()

        # --- Handle x-range changes (threshold skip) ---
        if not channels_changed:
            log.debug("viewport_update: x=[%.2f, %.2f]", x_start, x_end)
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
        channels = list(self.channels)  # snapshot — channels may change during fetch
        offsets = self._offsets.copy()
        gain = self._gain
        log.debug("request_update: submitting fetch gen=%d t0=%d t1=%d, %d channels", gen, t0, t1, len(channels))

        def _fetch_and_render() -> None:
            import time

            t_start_fetch = time.perf_counter()
            result = self.store.get_window(t0, t1, channels=channels, num_points=self.width)
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
            # Use captured channels/offsets/gain — not self.* which may have changed.
            display = self._prepare_display_data(result, channels=channels, offsets=offsets, gain=gain)
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
                self._last_result = result

                # Apply pre-computed display data (colors are slot-based, never change)
                use_envelope = display["mode"] == "envelope"
                n_fetched = len(channels)
                if self._last_was_raw is not (not use_envelope):
                    for i in range(self._max_glyphs):
                        self._set_envelope_visible(i, use_envelope and i < n_fetched)
                        self._line_glyphs[i].visible = (not use_envelope) and i < n_fetched

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

        channels = list(self.channels)
        result = self.store.get_window(t0, t1, channels=channels, num_points=self.width)
        display = self._prepare_display_data(result, channels=channels, offsets=self._offsets.copy(), gain=self._gain)

        self._last_x_start = x_start
        self._last_x_end = x_end
        self._last_result = result

        use_envelope = display["mode"] == "envelope"
        n_ch = len(channels)
        if self._last_was_raw is not (not use_envelope):
            for i in range(self._max_glyphs):
                self._set_envelope_visible(i, use_envelope and i < n_ch)
                self._line_glyphs[i].visible = (not use_envelope) and i < n_ch

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

        # Update scale bar position
        self._update_scale_bar()

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

    def _pad_source_data(self, data: dict[str, Any], mode: str, n_channels: int) -> dict[str, Any]:
        """Pad a source data dict with NaN columns for unused glyph slots."""
        n_points = len(data["x"])
        nans = np.full(n_points, np.nan)
        if mode == "envelope":
            for i in range(n_channels, self._max_glyphs):
                data[f"y1_{i}"] = nans
                data[f"y2_{i}"] = nans
        else:
            for i in range(n_channels, self._max_glyphs):
                data[f"y_{i}"] = nans
        return data

    def _prepare_display_data(
        self,
        result: WindowResult,
        *,
        channels: list[int],
        offsets: npt.NDArray[Any],
        gain: float,
    ) -> dict[str, Any]:
        """Prepare downsampled display data dict on the fetch thread.

        Uses captured *channels*, *offsets*, and *gain* — not ``self.*``
        which may have changed during the fetch.
        """
        if result.is_raw:
            return self._prepare_raw_display(result, channels=channels, offsets=offsets, gain=gain)
        return self._prepare_envelope_display(result, channels=channels, offsets=offsets, gain=gain)

    def _prepare_envelope_display(
        self,
        result: WindowResult,
        *,
        channels: list[int],
        offsets: npt.NDArray[Any],
        gain: float,
    ) -> dict[str, Any]:
        """Prepare envelope display data, downsampling if needed."""
        stride = result.stride
        assert stride is not None
        data = result.data  # (C, 2, B)
        n_blocks = data.shape[2]
        n_ch = len(channels)

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
        for i in range(n_ch):
            new_data[f"y1_{i}"] = data[i, 0, :] * gain + offsets[i]
            new_data[f"y2_{i}"] = data[i, 1, :] * gain + offsets[i]
        return {"mode": "envelope", "data": self._pad_source_data(new_data, "envelope", n_ch)}

    def _prepare_raw_display(
        self,
        result: WindowResult,
        *,
        channels: list[int],
        offsets: npt.NDArray[Any],
        gain: float,
    ) -> dict[str, Any]:
        """Prepare raw display data, switching to envelope if too many points."""
        data = result.data
        n_samples = data.shape[1]
        n_ch = len(channels)
        max_points = self.width * 2

        if n_samples <= max_points:
            x = np.linspace(
                self._samples_to_display(result.t0_actual),
                self._samples_to_display(result.t1_actual),
                n_samples,
            )
            new_data: dict[str, Any] = {"x": x}
            for i in range(n_ch):
                new_data[f"y_{i}"] = data[i, :] * gain + offsets[i]
            return {"mode": "line", "data": self._pad_source_data(new_data, "line", n_ch)}

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
        for i in range(n_ch):
            new_data[f"y1_{i}"] = mins[i, :] * gain + offsets[i]
            new_data[f"y2_{i}"] = maxs[i, :] * gain + offsets[i]
        return {"mode": "envelope", "data": self._pad_source_data(new_data, "envelope", n_ch)}

    def _rerender_from_cache(self) -> None:
        """Re-prepare and apply display data from the cached result.

        Used when gain changes — the underlying data hasn't changed,
        only the scaling.  Avoids a redundant fetch.
        """
        if self._last_result is None:
            return
        channels = list(self.channels)
        display = self._prepare_display_data(
            self._last_result, channels=channels, offsets=self._offsets.copy(), gain=self._gain
        )
        use_envelope = display["mode"] == "envelope"
        n_ch = len(channels)
        if self._last_was_raw is not (not use_envelope):
            for i in range(self._max_glyphs):
                self._set_envelope_visible(i, use_envelope and i < n_ch)
                self._line_glyphs[i].visible = (not use_envelope) and i < n_ch
        if use_envelope:
            self._envelope_source.data = display["data"]
        else:
            self._line_source.data = display["data"]
        self._last_was_raw = not use_envelope

    def set_channels(self, channels: list[int]) -> None:
        """Change which channels are displayed.

        Grows the glyph pool if needed, then updates visibility, y-axis
        labels, and offsets on the existing figure.

        Parameters
        ----------
        channels : list[int]
            New channel indices.
        """
        # Invalidate any in-flight fetches (they have the wrong channel count)
        self._fetch_generation += 1
        self._pending_viewport = None

        max_ch = min(len(channels), min(64, self.store.n_channels))
        self.channels = list(channels[:max_ch])

        # Grow glyph pool if needed (colors set by _update_glyph_colors below)
        if max_ch > self._max_glyphs:
            for _ in range(self._max_glyphs, max_ch):
                self._add_envelope_slot()
                self._add_line_slot()
            self._max_glyphs = max_ch
        self._offsets = self._compute_channel_offsets()
        self._last_was_raw = None  # force visibility reset on next render

        # Update y-axis labels and range
        n_ch = len(self.channels)
        self._fig.yaxis.ticker = FixedTicker(ticks=self._offsets.tolist())  # type: ignore[assignment]
        self._fig.yaxis.major_label_overrides = {  # type: ignore[attr-defined]
            float(self._offsets[i]): (
                str(self._channel_ids[self.channels[i]]) if self._channel_ids is not None else str(self.channels[i])
            )
            for i in range(n_ch)
        }
        self._fig.y_range.start = float(self._offsets[-1] - self.channel_spacing)  # type: ignore[union-attr]
        self._fig.y_range.end = float(self._offsets[0] + self.channel_spacing)  # type: ignore[union-attr]

        # Hide all glyphs — colors are updated in _apply atomically with data
        for i in range(self._max_glyphs):
            self._set_envelope_visible(i, False)
            self._line_glyphs[i].visible = False

        # Re-render with current x viewport (non-blocking)
        x_start = float(self._fig.x_range.start)  # type: ignore[arg-type]
        x_end = float(self._fig.x_range.end)  # type: ignore[arg-type]
        self._request_update(x_start, x_end)

    def set_gain(self, gain: float) -> None:
        """Change the amplitude gain.

        Re-renders from the cached result without fetching.

        Parameters
        ----------
        gain : float
            Gain multiplier (1.0 = default).
        """
        self._gain = gain
        self._rerender_from_cache()
        self._update_scale_bar()

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
            end=min(64, self.store.n_channels),
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

    def show(self, *, remote: RemoteConfig | None = None, **kwargs: Any) -> None:
        """Launch the viewer in a browser tab (or bind for a remote proxy).

        Parameters
        ----------
        remote : RemoteConfig or None, optional
            When set, configure :func:`panel.serve` for reverse-proxy
            deployment: bind on ``0.0.0.0``, accept the configured
            websocket origin, apply the URL prefix, and do NOT open a
            local browser tab.  The expected URL is logged to the
            module logger before the server starts.  When ``None``
            (default), opens a local browser tab as usual.
        **kwargs
            Additional overrides forwarded to ``pn.serve()``.  Merged
            on top of the remote-derived kwargs (user wins).
        """
        if remote is None:
            pn.serve(self.panel(), show=True, **kwargs)
            return

        serve_kwargs = remote.to_serve_kwargs()
        serve_kwargs.update(kwargs)
        log.info("viewer: serving remotely — expected URL: %s", remote.describe_url())
        pn.serve(self.panel(), **serve_kwargs)


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
        # Sort by (y, x) position. Shallowest (largest y) first = top of display,
        # deepest (smallest y) last = bottom. x as tiebreaker for deterministic
        # ordering of channels at the same depth.
        ch_with_pos = [(ch, float(locations[ch, 1]), float(locations[ch, 0])) for ch in channels if ch < len(locations)]
        ch_with_pos.sort(key=lambda t: (-t[1], t[2]))  # descending y, ascending x
        return [ch for ch, _, _ in ch_with_pos]

    msg = f"Unknown channel_order: {order!r}. Use 'index', 'depth', or a list of channel indices."
    raise ValueError(msg)


def load_phy_spikes(
    phy_dir: str | Path,
    sampling_frequency: float = 30000.0,
    unit_ids: list[int] | None = None,
    t_start: float = 0.0,
) -> tuple[SpikeData, dict[str, list[int]]]:
    """Load spike data from a phy/Kilosort output directory.

    Parameters
    ----------
    phy_dir : str or Path
        Path to phy output directory containing ``spike_times.npy``,
        ``spike_clusters.npy``, ``channel_map.npy``, ``templates.npy``,
        and ``template_ind.npy``.
    sampling_frequency : float, optional
        Sampling rate in Hz, by default 30000.
    unit_ids : list[int] or None, optional
        Subset of unit IDs to load.  ``None`` loads all units.
    t_start : float, optional
        Recording start time in seconds.  Added to spike times so they
        align with the viewer's time axis.  Pass ``store.t_start``.

    Returns
    -------
    spike_data : SpikeData
    template_channels : dict[str, list[int]]
        Mapping of unit ID to active channel indices (channels where
        the template amplitude exceeds 10% of peak).
    """
    phy_path = Path(phy_dir)
    spike_times_samples = np.load(phy_path / "spike_times.npy").flatten()
    spike_clusters = np.load(phy_path / "spike_clusters.npy").flatten()
    channel_map = np.load(phy_path / "channel_map.npy").flatten()
    templates = np.load(phy_path / "templates.npy")
    template_ind = np.load(phy_path / "template_ind.npy")

    # Peak channel per template: channel with largest absolute amplitude
    # argmax gives index into the active channel axis → look up via template_ind → channel_map
    n_cm = len(channel_map)
    peak_active_idx = np.argmax(np.abs(templates).max(axis=1), axis=1)
    peak_channels = np.array(
        [
            channel_map[template_ind[i, peak_active_idx[i]]] if 0 <= template_ind[i, peak_active_idx[i]] < n_cm else 0
            for i in range(len(templates))
        ]
    )

    all_unit_ids = np.unique(spike_clusters)
    if unit_ids is not None:
        all_unit_ids = np.array([u for u in all_unit_ids if u in set(unit_ids)])

    spike_times_dict: dict[str, npt.NDArray[Any]] = {}
    unit_channels: dict[str, int] = {}
    template_channels: dict[str, list[int]] = {}
    for uid in all_unit_ids:
        mask = spike_clusters == uid
        spike_times_dict[str(uid)] = spike_times_samples[mask] / sampling_frequency + t_start
        unit_channels[str(uid)] = int(peak_channels[uid])

        # Active channels: where template amplitude > 10% of peak
        t = templates[uid]  # (n_time, n_active)
        peak_amp = np.abs(t).max(axis=0)
        threshold = peak_amp.max() * 0.1
        active_idx = template_ind[uid][peak_amp > threshold]
        # Filter out -1 padding entries, map through channel_map
        active_chs = [int(channel_map[i]) for i in active_idx if 0 <= i < n_cm]
        template_channels[str(uid)] = sorted(active_chs)

    return SpikeData(spike_times=spike_times_dict, unit_channels=unit_channels), template_channels


def view(
    store_or_path: MipmapStore | str | Path,
    channels: list[int] | None = None,
    n_channels: int = 32,
    channel_order: str | list[int] | None = "depth",
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
