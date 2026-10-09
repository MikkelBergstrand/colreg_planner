"""visualization/multi_panel.py

Static multi-panel trajectory figure for paper use: N trajectory plots side by
side, each a snapshot of the same encounter at a different time step. All panels
share a single y-axis (North); each keeps its own x-axis (East) underneath. Only
one panel carries the legend.

Driven from ``scripts/run_AIS.py`` when ``viz.multi_panel.enabled`` is true (or
``--multi-panel`` is passed) — see that script for how the per-step snapshots are
captured while stepping the predictor forward.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

from colregs_planner.vessel.state import VesselState
from colregs_planner.visualization.live_viz import (
    FONT_SIZE,
    PANEL_BG,
    VESSEL_COLORS,
    SimInfo,
    VizConfig,
    _apply_font_config,
    _draw_colregs_graph,
    _ship_polygon_body,
    _rotate_translate,
)

try:
    from colregs_planner.visualization.map_background import render_chart_background, set_water_facecolor
    _SEACHART_AVAILABLE = True
except ImportError:
    _SEACHART_AVAILABLE = False


# Every piece of text in the figure is drawn at ``FONT_SIZE`` (= viz.font_size)
# times one of these fixed multipliers, so changing the single config value
# rescales the whole figure proportionally.
_FS_AXIS        = 1.0   # axis labels + tick labels (trajectory & responding panels)
_FS_TIMESTAMP   = 1.0   # the "t = ... s" stamp on each trajectory panel
_FS_LEGEND      = 0.8   # line-style legend (trajectory panels / figure bar)
_FS_RESP_LEGEND = 0.8   # responding-panel vessel legend
_FS_RESP_VALUE  = 0.8   # responding-panel p(r=R) value labels at the capture dots
_FS_CG_ROLE     = 0.62  # interaction-graph inset: edge role labels (GIVE-WAY, ...)
_FS_CG_NODE     = 1.0   # interaction-graph inset: node ID labels (V1, V2, ...)

# Line / marker geometry (all further scaled by marker_scale).
_LW_TRAJ        = 2.3   # trajectory line width (trail / prediction / nominal / future)
_LW_RESP        = 2.1   # responding-panel curve width
_HEADING_MARKER = 0.8   # heading-triangle size vs vc.heading_marker_size
_LW_CG_EDGE     = 0.82  # interaction-graph inset edge width vs the live-view default
_LW_LEGEND      = 1.8   # line-sample thickness inside the legend keys
_FUTURE_ALPHA   = 0.45  # ground-truth future line alpha — shared with its legend swatch


@dataclass
class PanelSnapshot:
    """Everything needed to draw one panel — the encounter frozen at ``step``."""

    step:   int
    t:      float                                             # sim time [s]
    states: dict[str, VesselState]                            # vid -> current state
    trails: dict[str, tuple[list[float], list[float]]]        # vid -> (east_hist, north_hist)
    pred:     dict[str, list[VesselState]] = field(default_factory=dict)
    nominal:  dict[str, list[VesselState]] = field(default_factory=dict)
    future:   dict[str, list[VesselState]] = field(default_factory=dict)
    colregs_graph: object | None = None   # ColregsGraph at this step (for the inset)


def _draw_panel(
    ax:            plt.Axes,
    snap:          PanelSnapshot,
    vid_order:     list[str],
    vessel_colors: dict[str, str],
    vc:            VizConfig,
    *,
    future_alpha:  float = _FUTURE_ALPHA,
    scale:         float = 1.0,
) -> None:
    """Draw one trajectory snapshot onto ``ax`` (limits/aspect set by the caller).

    ``scale`` shrinks the on-screen-sized elements — heading triangle, V-label,
    trajectory line widths — together, so a narrow panel doesn't get swamped by
    markers sized for the full-width live map. Positions/extents (data coords)
    are untouched.
    """
    body = _ship_polygon_body(vc.vessel_length, vc.vessel_width, vc.bow_fraction)
    lw = _LW_TRAJ * scale

    for vid in vid_order:
        color = vessel_colors[vid]

        # --- history trail -------------------------------------------------
        e_hist, n_hist = snap.trails.get(vid, ([], []))
        if e_hist:
            ax.plot(e_hist, n_hist, "-", color=color, alpha=vc.trail_alpha,
                    lw=lw, zorder=2)

        # --- future ground-truth trajectory ------------------------------
        # Grey rather than the vessel colour so it doesn't read as the same
        # colour as that vessel's dashed prediction sitting next to it — the
        # history trail (above) stays vessel-coloured for identification.
        fut = snap.future.get(vid, [])
        if fut:
            ax.plot([s.east for s in fut], [s.north for s in fut], "-",
                    color="gray", alpha=future_alpha, lw=lw, zorder=2)

        # --- interaction-aware prediction -------------------------------
        pr = snap.pred.get(vid, [])
        if pr:
            ax.plot([s.east for s in pr], [s.north for s in pr], "--",
                    color=color, alpha=0.75, lw=lw, zorder=3)

        # --- nominal prediction ---------------------------------------
        nm = snap.nominal.get(vid, [])
        if nm:
            ax.plot([s.east for s in nm], [s.north for s in nm], ":",
                    color=color, alpha=0.55, lw=lw, zorder=3)

        # --- current pose ---------------------------------------------
        st = snap.states.get(vid)
        if st is None:
            continue
        e_rot, n_rot = _rotate_translate(body, st.heading, st.north, st.east)
        ax.add_patch(plt.Polygon(np.column_stack([e_rot, n_rot]), closed=True,
                                 facecolor=color, alpha=vc.vessel_alpha,
                                 edgecolor="white", linewidth=1.2 * scale, zorder=4))
        if vc.heading_marker:
            ax.plot([st.east], [st.north],
                    marker=(3, 0, -np.degrees(st.heading)),
                    markersize=vc.heading_marker_size * scale * _HEADING_MARKER,
                    markerfacecolor=color, markeredgecolor="white",
                    markeredgewidth=1.0 * scale, alpha=vc.heading_marker_alpha,
                    linestyle="none", zorder=3)

        # --- V-label -------------------------------------------------
        n_min, n_max = ax.get_ylim()
        lbl_off = (n_max - n_min) * 0.025
        ax.text(st.east, st.north + lbl_off, vid,
                fontsize=vc.vessel_label_fontsize * scale, color=color,
                ha="center", va="bottom", fontweight="bold", fontfamily="serif",
                zorder=6,
                path_effects=[patheffects.withStroke(linewidth=2.5 * scale, foreground="white")])

    ax.text(0.02, 0.97, rf"$t = {snap.t:.1f}\ \mathrm{{s}}$",
            transform=ax.transAxes, fontsize=FONT_SIZE * _FS_TIMESTAMP, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.7), zorder=7)


def _draw_cg_inset(
    ax_host:       plt.Axes,
    graph:         object,
    states:        dict[str, VesselState],
    vid_order:     list[str],
    vessel_colors: dict[str, str],
    vc:            VizConfig,
    t:             float,
    *,
    side:  str   = "bottom-right",
    size:  float = 0.30,
    scale: float = 0.8,
) -> None:
    """Add a corner inset to ``ax_host`` showing the Interaction (COLREGs) graph
    at time ``t`` — same renderer as the live view's ``cg_inset``.

    ``side`` accepts any of ``top-left`` / ``top-right`` / ``bottom-left`` /
    ``bottom-right`` (``left`` / ``right`` are aliases for the bottom corners).
    """
    c    = side.lower().replace("_", "-")
    left = "left" in c
    top  = "top" in c
    x0 = 0.02 if left else (0.98 - size)
    y0 = (0.97 - size) if top else 0.03
    axins = ax_host.inset_axes([x0, y0, size, size])
    axins.set_zorder(6)
    for spine in axins.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(0.8)

    idx_to_id    = {i: vid for i, vid in enumerate(vid_order)}
    idx_to_color = {i: vessel_colors[vid] for i, vid in enumerate(vid_order)}
    # Geometry (dots, line widths, padding) follows `scale`; the text is sized
    # as a fixed multiple of FONT_SIZE so it tracks viz.font_size like the rest
    # of the figure. Role labels are drawn at fs*0.8 inside _draw_colregs_graph,
    # so pre-divide to land on FONT_SIZE * _FS_CG_ROLE.
    _draw_colregs_graph(
        axins, states, graph, idx_to_id, idx_to_color, t, vc,
        scale=scale, idx_to_label=idx_to_id,
        text_scale=_FS_CG_ROLE / 0.8,
        node_label_size=FONT_SIZE * _FS_CG_NODE,
        edge_scale=_LW_CG_EDGE,
    )


def _draw_responding_panel(
    ax:            plt.Axes,
    history:       dict[str, list[float]],
    vessel_colors: dict[str, str],
    vid_order:     list[str],
    *,
    t0:         float,
    dt:         float,
    mark_times: list[float],
    window:      Optional[tuple[float, float]] = None,
    scale:       float = 1.0,
    show_values: bool  = True,
) -> None:
    """Wide p(r=R)-vs-time panel spanning all trajectory columns, with a dashed
    vertical marker at each panel's capture time so the responding posterior can
    be read against the prediction it produced. All text is a fixed multiple of
    the shared ``viz.font_size`` (``FONT_SIZE``), like the trajectory panels.

    ``show_values`` prints the p(r=R) value next to each dot (as the simulation
    responding panel does)."""
    fs = FONT_SIZE * _FS_AXIS

    ax.set_facecolor(PANEL_BG)
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.set_ylabel(r"$p(r = R)$", fontsize=fs)
    ax.set_xlabel(r"$t\;[\mathrm{s}]$", fontsize=fs)
    ax.tick_params(labelsize=fs)

    max_len = max((len(h) for h in history.values()), default=0)
    t_end   = t0 + max(max_len - 1, 0) * dt

    ax.axhline(0.5, color="dimgray", lw=0.8, ls=":", alpha=0.6, zorder=1)

    for vid in vid_order:
        h = history.get(vid) or []
        if not h:
            continue
        t = t0 + np.arange(len(h)) * dt
        ax.plot(t, h, "-", color=vessel_colors[vid], lw=_LW_RESP, zorder=3, label=vid)

    # Capture-time markers — dashed vertical line + a dot on each curve, with the
    # p(r=R) value printed next to it (like the simulation responding panel).
    val_fs   = FONT_SIZE * _FS_RESP_VALUE
    last_tm  = max(mark_times) if mark_times else None
    for tm in mark_times:
        ax.axvline(tm, color="0.35", lw=1.4, ls="--", alpha=0.85, zorder=2)
        idx = int(round((tm - t0) / dt))
        # value label to the left of the rightmost marker, right of the others
        left_of = (tm == last_tm and len(mark_times) > 1)
        dx      = -6 if left_of else 6
        ha      = "right" if left_of else "left"

        vals = [(vid, history[vid][idx]) for vid in vid_order
                if history.get(vid) and 0 <= idx < len(history[vid])]

        # dots on every curve ...
        for vid, p in vals:
            ax.plot(tm, p, "o", color=vessel_colors[vid], ms=6.5,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=4)
        if not show_values:
            continue

        # ... but only one value label per distinct (rounded) value — when
        # curves overlap, the last vessel (in V-order) at that value gets it.
        by_val: dict[float, tuple[str, float]] = {}
        for vid, p in vals:
            by_val[round(p, 2)] = (vid, p)
        label_pts = sorted(by_val.values(), key=lambda kv: kv[1], reverse=True)
        for rank, (vid, p) in enumerate(label_pts):
            if p > 0.92:
                dy, va = -9, "center"        # below a near-1.0 dot
            elif p < 0.08:
                dy, va = 9, "center"         # above a near-0 dot
            else:
                dy, va = 0, "center"
            ax.annotate(
                f"{p:.2f}", (tm, p), textcoords="offset points",
                xytext=(dx, dy), ha=ha, va=va,
                fontsize=val_fs, color=vessel_colors[vid], fontweight="bold",
                zorder=6, clip_on=False,
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
            )

    # x-limits (set last so plotting can't autoscale them back out) — the window,
    # or the span from the first sample to the last capture time.
    if window is not None:
        lo, hi = float(window[0]), float(window[1])
    else:
        lo = min([t0] + list(mark_times))
        hi = max([t_end] + list(mark_times))
    ax.set_xlim(lo, hi)
    ax.set_autoscalex_on(False)

    # x-ticks only on the capture times — they line up with the trajectory
    # panels above; no ticks at the axis ends.
    ticks = sorted({round(float(t), 3) for t in mark_times if lo - 1e-6 <= t <= hi + 1e-6})
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.0f}" for t in ticks])

    if any(history.values()):
        ax.legend(fontsize=FONT_SIZE * _FS_RESP_LEGEND, loc="lower left",
                  ncol=max(1, len(vid_order)), framealpha=0.85,
                  handlelength=1.6, columnspacing=1.2, borderpad=0.4)


def _legend_handles(
    *, show_future: bool, show_pred: bool, show_nominal: bool, is_ais: bool,
) -> list[Line2D]:
    # With pred_top_k == 1 there's only ever one hypothesis, so it's named
    # consistently with the multi-hypothesis "Top-k" entries elsewhere.
    pred_label = "Top-1 prediction" if is_ais else "Prediction"
    lw = _LW_LEGEND
    handles = [
        Line2D([0], [0], ls="-", color="gray", lw=lw,
               label="Observed" if show_future else "Trajectory")
    ]
    if show_future:
        handles.append(Line2D([0], [0], ls="-", color="gray", lw=lw, alpha=_FUTURE_ALPHA,
                              label="Ground Truth"))
    if show_pred:
        handles.append(Line2D([0], [0], ls="--", color="gray", lw=lw, label=pred_label))
    if show_nominal:
        handles.append(Line2D([0], [0], ls=":", color="gray", lw=lw,
                              label="Nominal prediction"))
    return handles


def save_multi_panel_trajectory_plot(
    *,
    snapshots:     list[PanelSnapshot],
    sim_info:      SimInfo,
    vc:            VizConfig,
    xlim:          tuple[float, float],
    ylim:          tuple[float, float],
    chart_bg=None,
    water_color:   Optional[str] = None,
    legend_panel:  int   = 0,
    legend_loc:    str   = "panel",   # "panel" = corner of legend_panel;
                                      # "bottom" = one horizontal bar under all panels;
                                      # "panel-bottom" = horizontal bar inside legend_panel
    legend_ncol:   Optional[int] = None,
    panel_width:   float = 4.0,
    marker_scale:  float = 1.0,
    show_pred:     bool  = True,
    show_nominal:  bool  = True,
    show_future:   bool  = True,
    is_ais:        bool  = True,
    cg_inset:       bool  = False,
    cg_inset_side:  str   = "right",
    cg_inset_size:  float = 0.30,
    cg_inset_scale: float = 0.8,
    responding_panel:   str = "off",   # "over" | "under" | "off" — wide p(r=R) panel
    responding_history: Optional[dict[str, list[float]]] = None,
    responding_t0:      float = 0.0,
    responding_dt:      float = 1.0,
    responding_window:  Optional[tuple[float, float]] = None,
    responding_ratio:   float = 0.25,
    responding_values:  bool  = True,
    panel_gap:          float = 0.012,   # gap between packed panels (figure fraction)
    frames_dir:    Optional[str] = None,
    frame_format:  str   = "pdf",
    frame_dpi:     int   = 300,
    sim_name:      str   = "multi_panel",
) -> Optional[Path]:
    """Render the N-panel figure and save it. Returns the saved path."""
    if not snapshots:
        print("  multi_panel: no snapshots to plot.")
        return None

    # Sync this module's FONT_SIZE to viz.font_size (the name imported at module
    # load is only the default). All figure text is then FONT_SIZE * a fixed
    # _FS_* multiplier, so the single config value rescales everything.
    global FONT_SIZE
    _apply_font_config(vc)
    FONT_SIZE = float(vc.font_size)

    vid_order = [m.label for m in sim_info.vessels]
    vessel_colors = {
        m.label: VESSEL_COLORS[i % len(VESSEL_COLORS)]
        for i, m in enumerate(sim_info.vessels)
    }

    legend_loc = legend_loc.lower().replace("_", "-")
    bottom_legend      = legend_loc in ("bottom", "bar", "figure")
    panel_bottom_legend = legend_loc in ("panel-bottom", "panel-bar", "inside-bottom")

    resp_pos = responding_panel.lower()
    resp_on  = (
        resp_pos in ("over", "under", "top", "bottom")
        and bool(responding_history) and any(responding_history.values())
    )
    resp_over = resp_pos in ("over", "top")

    n = len(snapshots)
    e_range = xlim[1] - xlim[0]
    n_range = ylim[1] - ylim[0]
    panel_h = panel_width * (n_range / e_range) if e_range > 0 else panel_width
    legend_h = 0.55 if bottom_legend else 0.0   # [in] reserved strip under the panels
    traj_row_h = panel_h + 0.85                  # trajectory panels + their x-label
    resp_h = max(0.95, panel_h * responding_ratio) if resp_on else 0.0

    fig = plt.figure(figsize=(panel_width * n, traj_row_h + resp_h + legend_h))

    if resp_on:
        hr = [resp_h, traj_row_h] if resp_over else [traj_row_h, resp_h]
        gs = fig.add_gridspec(2, n, height_ratios=hr, hspace=0.06)
        traj_r = 1 if resp_over else 0
        resp_r = 0 if resp_over else 1
        resp_ax = fig.add_subplot(gs[resp_r, :])
    else:
        gs = fig.add_gridspec(1, n)
        traj_r = 0
        resp_ax = None

    axes = []
    for j in range(n):
        ax = fig.add_subplot(gs[traj_r, j], sharey=axes[0] if axes else None)
        if j > 0:
            ax.tick_params(labelleft=False)
        axes.append(ax)

    legend_panel = max(0, min(legend_panel, n - 1))
    legend_handles = _legend_handles(
        show_future=show_future, show_pred=show_pred,
        show_nominal=show_nominal, is_ais=is_ais,
    )

    for i, (ax, snap) in enumerate(zip(axes, snapshots)):
        ax.set_facecolor(PANEL_BG)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.tick_params(labelsize=FONT_SIZE * _FS_AXIS)
        ax.set_xlabel(r"East $[\mathrm{m}]$", fontsize=FONT_SIZE * _FS_AXIS)
        if i == 0:
            ax.set_ylabel(r"North $[\mathrm{m}]$", fontsize=FONT_SIZE * _FS_AXIS)

        if chart_bg is not None and _SEACHART_AVAILABLE and not chart_bg.is_empty():
            if water_color:
                set_water_facecolor(ax, water_color)
            render_chart_background(ax, chart_bg)

        _draw_panel(ax, snap, vid_order, vessel_colors, vc, scale=marker_scale)

        if i == legend_panel and not bottom_legend:
            if panel_bottom_legend:
                # Horizontal bar along the bottom of the panel. Default to 2
                # columns (a compact 2x2 block) so it fits a narrow packed
                # panel; set legend_ncol: <n entries> for a single row.
                ncol = legend_ncol or min(2, len(legend_handles))
                ax.legend(
                    handles=legend_handles, loc="lower center",
                    bbox_to_anchor=(0.5, 0.02), ncol=ncol,
                    frameon=True, fancybox=False, edgecolor="0.5",
                    framealpha=0.85, fontsize=FONT_SIZE * _FS_LEGEND,
                    columnspacing=1.2, handlelength=1.8, handletextpad=0.6,
                    borderpad=0.5, labelspacing=0.35, borderaxespad=0.0,
                )
            else:
                ax.legend(handles=legend_handles, loc="upper right",
                          fontsize=FONT_SIZE * _FS_LEGEND)

        if cg_inset and snap.colregs_graph is not None:
            _draw_cg_inset(
                ax, snap.colregs_graph, snap.states, vid_order, vessel_colors,
                vc, snap.t, side=cg_inset_side, size=cg_inset_size,
                scale=cg_inset_scale,
            )

    if resp_ax is not None:
        _draw_responding_panel(
            resp_ax, responding_history, vessel_colors, vid_order,
            t0=responding_t0, dt=responding_dt,
            mark_times=[s.t for s in snapshots],
            window=responding_window, scale=marker_scale,
            show_values=responding_values,
        )

    fig_h = traj_row_h + resp_h + legend_h
    tl_kwargs = dict(h_pad=1.0, w_pad=0.3) if resp_on else {}
    legend_rect = (0.0, legend_h / fig_h, 1.0, 1.0)
    if bottom_legend:
        fig.tight_layout(rect=legend_rect, **tl_kwargs)
    else:
        fig.tight_layout(**tl_kwargs)

    # With aspect="equal" the trajectory panels end up height-constrained, so
    # each is narrower than its gridspec column and gaps open between them. Pack
    # them flush (equal width, hair-thin gap) and make the responding panel span
    # exactly that block instead of the full figure width.
    block_x0 = block_w = None
    if resp_on:
        fig.canvas.draw()
        pos = [a.get_position() for a in axes]
        pw  = min(p.width for p in pos)
        ph  = pos[0].height
        y0  = pos[0].y0
        gap = max(0.0, panel_gap)
        block_x0 = min(p.x0 for p in pos)
        block_w  = n * pw + (n - 1) * gap
        for j, a in enumerate(axes):
            a.set_position([block_x0 + j * (pw + gap), y0, pw, ph])
        if resp_ax is not None:
            rb = resp_ax.get_position()
            resp_ax.set_position([block_x0, rb.y0, block_w, rb.height])

    if bottom_legend:
        lx = (block_x0 + block_w / 2) if block_x0 is not None else 0.5
        fig.legend(
            handles=legend_handles, loc="lower center",
            bbox_to_anchor=(lx, 0.0), ncol=legend_ncol or len(legend_handles),
            frameon=True, fancybox=False, edgecolor="0.4",
            fontsize=FONT_SIZE * _FS_LEGEND, borderaxespad=0.0,
            columnspacing=1.6, handlelength=2.4,
        )

    out_dir = (
        Path(frames_dir) if frames_dir
        else Path(__file__).resolve().parents[2] / "outputs" / "frames"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    slug      = re.sub(r"[^\w\-]+", "_", sim_name).strip("_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fmt       = frame_format.lower()
    path      = out_dir / f"{slug}_multipanel_{timestamp}.{fmt}"
    save_kwargs = {"dpi": frame_dpi} if fmt == "png" else {}
    fig.savefig(path, bbox_inches="tight", **save_kwargs)
    plt.close(fig)
    print(f"Saved multi-panel trajectory plot -> {path}")
    return path
