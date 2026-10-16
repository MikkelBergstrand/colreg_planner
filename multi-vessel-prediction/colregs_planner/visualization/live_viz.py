"""visualization/live_viz.py

Live scenario visualizer with optional velocity/action-space debug panels.

Main panel : NED position map (vessels + trails).
Debug panels: per-vessel velocity space (VO) or action space (CTO).

Keyboard
--------
  Space      : pause / resume
  Q / Escape : quit
  S          : save current frame (outputs/frames/, or VizConfig.frames_dir; format via
               VizConfig.frame_format — "pdf"/"svg"/"png", default "pdf")
  Ctrl-C     : quit (terminal fallback)

Config (viz section)
--------------------
  debug_viz: false   # show velocity/action-space panel (default false)

Author: Johannes Robert Skarø
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patheffects as patheffects
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import offset_copy

from colregs_planner.vessel.state import VesselState
from colregs_planner.policies.debug_types import CTODebugFrame, VODebugFrame, PolicyDebugFrame, StarboardConstraintDebugData
from colregs_planner.colregs_graph.colregs_graph import ColregsGraph, VesselRole


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------
#
# One uniform font size is used for every label, tick, legend entry, and
# title across the main map and all debug/side panels, so a saved frame
# reads consistently at print size. Only the vessel-ID markers (V1, V2, ...)
# on the main map use a larger size, set separately, since they double as
# a spatial pointer to "where the vessel actually is" and need to stand out.
# Both are configurable via VizConfig.font_size / vessel_label_fontsize
# (see _apply_font_config, called once per LiveVisualizer instance).

FONT_SIZE = 13.0

# Shared panel background — kept identical across the main map and the
# interaction graph so the two read as one figure.
PANEL_BG = "#f8fafd"

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.labelsize":   FONT_SIZE,
    "xtick.labelsize":  FONT_SIZE,
    "ytick.labelsize":  FONT_SIZE,
    "legend.fontsize":  FONT_SIZE,
    "axes.titlesize":   FONT_SIZE,
    # Embed real (Type 42/TrueType) outlines in saved PDF/PS figures instead
    # of matplotlib's default Type 3 bitmap fonts — Type 3 fails PDF/A and
    # accessibility checkers ("Type 3 fonts are rasterized... Not allowed").
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
})


def _apply_font_config(vc: "VizConfig") -> None:
    """Sync the module-level font sizes (and matplotlib rcParams) to VizConfig.

    Called once when a LiveVisualizer is constructed so every panel — main
    map, VO/CTO debug, COLREGs graph, responding — draws text at the same
    configured size.
    """
    global FONT_SIZE
    FONT_SIZE = vc.font_size
    plt.rcParams.update({
        "axes.labelsize":  FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "axes.titlesize":  FONT_SIZE,
    })


VESSEL_COLORS = ["#2962ff", "#ff4d3d", "#00a854", "#ff9500", "#8b3dff"]  # vivid blue/red/green/orange/purple

# Interaction-graph edges: a nominal edge is drawn in its encounter-type
# colour (the colour of its "active", non-stand-on role — see _cg_edge_color),
# solid. An edge touching a non-responsive vessel — its COLREGs role is then
# overridden — drops to a neutral slate dashed line, since the role no longer
# applies. Role colours (_ROLE_COLOR) are kept clear of the vessel palette so
# this no longer competes with the node colours.
CG_EDGE_COLOR    = "#5b6470"        # neutral slate — overridden edges
CG_OVERRIDDEN_LS = (0, (5, 3))     # dashed — edge whose COLREGs role is overridden

_ROLE_LABEL: dict[VesselRole, str] = {
    VesselRole.GIVE_WAY:        "GIVE-WAY",
    VesselRole.STAND_ON:        "STAND-ON",
    VesselRole.HEAD_ON:         "HEAD-ON",
    VesselRole.OVERTAKING:      "OVERTAKING",
    VesselRole.AVOID_COLLISION: "AVOID-COLL",
}

# Role-label text colours — a mid-tone palette kept clear of every hue in
# VESSEL_COLORS (blue / red / green / orange / purple) so a role label is
# never confused with a vessel's own colour.
_ROLE_COLOR: dict[VesselRole, str] = {
    VesselRole.GIVE_WAY:        "#b45309",   # bronze / amber
    VesselRole.STAND_ON:        "#6b7280",   # grey (passive role)
    VesselRole.HEAD_ON:         "#0891b2",   # cyan
    VesselRole.OVERTAKING:      "mediumpurple",
    VesselRole.AVOID_COLLISION: "#1f2937",   # near-black (imminent — everyone manoeuvres)
}


def _cg_edge_color(role_a: VesselRole, role_b: VesselRole) -> str:
    """Edge colour = the colour of the 'active' (non-stand-on) role: a crossing
    edge takes the give-way colour, an overtaking edge the overtaking colour,
    a head-on edge the head-on colour."""
    for r in (role_a, role_b):
        if r != VesselRole.STAND_ON:
            return _ROLE_COLOR.get(r, CG_EDGE_COLOR)
    return _ROLE_COLOR[VesselRole.STAND_ON]


# ---------------------------------------------------------------------------
# Metadata / config
# ---------------------------------------------------------------------------

def _coerce_off_string(v) -> str:
    """Normalize a string-valued config option that accepts "off" as one
    of its choices (legend_loc, cg_monochrome, ...).

    YAML 1.1 (what PyYAML loads) parses an unquoted `off` as the boolean
    False, not the string "off" — same trap as `on`/`yes`/`no`. Left
    un-normalized, `str(False)` becomes "False", which doesn't match any
    of these options' recognized values and silently falls through to
    the default (on) behavior instead of turning the feature off.
    """
    if v is False:
        return "off"
    return str(v)


@dataclass
class VesselMeta:
    vessel_id:     str
    initial_state: VesselState
    display_id:    str = ""   # label shown on the map / legend / graph (e.g. "V1").
                              # Falls back to vessel_id when empty. Internal dict keys
                              # (trajectories, predictions, ...) always stay vessel_id.

    @property
    def label(self) -> str:
        return self.display_id or self.vessel_id


@dataclass
class SimInfo:
    name:    str
    n_steps: int
    dt:      float
    vessels: list[VesselMeta] = field(default_factory=list)


@dataclass
class PlanOverlay:
    """A vessel's planned track, drawn dash-dot with labelled waypoints.

    north, east : the planned track [m]
    marks       : (north, east, label) points along it, e.g. where each
                  manoeuvre of the plan starts
    """

    north: list[float]
    east:  list[float]
    marks: list[tuple[float, float, str]] = field(default_factory=list)


@dataclass
class VizConfig:
    figsize:      tuple[float, float] = (12, 10)
    padding:      float               = 100.0
    xlim:         Optional[tuple[float, float]] = None  # explicit NED east range [m];
    ylim:         Optional[tuple[float, float]] = None  # set with xlim for a manual view
                                                       # window (overrides padding/auto extent)
    trail_length: int                 = 80
    trail_alpha:  float               = 0.65
    realtime_factor: float            = 100.0
    save_path:    Optional[str]       = None
    fps:          int                 = 20
    debug_viz:          bool  = False  # show velocity/action-space panel
    colregs_graph_viz:  bool  = True   # show COLREGs role graph in separate figure
    responding_panel:   bool  = False  # show responsiveness posterior time-series panel

    video_enabled: bool = False  # record live playback to outputs/videos/
    video_fps:     int  = 15     # frame rate of the recorded video (independent of on-screen playback speed)
    video_start_time_s: Optional[float] = None  # restrict the recorded video to this sim-time
    video_end_time_s:   Optional[float] = None  # window [s] (t = frame * dt, from the start of
        # the window — AIS playback only). Live playback is unaffected — it still plays the full
        # requested range; only which frames get grabbed into the video is trimmed. None on either
        # side = unrestricted on that side (record from/to wherever playback itself starts/ends).

    vessel_length: float = 40.0
    vessel_width:  float = 12.0
    bow_fraction:  float = 0.25
    vessel_alpha:  float = 0.85
    label_offset:  float = 2.0

    # Fixed-screen-size directional marker drawn at each vessel position (a
    # triangle pointing along heading). The true hull polygon is invisible at
    # AIS/basin scale, so this keeps heading readable regardless of zoom.
    heading_marker:      bool  = False  # opt-in (enabled for AIS playback); see scenarios/real/window_098.yaml
    heading_marker_size: float = 24.0   # marker size [pt]
    heading_marker_alpha: float = 0.9

    font_size:              float = 13.0  # uniform size for labels/ticks/legends/titles everywhere
    vessel_label_fontsize:  float = 17.0  # V1, V2, ... markers on the main map (larger, for spatial clarity)

    frames_dir: Optional[str] = None  # output dir for 's'-key snapshots (default: outputs/frames)
    frame_format: str = "pdf"  # "pdf" | "svg" | "png" — pdf/svg are vector; png is rasterized by
                                # the same Agg renderer used on screen, so it's guaranteed to match
                                # the live view exactly (pdf viewers can composite alpha differently)
    frame_dpi: int = 300  # only used when frame_format == "png"
    goal_markers: bool = False  # scenario plots: draw each vessel's goal as a single marker
                                 # (projected endpoint of its nominal track) instead of the
                                 # full dotted nominal trajectory — less clutter for paper figures.
    center_ego_x: bool = False  # scenario plots: shift the auto-computed x-limits so the ego
                                 # vessel's initial position sits at the horizontal centre of the
                                 # plot (view width is still derived from the vessel trajectories).
    stop_time_s: Optional[float] = None  # [s] auto-pause + save a frame once playback reaches
                                          # this sim time (t = step * dt). None = never auto-pause.
                                          # CLI --stop-time overrides this.
    cg_inset: bool = False  # show the Interaction Graph as a corner inset of the main map
                             # instead of its own side-by-side column (live view too, not just save)
    cg_inset_side: str = "right"  # "top-left" | "top-right" | "bottom-left" | "bottom-right"
                             # — which corner the inset sits in ("left"/"right" are aliases
                             # for the bottom corners, kept for backward compatibility)
    cg_inset_size: float = 0.36  # inset box width as a fraction of the main map; height is
                             # derived so the inset renders square in inches regardless of
                             # the trajectory plot's xlim/ylim aspect ratio
    cg_legend: bool = True  # show the role-colour legend (Head-on / Crossing / …) on the
                             # Interaction Graph panel
    cg_monochrome: str = "black-grey"  # "off"        = colour-coded edges + role labels
                                 # "black"      = all edges/role labels near-black
                                 # "grey"       = all edges/role labels mid-grey
                                 # "black-grey" = near-black role labels, mid-grey edges
                                 # (nominal vs overridden edges still differ by solid/dashed)

    responding_inline: bool = False  # draw each target vessel's current p(responding) as a small
        # horizontal belief bar (green→red) next to its V-label on the main map. Needs a responding
        # history (prediction.use_responding_filter: true); independent
        # of responding_panel. Shows the current value only — pair with responding_plot / the panel
        # for the time evolution.

    responding_plot: bool = False  # save a standalone p(responding)-vs-time plot once playback
        # ends (requires prediction.interaction.use_responding_filter: true
        # in config, so there's a responding history to plot at all). Independent of
        # responding_panel — this doesn't need the live side panel turned on.
    responding_plot_window: Optional[tuple[float, float]] = None  # [t_start, t_end] in seconds to
        # crop the standalone plot's time axis to; None = full run

    responding_under_graph: bool = False  # scenario live view: stack a live p(responding)-vs-time
        # panel directly beneath the Interaction Graph, in the same right column. The graph is
        # shortened to make room so the two together span exactly the trajectory plot's height.
        # Needs a responding history and colregs_graph_viz: true (and not cg_inset). Independent
        # of responding_panel.
    responding_under_graph_ratio: float = 0.45  # height of the responding panel relative to the
        # Interaction Graph above it (0.45 → graph gets ~69% of the column, responding ~31%)

    legend_loc: str = "corner"  # main-map line-style legend: "corner" = boxed legend in the
        # upper-right of the map; "top" = a horizontal legend bar above the map. "off" hides it.

    panel_titles: bool = True   # show a heading above each panel: "Trajectory Plot" over the
        # main map, "Top-1 Interaction Graph" over the COLREGs graph panel (skipped when it's
        # folded into the main map as a corner inset via cg_inset), "Responding Posterior" over
        # the standalone responding panel (skipped on the compact responding_under_graph panel —
        # too cramped alongside the graph above it).

    @classmethod
    def from_dict(cls, d: dict) -> "VizConfig":
        v = d.get("vessel_size", d.get("vessel", {}))
        video = d.get("video", {})
        # accept both "debug_viz" and the legacy "show_action_space" key
        debug_viz = d.get("debug_viz", d.get("show_action_space", False))
        return cls(
            figsize       = tuple(d.get("figsize",        (12, 10))),
            padding       = d.get("padding",              100.0),
            xlim          = (tuple(d["xlim"]) if d.get("xlim") is not None else None),
            ylim          = (tuple(d["ylim"]) if d.get("ylim") is not None else None),
            trail_length  = d.get("trail_length",         80),
            trail_alpha   = d.get("trail_alpha",          0.65),
            realtime_factor = d.get("realtime_factor",  100.0),
            save_path     = d.get("save_path",            None),
            fps           = d.get("fps",                  20),
            debug_viz          = debug_viz,
            colregs_graph_viz  = d.get("colregs_graph_viz", True),
            responding_panel   = d.get("responding_panel",  False),
            video_enabled      = video.get("enabled",       False),
            video_fps          = video.get("fps",            15),
            video_start_time_s = video.get("start_time_s",   None),
            video_end_time_s   = video.get("end_time_s",     None),
            vessel_length      = v.get("length",               40.0),
            vessel_width  = v.get("width",                12.0),
            bow_fraction  = v.get("bow_fraction",         0.25),
            vessel_alpha  = v.get("alpha",                0.85),
            label_offset  = v.get("label_offset",         2.0),
            heading_marker       = v.get("heading_marker",       False),
            heading_marker_size  = v.get("heading_marker_size",  24.0),
            heading_marker_alpha = v.get("heading_marker_alpha", 0.9),
            font_size             = d.get("font_size",             13.0),
            vessel_label_fontsize = d.get("vessel_label_fontsize", 17.0),
            frames_dir             = d.get("frames_dir",            None),
            frame_format            = d.get("frame_format",         "pdf"),
            frame_dpi               = d.get("frame_dpi",            300),
            stop_time_s             = d.get("stop_time_s",           None),
            goal_markers            = d.get("goal_markers",          False),
            center_ego_x            = d.get("center_ego_x",          False),
            cg_inset                = d.get("cg_inset",              False),
            cg_inset_side           = d.get("cg_inset_side",         "right"),
            cg_inset_size           = float(d.get("cg_inset_size",   0.36)),
            cg_legend               = d.get("cg_legend",             True),
            cg_monochrome           = _coerce_off_string(d.get("cg_monochrome", "black-grey")),
            responding_inline       = d.get("responding_inline",     False),
            responding_plot         = d.get("responding_plot",       False),
            responding_plot_window  = (
                tuple(d["responding_plot_window"]) if d.get("responding_plot_window") else None
            ),
            responding_under_graph       = d.get("responding_under_graph",       False),
            responding_under_graph_ratio = d.get("responding_under_graph_ratio", 0.45),
            legend_loc                   = _coerce_off_string(d.get("legend_loc", "corner")),
            panel_titles                 = d.get("panel_titles",                 True),
        )

    @classmethod
    def default(cls) -> "VizConfig":
        return cls()


# ---------------------------------------------------------------------------
# Ship polygon geometry
# ---------------------------------------------------------------------------

def _merge_plan_marks(
    marks: list[tuple[float, float, str]], ax, min_sep_pt: float,
) -> list[tuple[float, float, str]]:
    """Merge consecutive plan marks closer than *min_sep_pt* on screen into
    one mark at the first, so a stop or slow (whose track barely moves)
    doesn't stack its labels. The merged label reads as a sequence, with
    repeats of an action counted: "stop ×2 (t=40 s) → nominal". An action's
    name is the label's first word; anything after it is kept once."""
    if not marks:
        return []
    px = ax.transData.transform([(e, n) for n, e, _ in marks])
    min_sep_px = min_sep_pt * ax.figure.dpi / 72.0
    groups: list[list[int]] = [[0]]
    for i in range(1, len(marks)):
        if np.hypot(*(px[i] - px[groups[-1][0]])) < min_sep_px:
            groups[-1].append(i)
        else:
            groups.append([i])

    merged = []
    for g in groups:
        runs: list[list[str]] = []           # [name, suffix, count]
        for i in g:
            name, _, suffix = marks[i][2].partition(" ")
            if runs and runs[-1][0] == name:
                runs[-1][2] += 1
            else:
                runs.append([name, suffix, 1])
        label = " → ".join(
            name + (f" ×{count}" if count > 1 else "") + (f" {suffix}" if suffix else "")
            for name, suffix, count in runs
        )
        n, e, _ = marks[g[0]]
        merged.append((n, e, label))
    return merged


def _ship_polygon_body(length: float, width: float, bow_fraction: float) -> np.ndarray:
    L  = length
    W  = width
    Lb = bow_fraction * L
    return np.array([
        [-L / 2,       -W / 2],
        [-L / 2,        W / 2],
        [ L / 2 - Lb,  W / 2],
        [ L / 2,        0.0  ],
        [ L / 2 - Lb, -W / 2],
        [-L / 2,       -W / 2],
    ])


def _rotate_translate(
    poly: np.ndarray, heading: float, north: float, east: float,
) -> tuple[np.ndarray, np.ndarray]:
    c, s  = np.cos(heading), np.sin(heading)
    n_rot = c * poly[:, 0] - s * poly[:, 1] + north
    e_rot = s * poly[:, 0] + c * poly[:, 1] + east
    return e_rot, n_rot


def _draw_vessel_marker(
    ax:        plt.Axes,
    state:     VesselState,
    color:     str,
    vc:        "VizConfig",
    vessel_id: str,
    scale:     float = 1.0,
    *,
    label_fontsize: Optional[float] = None,
) -> None:
    """Plain position marker (a point, not the ship-hull polygon used on the
    main map — heading/shape isn't the point in a role graph) with the same
    larger, halo'd V1/V2 label styling as the main map. For panels that
    redraw from scratch every frame (e.g. the COLREGs graph) rather than
    keeping a persistent artist like VesselArtist does for the main map.

    `scale` shrinks both the marker and label together (e.g. for a small
    corner-inset copy of this panel) without touching global font state.
    `label_fontsize`, when given, sets the ID label size directly instead of
    `vc.vessel_label_fontsize * scale` (so text can be sized independently of
    the marker geometry).
    """
    ax.scatter(state.east, state.north, s=210 * scale**2, color=color,
               edgecolors="white", linewidths=1.6 * scale, zorder=5)

    n_min, n_max = ax.get_ylim()
    lbl_off = (n_max - n_min) * 0.025
    ax.text(
        state.east, state.north + lbl_off, vessel_id,
        fontsize=(label_fontsize if label_fontsize is not None
                  else vc.vessel_label_fontsize * scale),
        color=color,
        ha="center", va="bottom", fontweight="bold", fontfamily="serif",
        zorder=6,
        path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
    )


# ---------------------------------------------------------------------------
# Debug panel helpers
# ---------------------------------------------------------------------------



def _draw_cto_debug(
    ax: plt.Axes,
    frame: CTODebugFrame,
    color: str,
    vessel_id: str,
) -> None:
    """Render CTO action-space snapshot.

    Shows three-step chain: u_nom → u_projected (on ellipse boundary) →
    u_selected (after ω_max / speed clamping).  When clamping is active,
    u_selected is inside the ellipse and drawn in a distinct style.
    """
    ax.cla()
    theta = np.linspace(0, 2 * np.pi, 300)

    ax.axvline(0, color="lightgray", lw=0.6, zorder=0)
    ax.axhline(0, color="lightgray", lw=0.6, zorder=0)

    # --- Clearance ellipses — axes: x = u2 (v·ω), y = u1 (speed) -----------
    for i, ell in enumerate(frame.ellipses):
        ex = ell.c2 + ell.a2 * np.cos(theta)   # x = u2
        ey = ell.c1 + ell.a1 * np.sin(theta)   # y = u1
        lbl = "clearance ellipse" if i == 0 else None
        ax.fill(ex, ey, alpha=0.08, color="salmon", zorder=1)
        ax.plot(ex, ey, "-", color="tomato", lw=1.8, alpha=0.9, zorder=2,
                label=lbl)

    u_nom  = frame.u_nom
    u_proj = frame.u_projected
    u_sel  = frame.u_selected

    # Detect states
    evasion_needed = (
        abs(u_proj[0] - u_nom[0]) > 1e-6 or
        abs(u_proj[1] - u_nom[1]) > 1e-6
    )
    clamped = evasion_needed and (
        abs(u_sel[0] - u_proj[0]) > 1e-6 or
        abs(u_sel[1] - u_proj[1]) > 1e-6
    )

    # --- u_nom (star) — plot as (u2, u1) ------------------------------------
    ax.plot(u_nom[1], u_nom[0], "k*", ms=10, zorder=6,
            label=r"$\mathbf{u}_\mathrm{nom}$")

    if evasion_needed:
        # Arrow: u_nom → u_projected (on boundary)
        ax.annotate("", xy=(u_proj[1], u_proj[0]), xytext=(u_nom[1], u_nom[0]),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.4),
                    zorder=4)
        # u_projected marker (on ellipse boundary)
        ax.plot(u_proj[1], u_proj[0], "o", color="steelblue", ms=7,
                markeredgecolor="white", markeredgewidth=0.8, zorder=7,
                label=r"$\mathbf{u}_\mathrm{proj}$ (boundary)")

        if clamped:
            # Arrow: u_projected → u_selected (physical limit)
            ax.annotate("", xy=(u_sel[1], u_sel[0]), xytext=(u_proj[1], u_proj[0]),
                        arrowprops=dict(arrowstyle="->", color="tomato",
                                        lw=1.4, linestyle="dashed"),
                        zorder=4)
            # u_selected marker (inside ellipse — clamped)
            ax.plot(u_sel[1], u_sel[0], "x", color="tomato", ms=9,
                    markeredgewidth=2.0, zorder=8,
                    label=r"$\mathbf{u}_\mathrm{sel}$ (clamped)")
        else:
            # No clamping: u_selected coincides with u_projected
            ax.plot(u_sel[1], u_sel[0], "^", color="seagreen", ms=8,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=8,
                    label=r"$\mathbf{u}_\mathrm{sel}$")
    else:
        # u_nom is safe — no evasion, u_sel = u_nom
        ax.plot(u_sel[1], u_sel[0], "^", color="seagreen", ms=8,
                markeredgecolor="white", markeredgewidth=0.8, zorder=7,
                label=r"$\mathbf{u}_\mathrm{sel}$ (= nom)")

    # ref-speed horizontal guide
    ax.axhline(frame.ref_speed, color=color, lw=0.8, ls=":", alpha=0.55,
               label=f"ref speed ({frame.ref_speed:.1f} m/s)")

    # --- Fixed limits: show the full feasible action envelope ----------------
    # Physical limits: u1 in [0, ref_speed], u2 in [±ref_speed * omega_max]
    v   = frame.ref_speed
    u2m = v * frame.omega_max if frame.omega_max > 0 else max(abs(u_nom[1]), 0.5)
    pad1 = v   * 0.20
    pad2 = u2m * 0.30
    ax.set_xlim(-u2m - pad2, u2m + pad2)   # x = u2
    ax.set_ylim(-pad1, v + pad1)            # y = u1

    # Dashed lines marking the physical action limits
    ax.axhline(0,  color="dimgray", lw=0.7, ls="--", alpha=0.40, zorder=0)
    ax.axvline( u2m, color="dimgray", lw=0.7, ls="--", alpha=0.40, zorder=0,
                label=r"$\omega_\mathrm{max}$ limit")
    ax.axvline(-u2m, color="dimgray", lw=0.7, ls="--", alpha=0.40, zorder=0)

    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

    ax.set_xlabel(r"$u_2 = v\omega\;[\mathrm{m/s}^2]$", fontsize=FONT_SIZE)
    ax.set_ylabel(r"$u_1 = v\;[\mathrm{m/s}]$",         fontsize=FONT_SIZE)
    ax.set_title(f"{vessel_id}  —  CTO action space", fontsize=FONT_SIZE)
    ax.legend(fontsize=FONT_SIZE, loc="best")
    ax.grid(True, alpha=0.22)
    ax.tick_params(labelsize=FONT_SIZE)


def _draw_vo_debug(
    ax: plt.Axes,
    frame: VODebugFrame,
    color: str,
    vessel_id: str,
) -> None:
    """Render VO velocity-space snapshot (x = vₑ, y = vₙ).

    Three-layer annotation (mirroring the CTO panel):
      v_pref (★) → v_pref_proj (◆, when projection active) → v_sel (▲)
    The cone fill switches to coral when the fallback (d_coll) regime is active.
    """
    import math
    ax.cla()
    R = frame.ref_speed

    # Check projection validity early — needed for limit computation
    has_proj = not (math.isnan(frame.v_pref_proj_vn) or math.isnan(frame.v_pref_proj_ve))

    # Dynamic axis limits: scale to the actual velocity data in this frame
    pts = [R, abs(frame.v_pref_ve), abs(frame.v_pref_vn),
               abs(frame.v_sel_ve),  abs(frame.v_sel_vn)]
    if has_proj:
        pts += [abs(frame.v_pref_proj_ve), abs(frame.v_pref_proj_vn)]
    limit  = max(max(pts) * 1.3, 0.5)
    cone_R = limit * 3.0
    theta  = np.linspace(0, 2 * np.pi, 200)

    ax.axvline(0, color="lightgray", lw=0.6, zorder=0)
    ax.axhline(0, color="lightgray", lw=0.6, zorder=0)

    # Reference speed circle
    ax.plot(R * np.cos(theta), R * np.sin(theta),
            "--", color=color, lw=0.9, alpha=0.55, zorder=1,
            label=f"ref speed ({R:.1f} m/s)")

    # VO cones — coral fill/edge signals fallback (d_coll) regime
    cone_color = "coral" if frame.fallback_active else "slategray"
    for i, cone in enumerate(frame.cones):
        apex_x = cone.apex_ve
        apex_y = cone.apex_vn
        axis_a = np.arctan2(cone.axis_n, cone.axis_e)
        t0     = axis_a - cone.half_angle
        t1     = axis_a + cone.half_angle
        ts     = np.linspace(t0, t1, 60)
        arc_x  = apex_x + cone_R * np.cos(ts)
        arc_y  = apex_y + cone_R * np.sin(ts)
        poly_x = [apex_x, *arc_x, apex_x]
        poly_y = [apex_y, *arc_y, apex_y]
        lbl = ("VO cone (d_coll fallback)" if frame.fallback_active else "VO cone") if i == 0 else None
        ax.fill(poly_x, poly_y, alpha=0.18, color=cone_color, zorder=1, label=lbl)
        ax.plot([apex_x, apex_x + cone_R * np.cos(t0)],
                [apex_y, apex_y + cone_R * np.sin(t0)],
                "-", color=cone_color, lw=1.2, alpha=0.8, zorder=2)
        ax.plot([apex_x, apex_x + cone_R * np.cos(t1)],
                [apex_y, apex_y + cone_R * np.sin(t1)],
                "-", color=cone_color, lw=1.2, alpha=0.8, zorder=2)

    # Starboard constraints — port-AND-converging exclusion wedges.
    # The excluded region is the intersection of two half-planes:
    #   (1) port side:       cross(rel_pos, v - v_obs) < 0
    #   (2) converging side: dot(v - v_obs, rel_pos)   > 0
    # Both boundaries pass through the apex (obs_vel), forming a wedge.
    # In plot (ve=x, vn=y) space the two boundary ray directions are:
    #   dir1 = (bearing_e, bearing_n)   — port boundary, toward converging half
    #   dir2 = (-bearing_n, bearing_e)  — convergence boundary, toward port half
    sb_color = "darkorange"
    for i, sb in enumerate(frame.starboard_constraints):
        apex_plot = np.array([sb.apex_ve, sb.apex_vn])
        dir1 = np.array([ sb.bearing_e,  sb.bearing_n])   # port boundary ray
        dir2 = np.array([-sb.bearing_n,  sb.bearing_e])   # convergence boundary ray

        T = cone_R
        p1 = apex_plot
        p2 = apex_plot + T * dir1
        p3 = apex_plot + T * (dir1 + dir2)
        p4 = apex_plot + T * dir2

        lbl_fill = "starboard constraint" if i == 0 else None
        ax.fill(
            [p1[0], p2[0], p3[0], p4[0]],
            [p1[1], p2[1], p3[1], p4[1]],
            alpha=0.13, color=sb_color, hatch="///", zorder=1, label=lbl_fill,
        )
        # Port boundary ray — dashed
        ax.plot(
            [apex_plot[0], p2[0]], [apex_plot[1], p2[1]],
            "--", color=sb_color, lw=1.6, alpha=0.9, zorder=2,
        )
        # Convergence boundary ray — dotted
        ax.plot(
            [apex_plot[0], p4[0]], [apex_plot[1], p4[1]],
            ":", color=sb_color, lw=1.6, alpha=0.9, zorder=2,
        )

    # Forward half-plane constraint — shaded backward region + boundary line.
    # Active only in the primary regime (forward_unit is NaN in fallback).
    # In plot (ve=x, vn=y): forward_unit_plot = [fe, fn]; boundary line is
    # perpendicular to that direction, passing through the origin.
    has_fwd = not (math.isnan(frame.forward_unit_n) or math.isnan(frame.forward_unit_e))
    if has_fwd:
        fu_x = frame.forward_unit_e   # east → x
        fu_y = frame.forward_unit_n   # north → y
        bd_x, bd_y = -fu_y, fu_x     # 90° CCW in plot = boundary line direction
        T = limit * 2.5
        # Shade the forbidden backward half-plane
        p1 = np.array([-T * bd_x, -T * bd_y])
        p2 = np.array([ T * bd_x,  T * bd_y])
        p3 = p2 - T * np.array([fu_x, fu_y])
        p4 = p1 - T * np.array([fu_x, fu_y])
        fwd_color = "mediumseagreen"
        ax.fill(
            [p1[0], p2[0], p3[0], p4[0]],
            [p1[1], p2[1], p3[1], p4[1]],
            alpha=0.10, color=fwd_color, hatch="...", zorder=1,
            label="forward constraint",
        )
        ax.plot(
            [-T * bd_x, T * bd_x], [-T * bd_y, T * bd_y],
            "--", color=fwd_color, lw=1.6, alpha=0.9, zorder=2,
        )

    vp_x, vp_y = frame.v_pref_ve, frame.v_pref_vn
    vs_x, vs_y = frame.v_sel_ve,  frame.v_sel_vn

    # Scoring anchor (v_pref_proj): valid only when the field is not nan
    if has_proj:
        vpp_x    = frame.v_pref_proj_ve
        vpp_y    = frame.v_pref_proj_vn
        proj_shifted = abs(vpp_x - vp_x) > 0.01 or abs(vpp_y - vp_y) > 0.01
    else:
        proj_shifted = False

    # v_pref star
    ax.plot(vp_x, vp_y, "k*", ms=10, zorder=5,
            label=r"$\mathbf{v}_\mathrm{pref}$")

    # Projection anchor: arrow v_pref → v_pref_proj, diamond marker
    if has_proj and proj_shifted:
        ax.annotate("", xy=(vpp_x, vpp_y), xytext=(vp_x, vp_y),
                    arrowprops=dict(arrowstyle="->", color="steelblue", lw=1.4),
                    zorder=4)
        ax.plot(vpp_x, vpp_y, "D", color="steelblue", ms=7,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6,
                label=r"$\mathbf{v}_\mathrm{proj}$ (scoring anchor)")
        arrow_src_x, arrow_src_y = vpp_x, vpp_y
    else:
        arrow_src_x, arrow_src_y = vp_x, vp_y

    # v_sel triangle with arrow from the scoring anchor (or v_pref if no proj)
    sel_diff  = abs(vs_x - vp_x) > 0.01 or abs(vs_y - vp_y) > 0.01
    sel_color = "tomato" if sel_diff else "seagreen"
    sel_label = (r"$\mathbf{v}_\mathrm{sel}$" if sel_diff
                 else r"$\mathbf{v}_\mathrm{sel}$ (= pref)")
    if sel_diff:
        ax.annotate("", xy=(vs_x, vs_y), xytext=(arrow_src_x, arrow_src_y),
                    arrowprops=dict(arrowstyle="->", color="tomato", lw=1.5),
                    zorder=4)
    ax.plot(vs_x, vs_y, "^", color=sel_color, ms=8,
            markeredgecolor="white", markeredgewidth=0.8,
            zorder=7, label=sel_label)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel(r"East vel. $v_e\;[\mathrm{m/s}]$", fontsize=FONT_SIZE)
    ax.set_ylabel(r"North vel. $v_n\;[\mathrm{m/s}]$", fontsize=FONT_SIZE)

    title_suffix = "  [FALLBACK]" if frame.fallback_active else ""
    title_color  = "firebrick"    if frame.fallback_active else "black"
    ax.set_title(f"{vessel_id}  —  VO velocity space{title_suffix}",
                 fontsize=FONT_SIZE, color=title_color)
    ax.legend(fontsize=FONT_SIZE, loc="best")
    ax.grid(True, alpha=0.22)
    ax.tick_params(labelsize=FONT_SIZE)


def _render_debug(
    ax: plt.Axes,
    frame: PolicyDebugFrame,
    color: str,
    vessel_id: str,
) -> None:
    if isinstance(frame, CTODebugFrame):
        _draw_cto_debug(ax, frame, color, vessel_id)
    elif isinstance(frame, VODebugFrame):
        _draw_vo_debug(ax, frame, color, vessel_id)


def _draw_responding_panel(
    ax:                 plt.Axes,
    responding_history: dict[str, list[float]],
    vessel_colors:      dict[str, str],
    k:                  int,
    dt:                 float,
    n_steps:            int,
    window:            Optional[tuple[float, float]] = None,
    compact:           bool = False,
    title:             Optional[str] = None,
    t0:                float = 0.0,
) -> None:
    """Render the p(r = R) time series for all vessels.

    `window` optionally crops the time axis to [t_start, t_end] seconds
    instead of showing the full run.

    `compact` is for the panel tucked under the Interaction Graph
    (vc.responding_under_graph): the per-curve value labels sit next to the
    curve instead of in the legend, and the legend moves to the lower-left.
    Neither variant shows y-ticks or a y-label — the legend title names the
    quantity ($p(r=R)$) and the per-curve labels/annotations are the readout.

    `t0` is the sim time [s] of ``history[0]`` for every vessel — 0.0 when
    ``responding_history`` is a complete precomputed array indexed from the
    start of the run (run_simulation.py), or the playback start time when it's
    built incrementally frame-by-frame from some later start_step (run_AIS.py).
    It anchors both the curve's x-axis and which history index counts as
    "now", so the current-value dot/annotation always lands exactly on the
    vertical dashed current-time marker regardless of which convention the
    caller uses.
    """
    ax.cla()
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, alpha=0.22, linestyle="--")
    if title:
        ax.set_title(title, fontsize=FONT_SIZE)
    ax.set_xlim(*(window if window is not None else (0.0, n_steps * dt)))
    # A little headroom above the p = 1 line so labels/markers near the top
    # aren't cramped against the frame.
    ax.set_ylim(-0.05, 1.10) if compact else ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$t\;[\mathrm{s}]$", fontsize=FONT_SIZE)
    ax.axhline(0.5, color="dimgray", lw=0.7, ls=":", alpha=0.5)
    ax.axvline(k * dt, color="dimgray", lw=2.4, ls="--", alpha=0.85, zorder=2)
    ax.tick_params(labelsize=FONT_SIZE)
    # No y-ticks / y-label in either variant — see docstring.
    ax.set_yticks([])

    t_lo, t_hi = ax.get_xlim()
    k0 = round(t0 / dt) if dt else 0
    for vid, history in responding_history.items():
        if not history:
            continue
        color = vessel_colors.get(vid, "gray")
        t = t0 + np.arange(len(history)) * dt
        ax.plot(t, history, "-", color=color, lw=3.0, zorder=3,
                label=vid if compact else None)
        current_idx = min(max(k - k0, 0), len(history) - 1)
        p = history[current_idx]
        ax.plot(t[current_idx], p, "o", color=color, ms=11 if compact else 9, zorder=5,
                markeredgecolor="white", markeredgewidth=1.0,
                label=None if compact else f"{vid} = {p:.2f}")
        if compact:
            xt = min(max(t[current_idx], t_lo), t_hi)
            ax.annotate(
                f"{p:.2f}", (xt, p),
                textcoords="offset points", xytext=(8, 0),
                ha="left", va="center", fontsize=FONT_SIZE,
                color=color, fontweight="bold", zorder=6, clip_on=False,
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
            )

    if compact:
        ax.legend(title=r"$p(r{=}R)$", fontsize=FONT_SIZE, title_fontsize=FONT_SIZE,
                  loc="lower left", framealpha=0.85, handlelength=1.3,
                  handletextpad=0.5, borderpad=0.4, labelspacing=0.3)
    else:
        ax.legend(title=r"$p(r = R)$", fontsize=FONT_SIZE, title_fontsize=FONT_SIZE,
                  loc="best", framealpha=0.85)


def save_responding_plot(
    responding_history: dict[str, list[float]],
    vessel_colors:      dict[str, str],
    dt:                 float,
    vc:                 "VizConfig",
    sim_name:           str,
    window:             Optional[tuple[float, float]] = None,
    t0:                 float = 0.0,
) -> Optional[Path]:
    """Save a standalone p(responding)-vs-time plot, independent of any live
    LiveVisualizer panel — for a dedicated paper figure of just the
    responding state, rather than the full trajectory plot.

    Styled/saved the same way as LiveVisualizer._save_frame_pdf
    (vc.frame_format / frame_dpi), so it matches the trajectory plot's look.

    `t0` is the simulation time [s] of the first recorded sample (e.g.
    start_step * dt, since history collection starts at start_step, not
    frame 0) — needed so `window` lines up with the same absolute time axis
    shown live, rather than being measured from the start of the history.
    `window` crops the x-axis to [t_start, t_end] seconds.
    Returns the saved path, or None if there's no history to plot.
    """
    if not any(responding_history.values()):
        return None

    _apply_font_config(vc)
    fs     = FONT_SIZE
    fig_w  = vc.figsize[0]
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.32))
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"$t\;[\mathrm{s}]$", fontsize=fs)
    ax.set_ylabel(r"$p(r = R)$", fontsize=fs)
    ax.tick_params(labelsize=fs)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    max_len = max((len(h) for h in responding_history.values()), default=0)
    if window is not None:
        ax.set_xlim(*window)
    elif max_len:
        ax.set_xlim(t0, t0 + (max_len - 1) * dt)

    for vid, history in responding_history.items():
        if not history:
            continue
        color = vessel_colors.get(vid, "gray")
        t = t0 + np.arange(len(history)) * dt
        ax.plot(t, history, "-", color=color, lw=3.0, zorder=3, label=vid)

    ax.legend(fontsize=fs, loc="best", framealpha=0.85)
    fig.tight_layout()

    frames_dir = (
        Path(vc.frames_dir) if vc.frames_dir
        else Path(__file__).resolve().parents[2] / "outputs" / "frames"
    )
    frames_dir.mkdir(parents=True, exist_ok=True)
    slug      = re.sub(r"[^\w\-]+", "_", sim_name).strip("_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fmt       = vc.frame_format.lower()
    path      = frames_dir / f"{slug}_responding_{timestamp}.{fmt}"
    save_kwargs = {"dpi": vc.frame_dpi} if fmt == "png" else {}
    fig.savefig(path, bbox_inches="tight", **save_kwargs)
    plt.close(fig)
    print(f"Saved responding-state plot -> {path}")
    return path


def _draw_colregs_graph(
    ax:           plt.Axes,
    states:       dict[str, VesselState],
    graph:        ColregsGraph,
    idx_to_id:    dict[int, str],
    idx_to_color: dict[int, str],
    t:            float,
    vc:           "VizConfig",
    scale:        float   = 1.0,
    idx_to_label: Optional[dict[int, str]] = None,
    *,
    text_scale:      Optional[float] = None,
    node_label_size: Optional[float] = None,
    edge_scale:      float = 1.0,
    title:           Optional[str] = None,
) -> None:
    """Clear ax and draw the COLREGs role graph (nodes + edges).  No canvas flush.

    `scale` shrinks all text/markers together (relative to FONT_SIZE) for a
    smaller copy of this panel, e.g. the corner-inset layout (vc.cg_inset),
    without touching the module-wide FONT_SIZE used everywhere else.
    `text_scale` (default = `scale`) sizes the role labels / legend / "no
    encounters" text independently of the marker geometry; `node_label_size`
    likewise sets the node ID label size directly. Used by the multi-panel
    figure so all text stays a fixed multiple of the configured font size.
    `idx_to_id` maps vessel index -> the key used in `states` (the internal
    vessel_id); `idx_to_label` (optional) maps index -> the name shown on the
    node (e.g. "V1"). When omitted, the id doubles as the label.
    """
    if text_scale is None:
        text_scale = scale
    fs = FONT_SIZE * text_scale
    lbl_map = idx_to_label or idx_to_id
    # Monochrome test mode (vc.cg_monochrome): drop the encounter-type palette.
    #   "black" / "grey" — one ink for both edges and role labels
    #   "black-grey"     — near-black role labels over mid-grey edges
    _mono = str(getattr(vc, "cg_monochrome", "off")).lower()
    _BLACK, _GREY = "#1f2937", "#6b7280"
    if _mono in ("black-grey", "grey-black", "mixed"):
        mono_edge, mono_label = _GREY, _BLACK
    else:
        _c = {"black": _BLACK, "grey": _GREY, "gray": _GREY}.get(_mono)
        mono_edge = mono_label = _c
    ax.cla()
    ax.set_aspect("auto")
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fs)

    positions_e = [s.east  for s in states.values()]
    positions_n = [s.north for s in states.values()]
    pad = (max(positions_n) - min(positions_n) + max(positions_e) - min(positions_e)) / 2 * 0.15
    pad = max(pad, 20.0)  # minimum padding so single-vessel edge cases don't collapse

    e_min, e_max = min(positions_e) - pad, max(positions_e) + pad
    n_min, n_max = min(positions_n) - pad, max(positions_n) + pad

    ax.set_xlim(e_min, e_max)
    ax.set_ylim(n_min, n_max)

    # --- Edges (drawn under nodes) ---
    for edge in graph.edges:
        id_a = idx_to_id.get(edge.vessel_a)
        id_b = idx_to_id.get(edge.vessel_b)
        if id_a not in states or id_b not in states:
            continue
        sa, sb = states[id_a], states[id_b]
        ea, na = sa.east, sa.north
        eb, nb = sb.east, sb.north

        if edge.has_non_responsive:
            e_color, e_ls, e_alpha = mono_edge or CG_EDGE_COLOR, CG_OVERRIDDEN_LS, 0.85
        else:
            e_color = mono_edge or _cg_edge_color(edge.role_a, edge.role_b)
            e_ls, e_alpha = "-", 0.75
        ax.plot([ea, eb], [na, nb], color=e_color, ls=e_ls,
                lw=3.0 * scale * edge_scale, alpha=e_alpha, zorder=2)

        # Role labels at 28 % and 72 % along the edge. Sized down from fs:
        # bold text in a filled box reads larger than a plain label at the
        # same point size, and these sit on a small panel.
        for frac, role in ((0.28, edge.role_a), (0.72, edge.role_b)):
            lx = ea + frac * (eb - ea)
            ly = na + frac * (nb - na)
            ax.text(
                lx, ly, _ROLE_LABEL[role],
                fontsize=fs * 0.8, ha="center", va="center",
                color=mono_label or _ROLE_COLOR[role], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.82, ec="none"),
                zorder=4,
            )

    # --- Nodes — point marker + same halo'd ID label styling as the main map ---
    for idx, vid in idx_to_id.items():
        if vid not in states:
            continue
        _draw_vessel_marker(ax, states[vid], idx_to_color[idx], vc,
                            lbl_map.get(idx, vid), scale=scale,
                            label_fontsize=node_label_size)

    # --- Overlays ---
    if not graph.edges:
        ax.text(0.5, 0.5, "No active encounters",
                transform=ax.transAxes,
                fontsize=fs, ha="center", va="center",
                color="gray", alpha=0.5)

    # Encounter type is colour-coded on the edge and spelled out by its role
    # labels; the legend only explains the solid/dashed (nominal/overridden)
    # distinction, with a neutral proxy line.
    if vc.cg_legend:
        _proxy = mono_edge or CG_EDGE_COLOR
        ax.legend(handles=[
            Line2D([0], [0], color=_proxy, lw=3.0, label="Nominal edge"),
            Line2D([0], [0], color=_proxy, lw=3.0, ls=CG_OVERRIDDEN_LS,
                   label="Overridden edge"),
        ], loc="lower left", fontsize=fs, framealpha=0.8,
           handlelength=1.8, handletextpad=0.5, borderpad=0.4, labelspacing=0.3)


# ---------------------------------------------------------------------------
# Vessel artist (main map)
# ---------------------------------------------------------------------------

class VesselArtist:
    def __init__(self, ax, meta: VesselMeta, color: str, vc: VizConfig) -> None:
        self.ax         = ax
        self.color      = color
        self.vc         = vc
        self._vessel_id = meta.label

        self._body_poly = _ship_polygon_body(
            vc.vessel_length, vc.vessel_width, vc.bow_fraction
        )

        s0 = meta.initial_state

        self.trail_north: list[float] = []
        self.trail_east:  list[float] = []
        self.trail_line, = ax.plot(
            [], [], "-", color=color, alpha=vc.trail_alpha, lw=3.0, zorder=2,
        )

        e0, n0 = _rotate_translate(self._body_poly, s0.heading, s0.north, s0.east)
        self.poly = plt.Polygon(
            np.column_stack([e0, n0]),
            closed=True,
            facecolor=color, alpha=vc.vessel_alpha,
            edgecolor="white", linewidth=1.2,
            zorder=4,
        )
        ax.add_patch(self.poly)

        # Fixed-screen-size heading marker: a triangle pointing along the
        # vessel's heading. Sized in points, so it stays legible at any zoom
        # where the true hull polygon is just a dot. Compass heading is
        # clockwise from north; matplotlib marker angles are counter-clockwise
        # from "up" (+north), hence the sign flip.
        self.heading_marker = None
        if vc.heading_marker:
            self.heading_marker, = ax.plot(
                [s0.east], [s0.north],
                marker=(3, 0, -np.degrees(s0.heading)),
                markersize=vc.heading_marker_size,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=1.0, alpha=vc.heading_marker_alpha,
                linestyle="none", zorder=3,
            )

        self.text = ax.text(
            s0.east, s0.north, meta.label,
            fontsize=vc.vessel_label_fontsize, color=color,
            ha="center", va="center",
            fontweight="bold", fontfamily="serif",
            zorder=6,
            path_effects=[
                patheffects.withStroke(linewidth=2.5, foreground="white"),
            ],
        )

    def update(self, state: VesselState) -> None:
        n, e = state.north, state.east
        h    = state.heading

        self.trail_north.append(n)
        self.trail_east.append(e)
        if self.vc.trail_length > 0 and len(self.trail_north) > self.vc.trail_length:
            self.trail_north.pop(0)
            self.trail_east.pop(0)
        self.trail_line.set_data(self.trail_east, self.trail_north)

        e_rot, n_rot = _rotate_translate(self._body_poly, h, n, e)
        self.poly.set_xy(np.column_stack([e_rot, n_rot]))

        if self.heading_marker is not None:
            self.heading_marker.set_data([e], [n])
            self.heading_marker.set_marker((3, 0, -np.degrees(h)))
            # Hull is invisible at this scale, so the metre-based label offset
            # would leave the label sitting on top of the triangle. Offset it a
            # fixed number of points to starboard instead — same visual
            # relationship the label has to the hull in the simulation case.
            p = self.vc.heading_marker_size * 0.9
            self.text.set_transform(offset_copy(
                self.ax.transData, self.ax.figure,
                x=p * np.cos(h), y=-p * np.sin(h), units="points",
            ))
            self.text.set_position((e, n))
        else:
            offset  = self.vc.vessel_width * self.vc.label_offset
            label_n = n + (-np.sin(h)) * offset
            label_e = e + ( np.cos(h)) * offset
            self.text.set_position((label_e, label_n))
        self.text.set_text(self._vessel_id)


# ---------------------------------------------------------------------------
# Main visualizer
# ---------------------------------------------------------------------------

class LiveVisualizer:
    def __init__(
        self,
        sim_info:             SimInfo,
        vc:                   Optional[VizConfig]                  = None,
        xlim:                 Optional[tuple]                      = None,
        ylim:                 Optional[tuple]                      = None,
        debug_vessel_ids:     Optional[list[str]]                  = None,
        pred_fn:              Optional[Callable[[dict, int], dict]] = None,
        nominal_pred_fn:      Optional[Callable[[dict, int], dict]] = None,
        pred_every:           int                                   = 1,
        pred_top_k:           int                                   = 1,
        plan_fn:              Optional[Callable[[int], dict[str, PlanOverlay]]] = None,
        show_colregs_graph:   bool                                 = False,
        responding_history:   Optional[dict[str, list[float]]]     = None,
        responding_history_t0: float                                = 0.0,
        show_future_traj:     bool                                 = False,
        is_ais:               bool                                 = False,
        goals:                Optional[dict[str, tuple[float, float]]] = None,
    ) -> None:
        self.sim_info = sim_info
        self.vc       = vc or VizConfig.default()
        self._paused  = False
        self._stopped = False
        self._video_writer = None
        self._last_step = 0
        # sim time [s] of index 0 in every responding_history list — 0.0 when the
        # history is a complete precomputed array from the start of the run
        # (run_simulation.py), or the playback start time when it's instead built
        # incrementally from some later start_step (run_AIS.py). See
        # _draw_responding_panel's docstring for why this matters.
        self._responding_history_t0 = responding_history_t0

        _apply_font_config(self.vc)

        debug_vessel_ids = debug_vessel_ids or []
        show_debug       = self.vc.debug_viz and len(debug_vessel_ids) > 0
        n_debug          = min(len(debug_vessel_ids), 2) if show_debug else 0
        show_cg          = show_colregs_graph and self.vc.colregs_graph_viz
        # cg_inset: fold the Interaction Graph into a corner inset of the main
        # map instead of giving it its own side-by-side column — a genuine
        # layout switch (affects the live view too), not just a save-time
        # trick, so "save" stays simple: it just writes whatever's on screen.
        cg_inset         = show_cg and self.vc.cg_inset
        self._cg_inset       = cg_inset
        self._cg_inset_scale = 0.8 if cg_inset else 1.0
        self._panel_titles   = bool(self.vc.panel_titles)
        show_responding  = bool(responding_history) and self.vc.responding_panel
        n_responding     = 1 if show_responding else 0
        # Live p(responding)-vs-time panel tucked under the Interaction Graph in
        # the same right column (vc.responding_under_graph). Needs the graph
        # itself in its own column/stack cell (not the corner inset).
        show_resp_under  = (
            bool(responding_history) and self.vc.responding_under_graph
            and show_cg and not cg_inset
        )
        n_right          = n_debug + (1 if (show_cg and not cg_inset) else 0) + n_responding

        self._responding_history: Optional[dict[str, list[float]]] = (
            responding_history if show_responding else None
        )
        # Raw history kept regardless of the side panel — the inline per-vessel
        # belief bars (vc.responding_inline) only need the current value.
        self._responding_history_raw = responding_history or {}
        self._responding_inline = bool(self._responding_history_raw) and self.vc.responding_inline
        self._resp_cmap = plt.get_cmap("RdYlGn")
        # Keyed by both the internal vessel_id and the display label ("V1"), so
        # callers passing either (e.g. responding_history keyed by display name)
        # still resolve to the right colour.
        self._vessel_colors: dict[str, str] = {}
        for i, m in enumerate(sim_info.vessels):
            c = VESSEL_COLORS[i % len(VESSEL_COLORS)]
            self._vessel_colors[m.vessel_id] = c
            self._vessel_colors[m.label]     = c

        debug_panel_w = 5.5   # inches per right-column panel

        # Resolve the trajectory-plot view window now, before sizing the main
        # map's column below — with aspect="equal" it's this data aspect ratio
        # (not vc.figsize's) that determines how much of the column the map
        # actually fills. set_xlim/set_ylim on self.ax happen later, once the
        # axes exists; the values are just computed once, up here.
        # Precedence: explicit xlim/ylim args > viz.xlim/ylim (both must be set)
        # > auto extent from the trajectories.
        if xlim is None and self.vc.xlim is not None and self.vc.ylim is not None:
            xlim, ylim = tuple(self.vc.xlim), tuple(self.vc.ylim)
        if xlim is None or ylim is None:
            xlim, ylim = self._auto_limits(sim_info, self.vc.padding,
                                           center_ego_x=self.vc.center_ego_x)

        if n_right == 0:
            self.fig, self.ax = plt.subplots(figsize=self.vc.figsize)
            self.debug_axes: dict[str, plt.Axes] = {}
            self.colregs_ax: Optional[plt.Axes]  = None
            self._responding_ax: Optional[plt.Axes] = None
            self._resp_under_ax: Optional[plt.Axes] = None
        else:
            # Outer layout is always a single row, so every top-level column
            # (main map, and each side column) spans the exact same vertical
            # extent by construction — no aspect/hspace bookkeeping needed to
            # keep them aligned. COLREGs graph gets its own column only when
            # VO/CTO debug panels are also present (matching the previous
            # three-column layout); otherwise it stacks with debug/responding
            # in the single right column. Any column holding more than one
            # panel is subdivided with a nested GridSpec, whose combined span
            # is pinned to the parent column's full height — so two stacked
            # panels (e.g. COLREGs graph + responding) always sum to exactly
            # the trajectory plot's height, each getting an equal share.
            cg_own_column = show_cg and not cg_inset and n_debug > 0
            if cg_own_column:
                ncols         = 3
                n_stack_right = n_debug + n_responding
            else:
                ncols         = 2
                n_stack_right = n_debug + (1 if (show_cg and not cg_inset) else 0) + n_responding

            fh = max(self.vc.figsize[1], max(n_stack_right, 1) * 4.5)
            # Size the main map's column from the *data*'s east:north aspect
            # ratio (xlim/ylim resolved above), not vc.figsize's — aspect="equal"
            # (needed for undistorted NED distances) fits the map to whichever
            # of the column's width/height is limiting, so a column shaped
            # differently from the data always left a gap on one side (pushed
            # to the outer-left margin by the anchor="E" below). Matching the
            # column to the data's own ratio makes the map fill it edge-to-edge.
            _yr          = ylim[1] - ylim[0]
            data_aspect  = (xlim[1] - xlim[0]) / _yr if _yr else 1.0
            main_w       = fh * data_aspect
            width_ratios = [main_w] + [debug_panel_w] * (ncols - 1)
            fw           = main_w + (ncols - 1) * debug_panel_w
            self.fig = plt.figure(figsize=(fw, fh))

            OUTER_WSPACE = 0.04
            gs_outer = gridspec.GridSpec(1, ncols, width_ratios=width_ratios,
                                         wspace=OUTER_WSPACE)
            # Absolute size (inches) of the main-map ↔ side-column gutter, reused
            # below as the gap between stacked side panels so every inter-panel
            # gutter reads the same. wspace/hspace are both fractions of the mean
            # axes size along that axis, so matching them means dividing the same
            # absolute gap by the mean panel height instead of the mean width.
            _gutter_in   = OUTER_WSPACE * (fw / ncols)
            _usable_h    = fh * (0.94 - 0.09)   # subplots_adjust top - bottom, below
            self.ax = self.fig.add_subplot(gs_outer[0, 0])
            # aspect="equal" makes the main map square, so it never fills its
            # (wider) column — anchor it to the column's right edge so the
            # leftover whitespace falls on the figure's outer margin rather
            # than between the map and the side panels.
            self.ax.set_anchor("E")

            self.colregs_ax: Optional[plt.Axes] = None
            self._resp_under_ax: Optional[plt.Axes] = None
            if cg_own_column:
                if show_resp_under:
                    # Split the graph's own column: graph on top, responding
                    # panel below, summing to the column's full height.
                    gs_cg = gridspec.GridSpecFromSubplotSpec(
                        2, 1, subplot_spec=gs_outer[0, 1],
                        hspace=_gutter_in / (_usable_h / 2),
                        height_ratios=[1.0, self.vc.responding_under_graph_ratio],
                    )
                    self.colregs_ax = self.fig.add_subplot(gs_cg[0, 0])
                    self._resp_under_ax = self.fig.add_subplot(gs_cg[1, 0])
                else:
                    self.colregs_ax = self.fig.add_subplot(gs_outer[0, 1])
                self.colregs_ax.set_anchor("W")
                stack_cell = gs_outer[0, 2]
            else:
                stack_cell = gs_outer[0, 1]

            stack_items: list = []
            stack_ratios: list[float] = []
            if not cg_own_column and show_cg and not cg_inset:
                stack_items.append("colregs")
                stack_ratios.append(1.0)
                if show_resp_under:
                    stack_items.append("responding")
                    stack_ratios.append(self.vc.responding_under_graph_ratio)
            for vid in debug_vessel_ids[:n_debug]:
                stack_items.append(("debug", vid))
                stack_ratios.append(1.0)
            if show_responding:
                stack_items.append("responding_panel")
                stack_ratios.append(1.0)

            self.debug_axes = {}
            self._responding_ax = None
            if stack_items:
                # Match the main-map gutter for the responding-under-graph stack;
                # debug/CTO panels keep the roomier default (axis labels + titles).
                if show_resp_under and n_debug == 0 and len(stack_items) >= 2:
                    stack_hspace = _gutter_in / (_usable_h / len(stack_items))
                else:
                    stack_hspace = 0.18
                gs_stack = gridspec.GridSpecFromSubplotSpec(
                    len(stack_items), 1, subplot_spec=stack_cell, hspace=stack_hspace,
                    height_ratios=stack_ratios,
                )
                for i, item in enumerate(stack_items):
                    ax_i = self.fig.add_subplot(gs_stack[i, 0])
                    if item == "colregs":
                        self.colregs_ax = ax_i
                        ax_i.set_anchor("W")
                    elif item == "responding":
                        self._resp_under_ax = ax_i
                    elif item == "responding_panel":
                        self._responding_ax = ax_i
                    else:
                        _, vid = item
                        self.debug_axes[vid] = ax_i

        # Configure main axis
        self.ax.set_xlabel(r"East $[\mathrm{m}]$")
        self.ax.set_ylabel(r"North $[\mathrm{m}]$")
        self.ax.set_aspect("equal")
        self.ax.set_facecolor(PANEL_BG)
        self.ax.grid(False)
        if self._panel_titles:
            self.ax.set_title("Trajectory Plot", fontsize=FONT_SIZE)

        # (xlim/ylim already resolved above, before the map's column was sized)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        if cg_inset:
            inset_w  = float(self.vc.cg_inset_size)
            # inset_axes bounds are an axes-fraction of self.ax's own box, but
            # with aspect="equal" (adjustable="box", the default) that box is
            # itself shrunk ("letterboxed") so its width:height ratio matches
            # xrange:yrange — it's square only when xlim and ylim happen to
            # span equal ranges. An equal (w, w) fraction would then render
            # stretched, so scale the height fraction by that same ratio to
            # keep the inset visually square regardless of the trajectory
            # plot's xlim/ylim shape.
            _xr = xlim[1] - xlim[0]
            _yr = ylim[1] - ylim[0]
            inset_h  = inset_w * (_xr / _yr) if _yr else inset_w
            _side    = self.vc.cg_inset_side.lower().replace("_", "-")
            _left    = "left" in _side
            _top     = "top" in _side
            inset_x0 = 0.02 if _left else (0.98 - inset_w)
            inset_y0 = (0.97 - inset_h) if _top else 0.03
            self.colregs_ax = self.ax.inset_axes([inset_x0, inset_y0, inset_w, inset_h])
            self.colregs_ax.set_zorder(6)  # stay above the main map's own artists
            for spine in self.colregs_ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(0.8)

        # Colregs-graph panel state (only populated when colregs_ax is set).
        # _cg_idx_to_id keys match the `states` dict (internal vessel_id);
        # _cg_idx_to_label is the on-node display name ("V1", ...).
        self._cg_idx_to_id:    dict[int, str] = {}
        self._cg_idx_to_label: dict[int, str] = {}
        self._cg_idx_to_color: dict[int, str] = {}
        if self.colregs_ax is not None:
            self._cg_idx_to_id    = {i: m.vessel_id for i, m in enumerate(sim_info.vessels)}
            self._cg_idx_to_label = {i: m.label     for i, m in enumerate(sim_info.vessels)}
            self._cg_idx_to_color = {i: VESSEL_COLORS[i % len(VESSEL_COLORS)]
                                     for i in range(len(sim_info.vessels))}

        # Vessel artists
        self.artists: dict[str, VesselArtist] = {}
        for idx, meta in enumerate(sim_info.vessels):
            color = VESSEL_COLORS[idx % len(VESSEL_COLORS)]
            self.artists[meta.vessel_id] = VesselArtist(self.ax, meta, color, self.vc)

        # Inline p(responding) belief bars — one small horizontal bar per vessel
        # that has a responding history (typically the targets, not ego). The
        # geometry is recomputed every frame in _update_responding_bars().
        self._responding_bars: dict[str, tuple] = {}
        if self._responding_inline:
            for meta in sim_info.vessels:
                key = self._resp_history_key(meta)
                if key is None:
                    continue
                track = Rectangle((0, 0), 1, 1, facecolor="0.88", edgecolor="0.45",
                                  lw=1.6, alpha=0.92, zorder=5.0, visible=False,
                                  clip_on=False)
                fill  = Rectangle((0, 0), 1, 1, facecolor="0.5", edgecolor="none",
                                  zorder=5.1, visible=False, clip_on=False)
                tick, = self.ax.plot([], [], "-", color="0.20", lw=1.1, zorder=5.2,
                                     visible=False, clip_on=False)
                self.ax.add_patch(track)
                self.ax.add_patch(fill)
                self._responding_bars[meta.vessel_id] = (track, fill, tick, key)

        # Goal markers — a hollow star at the edge of the view, on each vessel's
        # nominal heading ray from its start (goals given as (north, east)).
        # Placed on the plot border rather than at the true (possibly far-off)
        # goal point, so it reads as "this vessel is bound that way". Drawn
        # instead of the dotted nominal trajectory to keep paper figures clean.
        self._goals = goals or {}
        if self._goals:
            (xmin, xmax) = self.ax.get_xlim()
            (ymin, ymax) = self.ax.get_ylim()
            mx, my = (xmax - xmin) * 0.045, (ymax - ymin) * 0.045
            for idx, meta in enumerate(sim_info.vessels):
                g = self._goals.get(meta.vessel_id)
                if g is None:
                    continue
                gn, ge = g
                e0, n0 = meta.initial_state.east, meta.initial_state.north
                de, dn = ge - e0, gn - n0
                if abs(de) < 1e-9 and abs(dn) < 1e-9:
                    continue
                ts: list[float] = []
                if de > 0:   ts.append(((xmax - mx) - e0) / de)
                elif de < 0: ts.append(((xmin + mx) - e0) / de)
                if dn > 0:   ts.append(((ymax - my) - n0) / dn)
                elif dn < 0: ts.append(((ymin + my) - n0) / dn)
                ts = [t for t in ts if t > 0]
                if not ts:
                    continue
                t = min(ts)
                color = VESSEL_COLORS[idx % len(VESSEL_COLORS)]
                self.ax.plot([e0 + t * de], [n0 + t * dn], marker="*",
                             markersize=18, mfc="none", mec=color, mew=1.8,
                             ls="none", zorder=3, clip_on=False)

        # Prediction trajectory lines — dashed, same colour, lower alpha.
        # With pred_top_k > 1 each vessel gets K lines: index 0 is the
        # most-likely hypothesis bundle, the rest are the next-most-likely
        # bundles drawn progressively fainter and thinner so their rank
        # (Top-1, Top-2, ... = most- to least-likely hypothesis r^(m)) reads
        # off the map directly.
        self._pred_fn = pred_fn
        self._pred_top_k = max(1, pred_top_k)

        def _pred_rank_style(j: int) -> tuple[float, float, int]:
            """(alpha, linewidth, zorder) for hypothesis rank j (0 = most likely).

            Fixed decrement per rank (not normalised over K) so the Top-2 bundle
            stays clearly visible whether K is 2 or 5, with a floor that keeps
            even the lowest-ranked bundle readable.
            """
            if self._pred_top_k == 1:
                return 0.75, 3.0, 3
            alpha = 0.90 if j == 0 else 0.30           # Top-1 bold, alternates faint
            lw    = max(1.4,  3.4 - 0.7 * j)           # 3.4, 2.7, 2.0, 1.4 ...
            return alpha, lw, (3 if j == 0 else 2)

        self._pred_rank_style = _pred_rank_style
        self._pred_lines: dict[str, list[Line2D]] = {}
        if pred_fn:
            for idx, meta in enumerate(sim_info.vessels):
                color = VESSEL_COLORS[idx % len(VESSEL_COLORS)]
                lines: list[Line2D] = []
                for j in range(self._pred_top_k):
                    alpha, lw, z = _pred_rank_style(j)
                    ln, = self.ax.plot([], [], "--", color=color,
                                       alpha=alpha, lw=lw, zorder=z)
                    lines.append(ln)
                self._pred_lines[meta.vessel_id] = lines

        # Nominal trajectory lines — dotted, same colour, lower alpha
        # Planned tracks (plan_fn) — dash-dot in the vessel colour, with a
        # marker and label at each manoeuvre of the plan.
        self._plan_fn = plan_fn
        self._plan_lines: dict[str, Line2D] = {}
        self._plan_marks: dict[str, Line2D] = {}
        self._plan_texts: list = []
        if plan_fn:
            for idx, meta in enumerate(sim_info.vessels):
                color = VESSEL_COLORS[idx % len(VESSEL_COLORS)]
                line, = self.ax.plot([], [], "-.", color=color, alpha=0.9, lw=2.2, zorder=4)
                marks, = self.ax.plot([], [], "o", mfc="white", mec=color, mew=2.0,
                                      ms=7, ls="none", zorder=5)
                self._plan_lines[meta.vessel_id] = line
                self._plan_marks[meta.vessel_id] = marks

        self._nominal_pred_fn = nominal_pred_fn
        self._nominal_pred_lines: dict[str, Line2D] = {}
        if nominal_pred_fn:
            for idx, meta in enumerate(sim_info.vessels):
                color = VESSEL_COLORS[idx % len(VESSEL_COLORS)]
                line, = self.ax.plot([], [], ":", color=color,
                                     alpha=0.55, lw=3.0, zorder=3)
                self._nominal_pred_lines[meta.vessel_id] = line

        # Future ground-truth trajectory lines — solid, like the history
        # trail (the common convention in the literature), but at a
        # noticeably lower alpha so "already happened" (history, opaque)
        # stays visually distinct from "shown for reference only" (future).
        # Drawn in grey rather than the vessel colour so it doesn't read as
        # the same colour as that vessel's dashed prediction sitting next to
        # it — the trail (past) stays vessel-coloured for identification.
        self._future_alpha = 0.45
        self._future_lines: dict[str, Line2D] = {}
        if show_future_traj:
            for meta in sim_info.vessels:
                line, = self.ax.plot([], [], "-", color="gray",
                                     alpha=self._future_alpha, lw=3.0, zorder=2)
                self._future_lines[meta.vessel_id] = line

        # Time overlay (small, top-left corner of main map)
        self.time_text = self.ax.text(
            0.02, 0.97, "",
            transform=self.ax.transAxes,
            fontsize=FONT_SIZE, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
        )

        # Legend — a per-vessel colored entry is only included when it carries
        # real information. Per-vessel entries are intentionally omitted: the
        # V1/V2/... labels already sit on the map next to each vessel, and
        # non-responsive vessels are flagged there directly (see VesselArtist),
        # so the legend only carries the line-style key below.
        legend_elements: list[Line2D] = []
        # With pred_top_k == 1 there's only ever one hypothesis, so it's named
        # consistently with the multi-hypothesis "Top-k" entries below.
        pred_label = "Top-1 prediction"
        legend_elements.append(
            Line2D([0], [0], ls="-", color="gray", lw=3.0,
                   label="Observed" if show_future_traj else "Trajectory")
        )
        if show_future_traj:
            legend_elements.append(
                Line2D([0], [0], ls="-", color="gray", lw=3.0,
                       alpha=self._future_alpha, label="Ground Truth")
            )
        # Indices into legend_elements of the K prediction entries, so their
        # labels can be refreshed with the live hypothesis probabilities.
        self._pred_legend_idx: list[int] = []
        if pred_fn:
            if self._pred_top_k > 1:
                for j in range(self._pred_top_k):
                    a, lw, _z = self._pred_rank_style(j)
                    self._pred_legend_idx.append(len(legend_elements))
                    legend_elements.append(
                        Line2D([0], [0], ls="--", color="gray", lw=lw, alpha=a,
                               label=rf"Top-${j+1}$: $p(\mathbf{{r}})$")
                    )
            else:
                self._pred_legend_idx.append(len(legend_elements))
                legend_elements.append(
                    Line2D([0], [0], ls="--", color="gray", lw=3.0, label=pred_label)
                )
            if nominal_pred_fn:
                legend_elements.append(
                    Line2D([0], [0], ls=":", color="gray", lw=3.0, label="Nominal prediction")
                )
        if plan_fn:
            legend_elements.append(
                Line2D([0], [0], ls="-.", color="gray", lw=2.2, marker="o", mfc="white",
                       mec="gray", label="Search plan")
            )
        if self._goals:
            legend_elements.append(
                Line2D([0], [0], marker="*", mfc="none", mec="gray", mew=1.8,
                       ls="none", markersize=12, label="Goal")
            )
        if self._responding_bars:
            legend_elements.append(
                Patch(facecolor=self._resp_cmap(1.0), edgecolor="0.45", lw=0.8,
                      label=r"$p(r = R)$")
            )
        _legend_loc = self.vc.legend_loc.lower().replace("_", "-")
        _top_legend = _legend_loc in ("top", "top-bar", "bar")
        self._legend = None
        if _legend_loc in ("off", "none", "hidden"):
            pass
        elif _top_legend:
            # Flat horizontal bar inside the map, hugging the top edge (like
            # panel-bottom in the multi-panel figure, just at the top instead).
            self._legend = self.ax.legend(
                handles=legend_elements, loc="upper right",
                ncol=min(2, len(legend_elements)),
                fontsize=FONT_SIZE, frameon=True, fancybox=False, edgecolor="0.4",
                framealpha=0.85, columnspacing=1.4, handlelength=2.0,
                handletextpad=0.6,
            )
        else:
            self._legend = self.ax.legend(
                handles=legend_elements, loc="upper right", fontsize=FONT_SIZE
            )

        # Live-updatable Text handles for the K prediction legend entries
        # (labels get the current hypothesis probability appended each frame).
        self._pred_legend_texts: list = []
        if self._legend is not None and self._pred_legend_idx:
            _texts = self._legend.get_texts()
            self._pred_legend_texts = [
                _texts[i] for i in self._pred_legend_idx if i < len(_texts)
            ]

        # tight_layout() does not correctly reflow nested GridSpecs (the right
        # column's stacked panels), so it can shift the main map relative to
        # them and break the equal/full-height layout computed above — use
        # fixed margins instead.
        if n_right == 0:
            plt.tight_layout()
        else:
            self.fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.09)

        # Responding-under-graph panel has no y-axis label or ticks (the legend
        # title names the quantity; the per-curve value labels are the readout),
        # so it can span the column's full width, edges flush with the
        # Interaction Graph above it — no repositioning needed.
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Prediction throttle state
        self._pred_tick  = 0
        self._pred_every = max(1, pred_every)

    # ------------------------------------------------------------------
    # Keyboard handler
    # ------------------------------------------------------------------

    def _stop(self) -> None:
        self._stopped = True

    def pause(self) -> None:
        """Pause playback. Safe to call from the driving script, e.g. to
        halt playback at a scripted stop-time."""
        self._paused = True

    def resume(self) -> None:
        """Resume playback."""
        self._paused = False

    def _on_key(self, event) -> None:
        if event.key in ("q", "escape"):
            self._stop()
        elif event.key == " ":
            self.resume() if self._paused else self.pause()
        elif event.key in ("s", "S"):
            self._save_frame_pdf()

    def _save_frame_pdf(self) -> None:
        """Save the current frame — the same content that's on screen
        (including the corner-inset layout, when vc.cg_inset is on: that's
        a real layout switch, not a save-time trick, so there's nothing
        special to do here) — to vc.frame_format (default "pdf")."""
        frames_dir = (
            Path(self.vc.frames_dir) if self.vc.frames_dir
            else Path(__file__).resolve().parents[2] / "outputs" / "frames"
        )
        frames_dir.mkdir(parents=True, exist_ok=True)
        slug      = re.sub(r"[^\w\-]+", "_", self.sim_info.name).strip("_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fmt       = self.vc.frame_format.lower()
        path      = frames_dir / f"{slug}_step{self._last_step:04d}_{timestamp}.{fmt}"
        save_kwargs = {"dpi": self.vc.frame_dpi} if fmt == "png" else {}
        self.fig.savefig(path, bbox_inches="tight", **save_kwargs)
        print(f"Saved frame -> {path}")

    # ------------------------------------------------------------------
    # Inline p(responding) belief bars
    # ------------------------------------------------------------------

    def _resp_history_key(self, meta: "VesselMeta") -> Optional[str]:
        """Return the key under which `meta`'s responding history is stored
        (the raw dict may be keyed by internal id or display label), or None."""
        hist = self._responding_history_raw
        for key in (meta.vessel_id, getattr(meta, "display_id", None), meta.label):
            if key and hist.get(key):
                return key
        return None

    def _update_responding_bars(self, states: dict[str, VesselState], k: int) -> None:
        if not self._responding_bars:
            return
        vw = self.vc.vessel_width
        bar_len = vw * 2.2          # bar length [m]
        bar_h   = vw * 0.5          # bar height [m]
        for vid, (track, fill, tick, key) in self._responding_bars.items():
            state  = states.get(vid)
            series = self._responding_history_raw.get(key) or []
            if state is None or not series:
                for a in (track, fill, tick):
                    a.set_visible(False)
                continue
            p = float(np.clip(series[min(k, len(series) - 1)], 0.0, 1.0))

            # Sit directly below the V-label (same offset convention as
            # VesselArtist) so it stays visually tied to the label without
            # crossing the hull. clip_on=False keeps it visible near the edge.
            off   = vw * self.vc.label_offset
            lab_e = state.east  + np.cos(state.heading) * off
            lab_n = state.north - np.sin(state.heading) * off
            cx    = lab_e
            cy    = lab_n - (bar_h / 2 + vw * 1.1)
            x0, y0 = cx - bar_len / 2, cy - bar_h / 2

            col = self._resp_cmap(p)
            track.set_bounds(x0, y0, bar_len, bar_h)
            track.set_edgecolor(col)
            fill.set_bounds(x0, y0, bar_len * max(p, 0.04), bar_h)
            fill.set_facecolor(col)
            # fixed reference notch at p = 0.5
            mx = x0 + bar_len * 0.5
            tick.set_data([mx, mx], [y0 - bar_h * 0.3, y0 + bar_h * 1.3])
            for a in (track, fill, tick):
                a.set_visible(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        states:        dict[str, VesselState],
        k:             int,
        debug_frames:  Optional[dict[str, Optional[PolicyDebugFrame]]] = None,
        colregs_graph: Optional[ColregsGraph]                          = None,
        future_trajs:  Optional[dict[str, list[VesselState]]]          = None,
    ) -> None:
        self._last_step = k
        t = k * self.sim_info.dt

        for vid, state in states.items():
            if vid in self.artists:
                self.artists[vid].update(state)

        self._update_responding_bars(states, k)

        # Throttled prediction: recompute only every _pred_every frames
        if self._pred_fn or self._nominal_pred_fn:
            self._pred_tick += 1
            if self._pred_tick >= self._pred_every:
                self._pred_tick = 0
                if self._pred_fn:
                    predictions = self._pred_fn(states, k)
                    # Normalize to (bundles, weights). pred_fn returns a bare
                    # per-vessel-traj dict when pred_top_k == 1, or a list of
                    # (traj_dict, weight) tuples (most-likely first) when > 1.
                    if isinstance(predictions, list):
                        bundles = [item[0] for item in predictions]
                        weights = [item[1] for item in predictions]
                    else:
                        bundles, weights = [predictions], [None]
                    for vid, lines in self._pred_lines.items():
                        for j, line in enumerate(lines):
                            pred = bundles[j].get(vid) if j < len(bundles) else None
                            if pred:
                                line.set_data([s.east for s in pred],
                                              [s.north for s in pred])
                            else:
                                line.set_data([], [])
                    # Refresh the K prediction legend labels with live p(hyp).
                    if self._pred_top_k > 1:
                        for j, txt in enumerate(self._pred_legend_texts):
                            w = weights[j] if j < len(weights) else None
                            n = j + 1
                            if w is None:
                                txt.set_text(rf"Top-${n}$: $p(\mathbf{{r}})$")
                            elif w >= 0.01:
                                txt.set_text(rf"Top-${n}$: $p(\mathbf{{r}}) = {w:.2f}$")
                            elif w > 0:
                                txt.set_text(rf"Top-${n}$: $p(\mathbf{{r}}) < 0.01$")
                            else:
                                txt.set_text(rf"Top-${n}$: $p(\mathbf{{r}}) \approx 0$")
                if self._nominal_pred_fn:
                    nominal_preds = self._nominal_pred_fn(states, k)
                    for vid, line in self._nominal_pred_lines.items():
                        if vid in nominal_preds:
                            pred     = nominal_preds[vid]
                            e_coords = [s.east  for s in pred]
                            n_coords = [s.north for s in pred]
                            line.set_data(e_coords, n_coords)

        if self._plan_fn:
            plans = self._plan_fn(k) or {}
            for txt in self._plan_texts:
                txt.remove()
            self._plan_texts = []
            for vid, line in self._plan_lines.items():
                plan = plans.get(vid)
                marks = self._plan_marks[vid]
                if plan is None:
                    line.set_data([], [])
                    marks.set_data([], [])
                    continue
                line.set_data(plan.east, plan.north)
                marks.set_data([m[1] for m in plan.marks], [m[0] for m in plan.marks])
                label_size = FONT_SIZE * 0.7
                # Labels go to port of the planned track: the vessel's own
                # label sits to starboard (VesselArtist).
                d_e, d_n = plan.east[-1] - plan.east[0], plan.north[-1] - plan.north[0]
                norm = np.hypot(d_e, d_n)
                port_e, port_n = (-d_n / norm, d_e / norm) if norm > 1e-6 else (-1.0, 0.0)
                for n, e, label in _merge_plan_marks(plan.marks, self.ax, 2.0 * label_size):
                    self._plan_texts.append(self.ax.annotate(
                        label, (e, n), xytext=(8 * port_e, 8 * port_n), textcoords="offset points",
                        ha="right" if port_e < -0.3 else "left" if port_e > 0.3 else "center",
                        va="top" if port_n < -0.3 else "bottom" if port_n > 0.3 else "center",
                        fontsize=label_size, color=line.get_color(), zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                    ))

        if self._future_lines:
            for vid, line in self._future_lines.items():
                if future_trajs and vid in future_trajs:
                    traj = future_trajs[vid]
                    line.set_data([s.east for s in traj], [s.north for s in traj])
                else:
                    line.set_data([], [])

        self.time_text.set_text(rf"$t = {t:.1f}\ \mathrm{{s}}$")

        if debug_frames:
            for vid, frame in debug_frames.items():
                if vid in self.debug_axes and frame is not None:
                    color = (self.artists[vid].color
                             if vid in self.artists else "royalblue")
                    _render_debug(self.debug_axes[vid], frame, color, vid)

        if colregs_graph is not None and self.colregs_ax is not None:
            _draw_colregs_graph(
                self.colregs_ax, states, colregs_graph,
                self._cg_idx_to_id, self._cg_idx_to_color,
                t, self.vc, scale=self._cg_inset_scale,
                idx_to_label=self._cg_idx_to_label,
                title=("Top-1 Interaction Graph"
                       if self._panel_titles and not self._cg_inset else None),
            )

        if self._responding_ax is not None and self._responding_history:
            _draw_responding_panel(
                self._responding_ax,
                self._responding_history,
                self._vessel_colors,
                k,
                self.sim_info.dt,
                self.sim_info.n_steps,
                window=self.vc.responding_plot_window,
                title="Responding Posterior" if self._panel_titles else None,
                t0=self._responding_history_t0,
            )

        if self._resp_under_ax is not None and self._responding_history_raw:
            _draw_responding_panel(
                self._resp_under_ax,
                self._responding_history_raw,
                self._vessel_colors,
                k,
                self.sim_info.dt,
                self.sim_info.n_steps,
                window=self.vc.responding_plot_window,
                t0=self._responding_history_t0,
                compact=True,
            )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self._video_writer is not None:
            self._video_writer.grab_frame()

    def show(self) -> None:
        plt.ion()
        plt.show()

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)

    # ------------------------------------------------------------------
    # Live video recording — grabs a frame from the figure canvas on every
    # step() call, so a quit/interrupt mid-playback still leaves a valid
    # video of whatever was shown up to that point (finalized in stop_recording).
    # ------------------------------------------------------------------

    def start_recording(self, path: str, fps: int) -> None:
        # .gif -> Pillow (no ffmpeg needed); anything else -> ffmpeg (mp4 etc.).
        from matplotlib.animation import FFMpegWriter, PillowWriter
        writer = PillowWriter(fps=fps) if path.lower().endswith(".gif") else FFMpegWriter(fps=fps)
        writer.setup(self.fig, path, dpi=120)
        self._video_writer = writer

    def stop_recording(self) -> None:
        if self._video_writer is not None:
            self._video_writer.finish()
            self._video_writer = None

    def save_animation(
        self,
        trajectories:   dict[str, list[VesselState]],
        path:           str,
        fps:            Optional[int]              = None,
        policy_debug:   Optional[dict[str, list]] = None,
        colregs_graphs: Optional[list[ColregsGraph]] = None,
        frame_dt:       Optional[float]            = None,
    ) -> None:
        """Save the playback to *path*. A .gif plays at the live view's pace:
        each frame lasts as long as it took to compute and draw plus the live
        loop's pause *frame_dt* (dt / realtime_factor). Other formats play at
        a fixed *fps* (default vc.fps)."""
        fps = fps or self.vc.fps

        def _update(k):
            states = {vid: traj[k] for vid, traj in trajectories.items()}
            debug_this: Optional[dict] = None
            if policy_debug:
                debug_this = {
                    vid: (frames[min(k, len(frames) - 1)] if frames else None)
                    for vid, frames in policy_debug.items()
                }
            graph_this = (
                colregs_graphs[min(k, len(colregs_graphs) - 1)]
                if colregs_graphs else None
            )
            self.step(states, k, debug_this, colregs_graph=graph_this)
            return []

        if path.lower().endswith(".gif") and frame_dt is not None:
            import io
            from PIL import Image
            frames, durations_ms = [], []
            for k in range(self.sim_info.n_steps + 1):
                t_start = time.perf_counter()
                _update(k)
                self.fig.canvas.draw()
                durations_ms.append(1000.0 * (time.perf_counter() - t_start + frame_dt))
                buf = io.BytesIO()
                self.fig.savefig(buf, format="png", dpi=120)
                frames.append(Image.open(buf).convert("RGB"))
            frames[0].save(path, save_all=True, append_images=frames[1:],
                           duration=[max(20, round(d)) for d in durations_ms], loop=0)
            print(f"Animation saved to {path}  "
                  f"(live pace, {sum(durations_ms) / 1000:.1f} s total)")
            return

        anim = FuncAnimation(
            self.fig, _update,
            frames=self.sim_info.n_steps + 1,
            interval=int(1000 / fps),
            blit=False,
        )
        anim.save(path, fps=fps, dpi=120)
        print(f"Animation saved to {path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_limits(sim_info: SimInfo, padding: float,
                     center_ego_x: bool = False) -> tuple[tuple, tuple]:
        T     = sim_info.n_steps * sim_info.dt
        all_n = [m.initial_state.north for m in sim_info.vessels]
        all_e = [m.initial_state.east  for m in sim_info.vessels]
        for m in sim_info.vessels:
            all_n.append(m.initial_state.north + m.initial_state.vn * T)
            all_e.append(m.initial_state.east  + m.initial_state.ve * T)

        n_min, n_max = min(all_n) - padding, max(all_n) + padding
        e_min, e_max = min(all_e) - padding, max(all_e) + padding

        n_range = n_max - n_min
        e_range = e_max - e_min
        if n_range > e_range:
            diff = (n_range - e_range) / 2
            e_min -= diff; e_max += diff
        else:
            diff = (e_range - n_range) / 2
            n_min -= diff; n_max += diff

        # Optionally recenter the horizontal (east) axis on the ego vessel's
        # initial position, keeping the width derived above.
        if center_ego_x and sim_info.vessels:
            ego_e = sim_info.vessels[0].initial_state.east
            half_w = (e_max - e_min) / 2
            e_min, e_max = ego_e - half_w, ego_e + half_w

        return (e_min, e_max), (n_min, n_max)


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_live(
    sim_info:           SimInfo,
    trajectories:       dict[str, list[VesselState]],
    vc:                 Optional[VizConfig]                  = None,
    speed:              Optional[float]                      = None,
    save_path:          Optional[str]                        = None,
    policy_debug:       Optional[dict[str, list]]            = None,
    debug_vessel_ids:   Optional[list[str]]                  = None,
    pred_fn:            Optional[Callable[[dict, int], dict]] = None,
    nominal_pred_fn:    Optional[Callable[[dict, int], dict]] = None,
    pred_every:         int                                   = 1,
    pred_top_k:         int                                   = 1,
    plan_fn:            Optional[Callable[[int], dict[str, PlanOverlay]]] = None,
    colregs_graphs:     Optional[list[ColregsGraph]]         = None,
    responding_history: Optional[dict[str, list[float]]]     = None,
    stop_time:          Optional[float]                      = None,
    goals:              Optional[dict[str, tuple[float, float]]] = None,
) -> None:
    vc        = vc or VizConfig.default()
    speed     = speed     or vc.realtime_factor
    save_path = save_path or vc.save_path
    stop_time = stop_time if stop_time is not None else vc.stop_time_s

    # Goal markers replace the nominal trajectory line — don't draw both.
    if goals:
        nominal_pred_fn = None

    show_cg = bool(colregs_graphs) and vc.colregs_graph_viz
    viz = LiveVisualizer(sim_info, vc, debug_vessel_ids=debug_vessel_ids,
                         pred_fn=pred_fn, nominal_pred_fn=nominal_pred_fn,
                         pred_every=pred_every, pred_top_k=pred_top_k,
                         plan_fn=plan_fn,
                         show_colregs_graph=show_cg,
                         responding_history=responding_history,
                         goals=goals)

    if save_path:
        viz.save_animation(trajectories, save_path,
                           policy_debug=policy_debug,
                           colregs_graphs=colregs_graphs if show_cg else None,
                           frame_dt=sim_info.dt / speed)
        return

    viz.show()

    # Recording window: viz.video.start_time_s / end_time_s [s] restrict which
    # frames get grabbed into the video (None on either side = unrestricted on
    # that side). Live playback itself always plays the full 0..n_steps range
    # regardless — only the video is trimmed.
    dt = sim_info.dt
    video_start_step = (0 if vc.video_start_time_s is None
                        else max(0, min(round(vc.video_start_time_s / dt), sim_info.n_steps)))
    video_end_step   = (sim_info.n_steps if vc.video_end_time_s is None
                        else max(0, min(round(vc.video_end_time_s   / dt), sim_info.n_steps)))
    if video_end_step < video_start_step:
        video_start_step, video_end_step = video_end_step, video_start_step
    if vc.video_enabled and (vc.video_start_time_s is not None or vc.video_end_time_s is not None):
        print(f"  Video    : recording restricted to t = {video_start_step * dt:.1f}.."
              f"{video_end_step * dt:.1f} s (frames {video_start_step}..{video_end_step})")

    video_path:   Optional[Path] = None
    video_active: bool           = False
    video_done:   bool           = False

    def _start_recording(k: int) -> None:
        nonlocal video_path, video_active
        videos_dir = Path(__file__).resolve().parents[2] / "outputs" / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        slug      = re.sub(r"[^\w\-]+", "_", sim_info.name).strip("_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = videos_dir / f"{slug}_{timestamp}.mp4"
        try:
            viz.start_recording(str(video_path), fps=vc.video_fps)
            print(f"  Recording video -> {video_path}  (fps={vc.video_fps})")
            video_active = True
        except Exception as exc:
            print(f"  Warning: could not start video recording: {exc}")
            video_path = None

    frame_dt = sim_info.dt / speed

    def _get_debug(k: int) -> Optional[dict]:
        if not policy_debug:
            return None
        return {
            vid: (frames[min(k, len(frames) - 1)] if frames else None)
            for vid, frames in policy_debug.items()
        }

    if stop_time is not None:
        print(f"  Stop-at  : t = {stop_time:.1f} s "
              f"(auto-pause + save {vc.frame_format})")

    k = 0
    stop_time_reached = False
    try:
        while k <= sim_info.n_steps and not viz._stopped:
            if not viz._paused:
                states  = {vid: traj[k] for vid, traj in trajectories.items()}
                graph_k = (colregs_graphs[min(k, len(colregs_graphs) - 1)]
                           if show_cg and colregs_graphs else None)

                if vc.video_enabled and not video_active and not video_done and k >= video_start_step:
                    _start_recording(k)

                viz.step(states, k, _get_debug(k), colregs_graph=graph_k)

                if video_active and k >= video_end_step:
                    viz.stop_recording()
                    print(f"  Video saved: {video_path}")
                    video_active = False
                    video_done   = True

                if (stop_time is not None and not stop_time_reached
                        and k * sim_info.dt >= stop_time):
                    stop_time_reached = True
                    viz.pause()
                    viz._save_frame_pdf()
                    print(f"  Reached stop-time t={k * sim_info.dt:.1f} s "
                          f"(step {k}) — paused, frame saved.")

                k += 1
                plt.pause(frame_dt)
            else:
                plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        if video_path is not None and not video_done:
            viz.stop_recording()
            print(f"  Video saved: {video_path}")
        if not viz._stopped:
            plt.show()
