"""visualization/map_background.py

Renders a ChartBackground (sea chart geometry in NED metres) onto a
matplotlib Axes.  Call render_chart_background(ax, bg) once after the axis
is set up — matplotlib's zorder ensures chart features sit behind vessels
regardless of the add order.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon as MplPolygon

from src.seachart.loader import ChartBackground

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

CHART_STYLE: dict[str, dict] = {
    "land": {
        "facecolor": "#e8dcc8",
        "edgecolor": "#8b7355",
        "linewidth": 0.6,
        "alpha":     0.90,
        "zorder":    1,
    },
    "shoal": {
        "facecolor": "#f5c842",
        "edgecolor": "#c49a00",
        "linewidth": 0.5,
        "alpha":     0.70,
        "zorder":    2,
    },
    "coastline": {
        "color":     "#4a3728",
        "linewidth": 0.9,
        "alpha":     1.0,
        "zorder":    3,
    },
    "depth_contour": {
        "color":     "#7ab8d4",
        "linewidth": 0.4,
        "alpha":     0.55,
        "zorder":    2,
    },
}

_WATER_COLOR = "#dfedf5"   # nautical chart blue for open water, lightened so
                            # trajectories/vessels stand out more clearly
_DEFAULT_BG  = "#f0f4f8"   # original live_viz facecolor (used when no chart)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def set_water_facecolor(ax: plt.Axes, color: str = _WATER_COLOR) -> None:
    """Set the axis background to the open-water colour (nautical-chart blue
    by default; pass `color` to match another background, e.g. the plain
    simulation-plot facecolor)."""
    ax.set_facecolor(color)


def render_chart_background(
    ax: plt.Axes,
    bg: ChartBackground,
    show_depth_contours: bool = False,
) -> None:
    """Draw all ChartBackground geometry onto a matplotlib Axes.

    Should be called once after the axis limits are set and before the
    playback loop starts.  All artists use low zorder values so they render
    behind vessel polygons (which default to zorder 4+).

    Parameters
    ----------
    ax : NED map axes (x=east [m], y=north [m], equal-aspect)
    bg : populated ChartBackground from load_chart_background()
    show_depth_contours : draw depth-contour lines (default False — clutters the water)
    """
    if bg.is_empty():
        return

    s = CHART_STYLE

    # Land polygons
    for arr in bg.land_polys:
        st = s["land"]
        ax.add_patch(MplPolygon(
            arr, closed=True,
            facecolor=st["facecolor"],
            edgecolor=st["edgecolor"],
            linewidth=st["linewidth"],
            alpha=st["alpha"],
            zorder=st["zorder"],
        ))

    # Shoal polygons
    for arr in bg.shoal_polys:
        st = s["shoal"]
        ax.add_patch(MplPolygon(
            arr, closed=True,
            facecolor=st["facecolor"],
            edgecolor=st["edgecolor"],
            linewidth=st["linewidth"],
            alpha=st["alpha"],
            zorder=st["zorder"],
        ))

    # Coastline — LineCollection for performance
    if bg.coastline_segs:
        st = s["coastline"]
        ax.add_collection(LineCollection(
            bg.coastline_segs,
            colors=st["color"],
            linewidths=st["linewidth"],
            alpha=st["alpha"],
            zorder=st["zorder"],
        ))

    # Depth contours — LineCollection for performance
    if show_depth_contours and bg.depth_contours:
        st = s["depth_contour"]
        ax.add_collection(LineCollection(
            bg.depth_contours,
            colors=st["color"],
            linewidths=st["linewidth"],
            alpha=st["alpha"],
            zorder=st["zorder"],
        ))
