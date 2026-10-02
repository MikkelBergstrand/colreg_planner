"""scripts/tools/play_encounters.py

Play back accepted AIS encounter windows with sea chart background.

Usage
-----
    python scripts/tools/play_encounters.py                        # play first accepted window
    python scripts/tools/play_encounters.py --window 260           # play window_id 260
    python scripts/tools/play_encounters.py --list                 # list accepted windows
    python scripts/tools/play_encounters.py --all                  # play all accepted in sequence
    python scripts/tools/play_encounters.py --data data/ais_test.pkl  # different data file

Controls (during playback)
--------------------------
    Space   : pause / resume
    Q / Esc : quit
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt


from src.ais.windows import traj_limits, window_to_frames
from src.vessel.state import VesselState
from src.visualization.live_viz import VesselMeta, SimInfo, VizConfig, LiveVisualizer

try:
    from src.seachart.loader import find_chart, load_chart_background
    from src.visualization.map_background import render_chart_background, set_water_facecolor
    _SEACHART_AVAILABLE = True
except ImportError:
    _SEACHART_AVAILABLE = False

_ROOT      = Path(__file__).resolve().parents[2]
_FGDB_PATH = find_chart() if _SEACHART_AVAILABLE else None
_DEFAULT_DATA   = _ROOT / "handover" / "Moss Horten Feb 2026 AIS" / "processed" / "ais_04.pkl"
_DEFAULT_REVIEW = _ROOT / "data" / "Moss Horten July 2025 AIS" / "processed" / "encounter_review.json"


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Visualizer helpers
# ---------------------------------------------------------------------------

def _build_sim_info(w: dict, frames: list[dict[str, VesselState]]) -> SimInfo:
    m = w["meta"]
    sorted_vids = sorted(frames[0].keys())
    vessel_metas = [
        VesselMeta(
            vessel_id=f"V{i+1}",
            initial_state=frames[0][vid],
            intent_label=f"MMSI {vid}",
        )
        for i, vid in enumerate(sorted_vids)
    ]
    return SimInfo(
        name=(
            f"AIS win={w['window_id']}  {m['scenario'].upper()}"
            f"  enc_CPA={m['enc_cpa_m']:.0f} m  TCPA={m.get('tcpa_s', float('inf')):.0f} s"
        ),
        n_steps=len(frames) - 1,
        dt=w["dt"],
        vessels=vessel_metas,
    )


def _build_viz_config(realtime_factor: float) -> VizConfig:
    return VizConfig(
        figsize=(11, 10),
        realtime_factor=realtime_factor,
        trail_length=40,
        trail_alpha=0.55,
        vessel_length=30.0,
        vessel_width=8.0,
        bow_fraction=0.25,
        vessel_alpha=0.85,
        debug_viz=False,
        colregs_graph_viz=False,
    )


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def play(w: dict, realtime_factor: float = 60.0) -> None:
    frames = window_to_frames(w)
    if len(frames) < 2:
        print(f"Window {w['window_id']}: not enough frames, skipping.")
        return

    m = w["meta"]
    ref_lat, ref_lon = w["ref_lat"], w["ref_lon"]
    dt = w["dt"]

    print(f"\n{'=' * 64}")
    print(f"  Window   : {w['window_id']}")
    print(f"  Scenario : {m['scenario'].upper()}")
    print(f"  enc_CPA  : {m['enc_cpa_m']:.0f} m   TCPA : {m.get('tcpa_s', float('inf')):.0f} s")
    print(f"  Frames   : {len(frames)}  ({len(frames)*dt/60:.1f} min at {dt:.0f} s/step)")
    print(f"  Ref pt   : {ref_lat:.5f} N  {ref_lon:.5f} E")
    print(f"  Vessels  :")
    sorted_mmsi = sorted(w["vessels"].keys())
    vid_map = {str(mmsi): f"V{i+1}" for i, mmsi in enumerate(sorted_mmsi)}
    for i, mmsi in enumerate(sorted_mmsi):
        df = w["vessels"][mmsi]
        print(f"    V{i+1}  MMSI {mmsi}  SOG {df['sog'].min():.1f}–{df['sog'].max():.1f} kn")
    print(f"{'=' * 64}")
    print(f"  Controls : SPACE = pause/resume   Q = quit\n")

    xlim, ylim = traj_limits(frames)
    sim_info   = _build_sim_info(w, frames)
    vc         = _build_viz_config(realtime_factor)
    frame_pause = dt / realtime_factor

    chart_bg = None
    if _SEACHART_AVAILABLE and _FGDB_PATH is not None:
        print("  Loading sea chart ...")
        try:
            chart_bg = load_chart_background(_FGDB_PATH, ref_lat, ref_lon, xlim, ylim)
            if not chart_bg.is_empty():
                counts = ", ".join(f"{r}={n}" for r, n in chart_bg.layer_counts.items())
                print(f"  Chart loaded: {counts}")
            else:
                print("  Chart: no features in this viewport.")
        except Exception as exc:
            print(f"  Warning: sea chart load failed: {exc}")

    viz = LiveVisualizer(sim_info, vc, xlim=xlim, ylim=ylim, is_ais=True)
    if chart_bg is not None and not chart_bg.is_empty():
        set_water_facecolor(viz.ax)
        render_chart_background(viz.ax, chart_bg)
    viz.show()

    k = 0
    try:
        while k <= sim_info.n_steps and not viz._stopped:
            if not viz._paused:
                states = {vid_map[mmsi_str]: state
                          for mmsi_str, state in frames[k].items()
                          if mmsi_str in vid_map}
                viz.step(states, k)
                k += 1
                plt.pause(frame_pause)
            else:
                plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        plt.ioff()
        if not viz._stopped:
            plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play accepted AIS encounter windows with sea chart background."
    )
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA,
                        help=f"Encounter pickle (default {_DEFAULT_DATA.name})")
    parser.add_argument("--review", type=Path, default=_DEFAULT_REVIEW,
                        help="Review JSON from browse_encounters.py")
    parser.add_argument("--window", type=int, default=None,
                        help="Play a specific window_id (ignores --review filter)")
    parser.add_argument("--all", action="store_true",
                        help="Play all accepted windows in sequence")
    parser.add_argument("--list", action="store_true",
                        help="List accepted windows and exit")
    parser.add_argument("--speed", type=float, default=1000.0,
                        help="Realtime factor: frame_pause = dt / speed (default 60)")
    args = parser.parse_args()

    data_path = args.data if args.data.exists() else _ROOT / "data" / "ais_test.pkl"
    print(f"Loading {data_path} ...")
    with open(data_path, "rb") as f:
        all_windows: list[dict] = pickle.load(f)
    print(f"Loaded {len(all_windows)} windows.")

    # Load review
    review: dict[str, str] = {}
    if args.review.exists():
        with open(args.review) as f:
            review = json.load(f)

    if args.window is not None:
        matches = [w for w in all_windows if w["window_id"] == args.window]
        if not matches:
            print(f"Window {args.window} not found.")
            sys.exit(1)
        play(matches[0], realtime_factor=args.speed)
        return

    # Filter to accepted
    accepted = [w for w in all_windows if review.get(str(w["window_id"])) == "accepted"]
    if not accepted:
        print("No accepted windows found. Use browse_encounters.py to accept some first.")
        sys.exit(1)

    if args.list:
        print(f"\n{'ID':>6}  {'Scenario':<12}  {'enc_CPA':>8}  {'Vessels':>7}  {'Duration':>8}")
        print("-" * 55)
        for w in accepted:
            m = w["meta"]
            dur = len(next(iter(w["vessels"].values()))) * w["dt"] / 60
            print(f"{w['window_id']:>6}  {m['scenario']:<12}  {m['enc_cpa_m']:>7.0f}m"
                  f"  {m['n_vessels']:>7}  {dur:>6.1f} min")
        return

    if args.all:
        for i, w in enumerate(accepted):
            print(f"\n[{i+1}/{len(accepted)}] Playing window {w['window_id']} ...")
            play(w, realtime_factor=args.speed)
    else:
        play(accepted[0], realtime_factor=args.speed)


if __name__ == "__main__":
    main()
