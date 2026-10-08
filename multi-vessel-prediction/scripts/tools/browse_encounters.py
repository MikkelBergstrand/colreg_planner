"""scripts/tools/browse_encounters.py

Systematic review interface for AIS encounter windows produced by process_raw_ais.py.

Loads encounter windows, sorts them by an interestingness score, and displays
them one at a time in a matplotlib figure.  Review state (accepted / rejected)
is saved to a JSON file so sessions can be resumed.

Controls
--------
    n / Right  : next window
    p / Left   : previous window
    a          : mark as accepted  (green border)
    r          : mark as rejected  (red border)
    u          : mark as unseen (clear label)
    v          : animate in VesselPredictor visualizer (uses LiveVisualizer)
    q / Escape : quit and save

Usage
-----
    python scripts/tools/browse_encounters.py
    python scripts/tools/browse_encounters.py --data data/processed/ais_encounters.pkl
    python scripts/tools/browse_encounters.py --filter accepted   # show only accepted
    python scripts/tools/browse_encounters.py --summary           # print table and exit
    python scripts/tools/browse_encounters.py --sort cpa          # sort by CPA (smallest first)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


from src.ais.windows import window_to_frames

_DEFAULT_DATA = Path(__file__).resolve().parents[2] / "handover" / "Moss Horten Feb 2026 AIS" / "processed" / "ais_04.pkl"
_DEFAULT_REVIEW = Path(__file__).resolve().parents[2] / "data" / "Moss Horten Feb 2026 AIS" / "processed" / "encounter_review.json"

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]


# ---------------------------------------------------------------------------
# Interestingness scoring
# ---------------------------------------------------------------------------

def _interestingness(w: dict) -> float:
    """Score a window for how useful it is as a VesselPrediction test case.

    Higher = more interesting. Favours:
    - close encounter CPA (< 500 m)
    - upcoming TCPA at rollout start (30-600 s)
    - fast-moving vessels (> 5 knots)
    - low interpolation rate (real observations preferred)
    """
    meta = w["meta"]
    rs = w["rollout_start_idx"]

    enc_cpa = meta.get("enc_cpa_m", float("nan"))
    tcpa = meta.get("tcpa_s", float("nan"))

    if math.isnan(enc_cpa) or math.isnan(tcpa):
        return -1.0
    if tcpa <= 0:
        return -1.0  # encounter already past at rollout start

    # Observation quality: fraction of non-interpolated rows across all vessels
    interp_fracs = []
    for df in w["vessels"].values():
        if "interpolated" in df.columns:
            interp_fracs.append(df["interpolated"].mean())
    obs_score = 1.0 - (float(np.mean(interp_fracs)) if interp_fracs else 0.5)

    # Speed: mean SOG of all vessels at rollout start
    sog_vals = [
        float(df["sog"].iloc[rs]) for df in w["vessels"].values()
        if rs < len(df)
    ]
    mean_sog = float(np.mean(sog_vals)) if sog_vals else 0.0
    sog_score = min(mean_sog / 15.0, 1.0)  # normalise to 15 kn

    cpa_score = math.exp(-((enc_cpa - 150) / 400) ** 2)
    tcpa_score = math.exp(-((tcpa - 200) / 250) ** 2)

    return 0.4 * cpa_score + 0.25 * tcpa_score + 0.2 * sog_score + 0.15 * obs_score


def _sort_windows(windows: list[dict], sort_key: str) -> list[dict]:
    if sort_key == "score":
        scored = [(w, _interestingness(w)) for w in windows]
        scored.sort(key=lambda x: -x[1])
        return [w for w, _ in scored]
    elif sort_key == "cpa":
        return sorted(windows, key=lambda w: w["meta"].get("enc_cpa_m", float("inf")))
    elif sort_key == "tcpa":
        return sorted(windows, key=lambda w: abs(w["meta"].get("tcpa_s", float("inf"))))
    elif sort_key == "window_id":
        return sorted(windows, key=lambda w: w["window_id"])
    return windows


# ---------------------------------------------------------------------------
# Static overview plot
# ---------------------------------------------------------------------------

def _plot_window(w: dict, fig: plt.Figure, review: str) -> None:
    fig.clear()
    meta = w["meta"]
    rs = w["rollout_start_idx"]

    mmsi_list = sorted(w["vessels"].keys())
    n_steps = len(next(iter(w["vessels"].values())))
    t_steps = np.arange(n_steps) * w["dt"] / 60.0  # minutes

    pair = list(w["pairs"].keys())[0] if w["pairs"] else None

    # Border color
    border_color = {"accepted": "#2ecc71", "rejected": "#e74c3c"}.get(review, "#888888")
    fig.patch.set_edgecolor(border_color)
    fig.patch.set_linewidth(4)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    # ------------------------------------------------------------------ #
    # Panel 1: trajectory (NED)
    # ------------------------------------------------------------------ #
    ax_traj = fig.add_subplot(gs[:, 0])
    for vi, mmsi in enumerate(mmsi_list):
        df = w["vessels"][mmsi]
        color = _COLORS[vi % len(_COLORS)]
        north = df["north"].values
        east = df["east"].values
        ax_traj.plot(east, north, color=color, linewidth=1.8, label=f"MMSI {mmsi}")
        ax_traj.scatter(east, north, s=6, color=color, alpha=0.5, zorder=3)
        # Rollout start marker
        if rs < len(df):
            ax_traj.scatter(
                east[rs], north[rs], s=100, color=color, marker="D",
                zorder=6, edgecolors="black", linewidths=0.8, label=f"RS v{vi}"
            )
        # Heading arrows every 5 steps
        for t in range(0, n_steps, 5):
            if t >= len(df):
                break
            row = df.iloc[t]
            arrow_len = 100.0
            ax_traj.annotate(
                "", xy=(float(row["east"]) + float(row["sin_hdg"]) * arrow_len,
                        float(row["north"]) + float(row["cos_hdg"]) * arrow_len),
                xytext=(float(row["east"]), float(row["north"])),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0, alpha=0.7),
            )
    ax_traj.set_xlabel("East [m]", fontsize=8)
    ax_traj.set_ylabel("North [m]", fontsize=8)
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.legend(fontsize=7, loc="best")
    ax_traj.axvline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
    ax_traj.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
    ax_traj.tick_params(labelsize=7)
    ax_traj.set_title("Trajectories (NED, diamond=rollout start)", fontsize=8)

    # ------------------------------------------------------------------ #
    # Panel 2: CPA over time
    # ------------------------------------------------------------------ #
    ax_cpa = fig.add_subplot(gs[0, 1])
    if pair is not None:
        pm = w["pairs"][pair]
        cpa_t = pm["cpa_t"]
        finite = np.isfinite(cpa_t) & (cpa_t < 1e6)
        ax_cpa.plot(t_steps[finite], cpa_t[finite], color="#e67e22", linewidth=1.8)
        ax_cpa.axvline(t_steps[rs], color="black", linestyle="--", linewidth=1.0,
                       label=f"rollout start (t={t_steps[rs]:.1f} min)")
        ax_cpa.axhline(meta.get("enc_cpa_m", float("nan")), color="red",
                       linestyle=":", linewidth=0.8, label=f"enc CPA={meta.get('enc_cpa_m', 0):.0f}m")
        ax_cpa.legend(fontsize=6)
    ax_cpa.set_xlabel("Time [min]", fontsize=8)
    ax_cpa.set_ylabel("CPA [m]", fontsize=8)
    ax_cpa.set_title("CPA over time (constant velocity projection)", fontsize=8)
    ax_cpa.tick_params(labelsize=7)

    # ------------------------------------------------------------------ #
    # Panel 3: TCPA over time
    # ------------------------------------------------------------------ #
    ax_tcpa = fig.add_subplot(gs[1, 1])
    if pair is not None:
        pm = w["pairs"][pair]
        tcpa_t = pm["tcpa_t"]
        finite = np.isfinite(tcpa_t) & (tcpa_t > 0) & (tcpa_t < 1800)
        ax_tcpa.plot(t_steps[finite], tcpa_t[finite] / 60.0, color="#8e44ad", linewidth=1.8)
        ax_tcpa.axvline(t_steps[rs], color="black", linestyle="--", linewidth=1.0)
        ax_tcpa.axhline(0, color="gray", linestyle="-", linewidth=0.4)
    ax_tcpa.set_xlabel("Time [min]", fontsize=8)
    ax_tcpa.set_ylabel("TCPA [min]", fontsize=8)
    ax_tcpa.set_title("TCPA over time", fontsize=8)
    ax_tcpa.tick_params(labelsize=7)

    # ------------------------------------------------------------------ #
    # Header text
    # ------------------------------------------------------------------ #
    sog_strs = []
    for mmsi, df in w["vessels"].items():
        sog = df["sog"].values
        sog_strs.append(f"MMSI {mmsi}: SOG {sog.min():.1f}-{sog.max():.1f} kn")

    review_str = f"[{review.upper()}]" if review != "unseen" else ""
    title = (
        f"Window {w['window_id']}  {meta['scenario'].upper()}"
        f"  enc_CPA={meta.get('enc_cpa_m', 0):.0f}m"
        f"  CPA@rs={meta.get('cpa_m', 0):.0f}m"
        f"  TCPA={meta.get('tcpa_s', 0):.0f}s  {review_str}\n"
        + "  ".join(sog_strs)
    )
    fig.suptitle(title, fontsize=8, y=0.99)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary_table(windows: list[dict], review_state: dict) -> None:
    print()
    print("=" * 90)
    print("ENCOUNTER WINDOWS SUMMARY")
    print("=" * 90)
    header = f"  {'#':>4}  {'win_id':>6}  {'scenario':<12}  {'enc_CPA[m]':>10}  {'CPA_rs[m]':>9}  {'TCPA[s]':>7}  {'vessels':>7}  {'review':<10}  score"
    print(header)
    print("  " + "-" * 86)
    for i, w in enumerate(windows):
        meta = w["meta"]
        rid = str(w["window_id"])
        rev = review_state.get(rid, "unseen")
        score = _interestingness(w)
        print(
            f"  {i + 1:4d}  {w['window_id']:6d}  {meta['scenario']:<12}"
            f"  {meta.get('enc_cpa_m', float('nan')):>10.0f}  {meta.get('cpa_m', float('nan')):>9.0f}"
            f"  {meta.get('tcpa_s', float('nan')):>7.0f}  {meta['n_vessels']:>7d}  {rev:<10}  {score:.3f}"
        )
    print("=" * 90)
    labels = {"accepted": 0, "rejected": 0, "unseen": 0}
    for rid, rev in review_state.items():
        if rev in labels:
            labels[rev] += 1
    total_unseen = len(windows) - labels["accepted"] - labels["rejected"]
    print(f"\n  Accepted: {labels['accepted']}   Rejected: {labels['rejected']}"
          f"   Unseen: {total_unseen}")
    print()


# ---------------------------------------------------------------------------
# Animation via LiveVisualizer
# ---------------------------------------------------------------------------

def _animate_window(w: dict) -> None:
    """Launch the live visualizer for one window."""
    try:
        from src.visualization.live_viz import (
            VesselMeta, SimInfo, VizConfig, LiveVisualizer,
        )
    except ImportError as e:
        print(f"  LiveVisualizer not available: {e}")
        return

    frames = window_to_frames(w)
    if len(frames) < 2:
        print("  Not enough frames for animation.")
        return

    dt = w["dt"]
    mmsi_list = sorted(w["vessels"].keys())

    vessel_metas = [
        VesselMeta(
            vessel_id=str(mmsi),
            initial_state=frames[0][str(mmsi)],
        )
        for mmsi in mmsi_list if str(mmsi) in frames[0]
    ]

    sim_info = SimInfo(
        name=f"AIS window {w['window_id']}  {w['meta']['scenario'].upper()}"
             f"  enc_CPA={w['meta'].get('enc_cpa_m', 0):.0f}m",
        n_steps=len(frames) - 1,
        dt=dt,
        vessels=vessel_metas,
    )
    vc = VizConfig(
        figsize=(10, 9),
        realtime_factor=60.0,
        trail_length=40,
        trail_alpha=0.55,
        vessel_length=30.0,
        vessel_width=8.0,
        bow_fraction=0.25,
        vessel_alpha=0.85,
    )

    all_n = [s.north for f in frames for s in f.values()]
    all_e = [s.east for f in frames for s in f.values()]
    pad = 300.0
    e_min, e_max = min(all_e) - pad, max(all_e) + pad
    n_min, n_max = min(all_n) - pad, max(all_n) + pad
    er, nr = e_max - e_min, n_max - n_min
    if nr > er:
        d = (nr - er) / 2; e_min -= d; e_max += d
    else:
        d = (er - nr) / 2; n_min -= d; n_max += d

    viz = LiveVisualizer(sim_info, vc, xlim=(e_min, e_max), ylim=(n_min, n_max), is_ais=True)
    viz.show()
    frame_pause = dt / vc.realtime_factor
    k = 0
    try:
        while k <= sim_info.n_steps and not viz._stopped:
            if not viz._paused:
                states = {str(mmsi): frames[k][str(mmsi)] for mmsi in mmsi_list if str(mmsi) in frames[k]}
                viz.step(states, k)
                k += 1
                plt.pause(frame_pause)
            else:
                plt.pause(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")


# ---------------------------------------------------------------------------
# Browser main loop
# ---------------------------------------------------------------------------

class Browser:
    def __init__(
        self,
        windows: list[dict],
        review_path: Path,
        filter_mode: str,
    ) -> None:
        self.windows = windows
        self.review_path = review_path
        self.review_state: dict[str, str] = {}
        self._load_review()

        if filter_mode != "all":
            filtered = [
                w for w in windows
                if self.review_state.get(str(w["window_id"]), "unseen") == filter_mode
            ]
            if not filtered:
                print(f"No windows with label '{filter_mode}'. Showing all.")
            else:
                self.windows = filtered

        self.idx = 0
        self._running = True
        self._animate_pending = False
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("close_event", lambda e: setattr(self, "_running", False))

    # ------------------------------------------------------------------ #

    def _load_review(self) -> None:
        if self.review_path.exists():
            with open(self.review_path) as f:
                self.review_state = json.load(f)

    def _save_review(self) -> None:
        with open(self.review_path, "w") as f:
            json.dump(self.review_state, f, indent=2)

    def _current_review(self) -> str:
        w = self.windows[self.idx]
        return self.review_state.get(str(w["window_id"]), "unseen")

    def _set_review(self, label: str) -> None:
        w = self.windows[self.idx]
        rid = str(w["window_id"])
        if label == "unseen":
            self.review_state.pop(rid, None)
        else:
            self.review_state[rid] = label
        self._save_review()
        self._redraw()
        print(f"  Window {w['window_id']} -> {label}")

    def _redraw(self) -> None:
        if not self.windows:
            return
        w = self.windows[self.idx]
        rev = self._current_review()
        _plot_window(w, self.fig, rev)
        n_acc = sum(1 for v in self.review_state.values() if v == "accepted")
        n_rej = sum(1 for v in self.review_state.values() if v == "rejected")
        status = (f"[{self.idx + 1}/{len(self.windows)}]  "
                  f"accepted={n_acc}  rejected={n_rej}  "
                  f"n=next  p=prev  a=accept  r=reject  u=unseen  v=animate  q=quit")
        try:
            self.fig.canvas.manager.set_window_title(status)
        except Exception:
            pass
        self.fig.canvas.draw_idle()

    def _on_key(self, event: object) -> None:
        k = getattr(event, "key", None)
        if k in ("n", "right"):
            self.idx = min(self.idx + 1, len(self.windows) - 1)
            self._redraw()
        elif k in ("p", "left"):
            self.idx = max(self.idx - 1, 0)
            self._redraw()
        elif k == "a":
            self._set_review("accepted")
        elif k == "r":
            self._set_review("rejected")
        elif k == "u":
            self._set_review("unseen")
        elif k == "v":
            self._animate_pending = True  # handled in run() loop to avoid nesting
        elif k in ("q", "escape"):
            self._quit()

    def _quit(self) -> None:
        self._save_review()
        n_acc = sum(1 for v in self.review_state.values() if v == "accepted")
        n_rej = sum(1 for v in self.review_state.values() if v == "rejected")
        print(f"\nReview saved: {n_acc} accepted, {n_rej} rejected.")
        print(f"State file: {self.review_path}")
        self._running = False
        plt.close(self.fig)

    def run(self) -> None:
        if not self.windows:
            print("No windows to display.")
            return
        print(f"Browsing {len(self.windows)} windows.")
        print("Controls: n=next  p=prev  a=accept  r=reject  u=unseen  v=animate  q=quit")
        plt.ion()
        self._redraw()
        plt.show(block=False)
        while self._running:
            if self._animate_pending:
                self._animate_pending = False
                _animate_window(self.windows[self.idx])
                # After animation window is closed, redraw browser
                if self._running:
                    self._redraw()
                    plt.show(block=False)
            try:
                plt.pause(0.05)
            except Exception:
                break
        plt.ioff()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Browse AIS encounter windows interactively."
    )
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA,
                        help=f"Encounter windows pkl (default {_DEFAULT_DATA})")
    parser.add_argument("--review", type=Path, default=_DEFAULT_REVIEW,
                        help=f"Review state JSON (default {_DEFAULT_REVIEW})")
    parser.add_argument(
        "--filter", choices=["all", "accepted", "rejected", "unseen"], default="all",
        help="Show only windows with this review label (default: all)",
    )
    parser.add_argument(
        "--sort", choices=["score", "cpa", "tcpa", "window_id"], default="score",
        help="Sort windows by this criterion (default: score)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print summary table and exit without launching the browser",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"ERROR: Data file not found: {args.data}", file=sys.stderr)
        print("Run process_raw_ais.py first to generate it.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.data} ...")
    with open(args.data, "rb") as f:
        windows: list[dict] = pickle.load(f)
    print(f"Loaded {len(windows)} windows.")

    windows = _sort_windows(windows, args.sort)

    review_state: dict[str, str] = {}
    if args.review.exists():
        with open(args.review) as f:
            review_state = json.load(f)

    if args.summary:
        _print_summary_table(windows, review_state)
        return

    browser = Browser(windows, args.review, args.filter)
    browser.run()


if __name__ == "__main__":
    main()
