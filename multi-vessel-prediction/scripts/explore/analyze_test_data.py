"""analyze_test_data.py

Exploratory analysis of the AIS encounter dataset (data/test.pkl).

Data structure
--------------
Each record (window) in the pickle is a dict with:
  window_id             : int
  rollout_start_idx     : int   index into the 40-step sequence where rollout starts
  source_rollout_start_idx : int
  group                 : list[ndarray(40, 27)]  one array per vessel in the encounter
  nbr1_idx / nbr2_idx   : array  which vessel index is nearest/second-nearest neighbor
  risk_t / risk_nbr1_t / risk_nbr2_t : array(n_vessels, 40)  collision risk over time
  cpa_nbr1_t / cpa_nbr2_t            : array(n_vessels, 40)  CPA [m] over time
  tcpa_nbr1_t / tcpa_nbr2_t          : array(n_vessels, 40)  TCPA [s] over time
  alpha_nbr1_t / alpha_nbr2_t        : array(n_vessels, 40)  bearing angle [deg]
  beta_nbr1_t / beta_nbr2_t          : array(n_vessels, 40)  relative heading [deg]
  risk_win                           : array(n_vessels,)  aggregate window risk

Group array column layout (27 columns)
---------------------------------------
  0  latitude  [deg]
  1  longitude [deg]
  2  SOG - speed over ground [knots]
  3  COG - course over ground [deg, 0-360, NED]
  4  timestamp [Unix s]
  5  MMSI
  6  AIS vessel type code
  7  internal track/group id
  8  valid flag  (1 = valid AIS message)
  9  relative north from group centroid [m]
 10  relative east  from group centroid [m]
 11  range from group centroid [m]
 12  cos(heading) - north component of unit velocity (NED)
 13  sin(heading) - east  component of unit velocity (NED)
 14  heading angular rate [rad/s] (estimated from consecutive steps)
 15  cos(heading), neighbor 1
 16  sin(heading), neighbor 1
 17-25 additional neighbor features (zero-padded when no second neighbor)
 26  encounter-active flag (1 = encounter ongoing)

Rows are ordered chronologically; zero-filled rows precede the observation window.
"""

import pickle
import math
import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Column index constants
# ---------------------------------------------------------------------------
COL_LAT    = 0
COL_LON    = 1
COL_SOG    = 2
COL_COG    = 3
COL_TS     = 4
COL_MMSI   = 5
COL_TYPE   = 6
COL_VALID  = 8
COL_RNORTH = 9
COL_REAST  = 10
COL_RANGE  = 11
COL_COSH   = 12
COL_SINH   = 13
COL_ACTIVE = 26


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(path: str | Path) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def valid_rows(arr: np.ndarray) -> np.ndarray:
    """Return only the rows where COL_VALID == 1."""
    return arr[arr[:, COL_VALID] == 1]


def extract_all_vessel_rows(windows: list[dict]) -> np.ndarray:
    """Flatten all valid vessel rows across the dataset into one (N, 27) array."""
    parts = []
    for w in windows:
        for vessel_arr in w["group"]:
            rows = valid_rows(vessel_arr)
            if rows.size:
                parts.append(rows)
    return np.vstack(parts)


def obs_lengths(windows: list[dict]) -> list[int]:
    """Number of valid timesteps per vessel track across all windows."""
    lengths = []
    for w in windows:
        for arr in w["group"]:
            lengths.append(int((arr[:, COL_VALID] == 1).sum()))
    return lengths


def encounter_stats(windows: list[dict]) -> dict:
    """Collect CPA, TCPA, alpha, beta, and risk arrays for all encounter windows."""
    cpa, tcpa, alpha, beta, risk_win = [], [], [], [], []
    for w in windows:
        n_v = len(w["group"])
        rs = w["rollout_start_idx"]
        for v_idx in range(n_v):
            cpa_row  = w["cpa_nbr1_t"][v_idx]
            tcpa_row = w["tcpa_nbr1_t"][v_idx]
            if not np.isnan(cpa_row[rs]):
                cpa.append(float(cpa_row[rs]))
            if not np.isnan(tcpa_row[rs]):
                tcpa.append(float(tcpa_row[rs]))

            a = w["alpha_nbr1_t"][v_idx][rs]
            b = w["beta_nbr1_t"][v_idx][rs]
            if not np.isnan(a):
                alpha.append(float(a))
            if not np.isnan(b):
                beta.append(float(b))

        risk_win.extend(w["risk_win"].tolist())

    return dict(
        cpa=np.array(cpa),
        tcpa=np.array(tcpa),
        alpha=np.array(alpha),
        beta=np.array(beta),
        risk_win=np.array(risk_win),
    )


def unique_mmsi(windows: list[dict]) -> set:
    rows = extract_all_vessel_rows(windows)
    return set(rows[:, COL_MMSI].astype(int).tolist())


def time_range(windows: list[dict]) -> tuple[datetime.datetime, datetime.datetime]:
    rows = extract_all_vessel_rows(windows)
    ts = rows[:, COL_TS]
    return (datetime.datetime.fromtimestamp(ts.min()),
            datetime.datetime.fromtimestamp(ts.max()))


def vessel_type_counts(windows: list[dict]) -> dict:
    rows = extract_all_vessel_rows(windows)
    types, counts = np.unique(rows[:, COL_TYPE].astype(int), return_counts=True)
    return dict(zip(types.tolist(), counts.tolist()))


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

_AIS_TYPE_LABELS = {
    0: "Not available", 20: "WIG", 21: "Fishing", 22: "Towing",
    23: "Fishing/Towing", 24: "Dredging/Diving", 25: "Military ops",
    26: "Sailing", 27: "Pleasure craft", 28: "HSC", 29: "Pilot",
    30: "Fishing", 31: "Towing", 32: "Towing (large)",
    36: "Sailing", 37: "Pleasure craft", 60: "Passenger",
    70: "Cargo", 80: "Tanker",
}


def print_summary(windows: list[dict]):
    rows = extract_all_vessel_rows(windows)
    n_per_win = [len(w["group"]) for w in windows]
    t0, t1 = time_range(windows)
    enc = encounter_stats(windows)
    mmsi_set = unique_mmsi(windows)
    type_counts = vessel_type_counts(windows)
    obs_len = obs_lengths(windows)

    sog = rows[:, COL_SOG]
    lats = rows[:, COL_LAT]
    lons = rows[:, COL_LON]

    high_risk = sum(1 for w in windows if w["risk_win"].max() > 0.1)

    print("=" * 60)
    print("AIS ENCOUNTER DATASET SUMMARY")
    print("=" * 60)
    print(f"Windows (encounter groups) : {len(windows)}")
    print(f"Unique vessels (MMSI)      : {len(mmsi_set)}")
    print(f"Time range                 : {t0:%Y-%m-%d %H:%M} -> {t1:%Y-%m-%d %H:%M}")
    print(f"Duration                   : {(t1-t0).total_seconds()/3600:.1f} h")
    print()
    print("Geographic bounds")
    print(f"  Latitude  : {lats.min():.5f} - {lats.max():.5f} deg")
    print(f"  Longitude : {lons.min():.5f} - {lons.max():.5f} deg")
    print(f"  Region    : Moss-Horten, Oslofjord, Norway")
    print()
    print("Vessels per window")
    for n in sorted(set(n_per_win)):
        cnt = n_per_win.count(n)
        print(f"  {n} vessel(s) : {cnt:5d} windows  ({100*cnt/len(windows):.1f}%)")
    print()
    print("Observation window")
    print(f"  Timesteps per vessel : 40")
    print(f"  Valid obs per track  : {min(obs_len)} - {max(obs_len)}"
          f"  (mean {np.mean(obs_len):.1f})")
    print()
    print("Speed over ground (SOG)")
    print(f"  Min / Mean / Max : {sog.min():.1f} / {sog.mean():.1f} / {sog.max():.1f} knots")
    print(f"  Std              : {sog.std():.1f} knots")
    print()
    print("Vessel type distribution (AIS code)")
    for code, cnt in sorted(type_counts.items(), key=lambda x: -x[1])[:10]:
        label = _AIS_TYPE_LABELS.get(code, f"type {code}")
        print(f"  {code:3d}  {label:<22s} : {cnt}")
    print()
    print("Encounter metrics (at rollout start, nearest neighbor)")
    if len(enc["cpa"]):
        print(f"  CPA  : {enc['cpa'].min():.0f} - {enc['cpa'].max():.0f} m"
              f"   (mean {enc['cpa'].mean():.0f} m,  median {np.median(enc['cpa']):.0f} m)")
        tcpa_v = enc["tcpa"][enc["tcpa"] < 600]
        if len(tcpa_v):
            print(f"  TCPA : {tcpa_v.min():.0f} - {tcpa_v.max():.0f} s"
                  f"   (mean {tcpa_v.mean():.0f} s,  capped at 600 s)")
    print(f"  risk_win > 0.1 : {high_risk} windows")
    print(f"  risk_win > 0.3 : {sum(1 for w in windows if w['risk_win'].max() > 0.3)} windows")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _polar_hist(ax, angles_deg: np.ndarray, bins: int = 36, title: str = ""):
    """Rose diagram."""
    edges = np.linspace(0, 360, bins + 1)
    angles_mod = angles_deg % 360
    counts, _ = np.histogram(angles_mod, bins=edges)
    theta = np.deg2rad(edges[:-1] + (edges[1] - edges[0]) / 2)
    width = 2 * np.pi / bins
    ax.bar(theta, counts, width=width * 0.85, bottom=0, align="center",
           color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, pad=8, fontsize=9)
    ax.tick_params(labelsize=7)


def plot_analysis(windows: list[dict], out_path: str | Path = "analysis.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    rows = extract_all_vessel_rows(windows)
    enc  = encounter_stats(windows)
    obs_len_vals = obs_lengths(windows)
    n_per_win = [len(w["group"]) for w in windows]

    sog  = rows[:, COL_SOG]
    lats = rows[:, COL_LAT]
    lons = rows[:, COL_LON]
    heading_deg = np.rad2deg(np.arctan2(rows[:, COL_SINH], rows[:, COL_COSH])) % 360

    # highest-risk window for the trajectory example panel
    best_w = max(windows, key=lambda w: float(w["risk_win"].max()))

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle("AIS Encounter Dataset  —  Moss-Horten Region, Norway",
                 fontsize=13, y=0.99)
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.38)

    # ------------------------------------------------------------------ #
    # 1. Geographic scatter (col-span 2)
    # ------------------------------------------------------------------ #
    ax_geo = fig.add_subplot(gs[0, :2])
    vmin, vmax = 0.0, float(np.percentile(sog, 99))
    sc = ax_geo.scatter(lons, lats, s=0.5, c=sog,
                        cmap="plasma", vmin=vmin, vmax=vmax,
                        alpha=0.35, rasterized=True)
    plt.colorbar(sc, ax=ax_geo, label="SOG [knots]", pad=0.02)
    ax_geo.set_xlabel("Longitude [deg]", fontsize=8)
    ax_geo.set_ylabel("Latitude [deg]", fontsize=8)
    ax_geo.set_title(f"Vessel positions  ({len(rows):,} valid AIS messages)", fontsize=9)
    ax_geo.set_aspect(1 / math.cos(math.radians(59.9)))
    ax_geo.tick_params(labelsize=7)

    # ------------------------------------------------------------------ #
    # 2. SOG histogram
    # ------------------------------------------------------------------ #
    ax_sog = fig.add_subplot(gs[0, 2])
    ax_sog.hist(sog, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    ax_sog.axvline(float(sog.mean()), color="crimson", linewidth=1.3, linestyle="--",
                   label=f"mean {sog.mean():.1f} kn")
    ax_sog.set_xlabel("SOG [knots]", fontsize=8)
    ax_sog.set_ylabel("Count", fontsize=8)
    ax_sog.set_title("Speed over ground", fontsize=9)
    ax_sog.legend(fontsize=7)
    ax_sog.tick_params(labelsize=7)

    # ------------------------------------------------------------------ #
    # 3. Heading rose
    # ------------------------------------------------------------------ #
    ax_rose = fig.add_subplot(gs[0, 3], projection="polar")
    _polar_hist(ax_rose, heading_deg, bins=36, title="Heading distribution")

    # ------------------------------------------------------------------ #
    # 4. Vessels per window
    # ------------------------------------------------------------------ #
    ax_nvw = fig.add_subplot(gs[1, 0])
    vals, counts = np.unique(n_per_win, return_counts=True)
    bar_colors = ["#4878CF", "#6ACC65", "#D65F5F"]
    ax_nvw.bar(vals, counts, color=bar_colors[:len(vals)], edgecolor="white")
    ax_nvw.set_xticks(vals)
    ax_nvw.set_xlabel("Vessels in window", fontsize=8)
    ax_nvw.set_ylabel("Windows", fontsize=8)
    ax_nvw.set_title("Encounter size distribution", fontsize=9)
    for v, c in zip(vals, counts):
        ax_nvw.text(v, c + 5, str(c), ha="center", fontsize=7)
    ax_nvw.tick_params(labelsize=7)

    # ------------------------------------------------------------------ #
    # 5. Observation length histogram
    # ------------------------------------------------------------------ #
    ax_obs = fig.add_subplot(gs[1, 1])
    ax_obs.hist(obs_len_vals, bins=20, color="teal", edgecolor="white", linewidth=0.3)
    ax_obs.set_xlabel("Valid timesteps per track", fontsize=8)
    ax_obs.set_ylabel("Count", fontsize=8)
    ax_obs.set_title("Observation length distribution", fontsize=9)
    ax_obs.tick_params(labelsize=7)

    # ------------------------------------------------------------------ #
    # 6. CPA histogram
    # ------------------------------------------------------------------ #
    ax_cpa = fig.add_subplot(gs[1, 2])
    if len(enc["cpa"]):
        ax_cpa.hist(enc["cpa"], bins=40, color="darkorange", edgecolor="white",
                    linewidth=0.3)
        ax_cpa.axvline(float(enc["cpa"].mean()), color="crimson", linewidth=1.3,
                       linestyle="--", label=f"mean {enc['cpa'].mean():.0f} m")
        ax_cpa.legend(fontsize=7)
    ax_cpa.set_xlabel("CPA [m]", fontsize=8)
    ax_cpa.set_ylabel("Count", fontsize=8)
    ax_cpa.set_title("Closest Point of Approach (at rollout start)", fontsize=9)
    ax_cpa.tick_params(labelsize=7)

    # ------------------------------------------------------------------ #
    # 7. TCPA histogram
    # ------------------------------------------------------------------ #
    ax_tcpa = fig.add_subplot(gs[1, 3])
    if len(enc["tcpa"]):
        tcpa_v = enc["tcpa"][enc["tcpa"] < 600]
        ax_tcpa.hist(tcpa_v, bins=40, color="mediumpurple", edgecolor="white",
                     linewidth=0.3)
        ax_tcpa.axvline(float(tcpa_v.mean()), color="crimson", linewidth=1.3,
                        linestyle="--", label=f"mean {tcpa_v.mean():.0f} s")
        ax_tcpa.legend(fontsize=7)
    ax_tcpa.set_xlabel("TCPA [s]  (< 600 s shown)", fontsize=8)
    ax_tcpa.set_ylabel("Count", fontsize=8)
    ax_tcpa.set_title("Time to CPA (at rollout start)", fontsize=9)
    ax_tcpa.tick_params(labelsize=7)

    # ------------------------------------------------------------------ #
    # 8. Window risk distribution (log scale)
    # ------------------------------------------------------------------ #
    ax_risk = fig.add_subplot(gs[2, 0])
    ax_risk.hist(enc["risk_win"], bins=60, color="crimson", edgecolor="white",
                 linewidth=0.3, log=True)
    ax_risk.set_xlabel("risk_win (aggregate window risk)", fontsize=8)
    ax_risk.set_ylabel("Count (log)", fontsize=8)
    ax_risk.set_title("Window-level risk distribution", fontsize=9)
    ax_risk.tick_params(labelsize=7)

    # ------------------------------------------------------------------ #
    # 9. Alpha (bearing angle) rose
    # ------------------------------------------------------------------ #
    ax_alpha = fig.add_subplot(gs[2, 1], projection="polar")
    if len(enc["alpha"]):
        _polar_hist(ax_alpha, enc["alpha"], bins=36,
                    title="Bearing angle alpha to nbr1")

    # ------------------------------------------------------------------ #
    # 10. Example high-risk encounter — tracks in local NED
    # ------------------------------------------------------------------ #
    ax_traj = fig.add_subplot(gs[2, 2])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    rs = best_w["rollout_start_idx"]

    for vi, vessel_arr in enumerate(best_w["group"]):
        vrows = valid_rows(vessel_arr)
        if len(vrows) == 0:
            continue
        lat0 = float(vrows[0, COL_LAT])
        lon0 = float(vrows[0, COL_LON])
        north = (vrows[:, COL_LAT] - lat0) * 111320.0
        east  = (vrows[:, COL_LON] - lon0) * 111320.0 * math.cos(math.radians(lat0))
        c = colors[vi % len(colors)]
        label = f"V{vi}  MMSI {int(vrows[0, COL_MMSI])}"
        ax_traj.plot(east, north, color=c, linewidth=1.4, label=label)
        ax_traj.scatter(east, north, s=5, color=c, alpha=0.5)
        # diamond at rollout start
        n_valid = len(vrows)
        rs_local = min(rs, n_valid - 1)
        ax_traj.scatter(east[rs_local], north[rs_local], s=80, color=c,
                        marker="D", zorder=6, edgecolors="black", linewidths=0.5)

    ax_traj.set_xlabel("East [m]", fontsize=8)
    ax_traj.set_ylabel("North [m]", fontsize=8)
    ax_traj.set_title(
        f"Highest-risk encounter (window {best_w['window_id']}, "
        f"risk={best_w['risk_win'].max():.3f})\n"
        f"  diamond = rollout start (step {rs})", fontsize=8)
    ax_traj.legend(fontsize=6)
    ax_traj.set_aspect("equal", adjustable="datalim")
    ax_traj.tick_params(labelsize=7)

    # ------------------------------------------------------------------ #
    # 11. Example encounter — risk over time
    # ------------------------------------------------------------------ #
    ax_rt = fig.add_subplot(gs[2, 3])
    for vi in range(len(best_w["group"])):
        risk_row = best_w["risk_t"][vi]
        ax_rt.plot(np.arange(len(risk_row)), risk_row,
                   color=colors[vi % len(colors)], linewidth=1.4, label=f"V{vi}")
    ax_rt.axvline(rs, color="black", linewidth=1.1, linestyle="--",
                  label=f"rollout start ({rs})")
    ax_rt.set_xlabel("Timestep", fontsize=8)
    ax_rt.set_ylabel("Collision risk", fontsize=8)
    ax_rt.set_title("Risk over time (highest-risk window)", fontsize=9)
    ax_rt.legend(fontsize=7)
    ax_rt.tick_params(labelsize=7)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved -> {Path(out_path).resolve()}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-vessel case extraction
# ---------------------------------------------------------------------------

_KNOTS_TO_MS = 0.5144  # 1 knot = 0.5144 m/s
DT = 60.0              # AIS message interval [s]
_LAT_TO_M = 111320.0   # metres per degree latitude


def _classify_scenario(alpha_deg: float) -> str:
    """Classify COLREGS encounter type from bearing angle alpha (degrees)."""
    if np.isnan(alpha_deg):
        return "unknown"
    a = abs(alpha_deg % 360)
    if a > 180:
        a = 360 - a
    if a < 22.5:
        return "head-on"
    if a > 157.5:
        return "overtaking"
    return "crossing"


def _interestingness(w: dict) -> float:
    """Score a multi-vessel window for how useful it is as a test case.

    Higher score = more interesting.  Favours:
      - close CPA   (< 500 m)
      - upcoming TCPA  (30-600 s at rollout start)
      - higher risk
      - more valid observations per vessel
    """
    rs = w["rollout_start_idx"]
    n_v = len(w["group"])

    cpa  = float(w["cpa_nbr1_t"][0][rs])
    tcpa = float(w["tcpa_nbr1_t"][0][rs])

    if np.isnan(cpa) or np.isnan(tcpa):
        return -1.0
    if tcpa <= 0:
        return -1.0  # encounter already past at rollout start

    obs_counts = [(w["group"][vi][:, COL_VALID] == 1).sum() for vi in range(n_v)]
    min_obs = min(obs_counts)
    if min_obs < 10:
        return -1.0  # too few observations to be useful

    risk = float(w["risk_win"].max())

    # normalise CPA: score peaks at ~100 m, decays for very close or very far
    cpa_score  = math.exp(-((cpa - 150) / 300) ** 2)
    tcpa_score = math.exp(-((tcpa - 200) / 250) ** 2)
    obs_score  = min(min_obs / 40.0, 1.0)

    return 0.4 * cpa_score + 0.3 * tcpa_score + 0.2 * risk + 0.1 * obs_score


def select_test_cases(
    windows: list[dict],
    n_cases: int = 10,
) -> list[dict]:
    """Return the most interesting multi-vessel windows as test cases.

    Applies diversity: tries to return at least one case per scenario type
    (head-on, crossing, overtaking) before filling remaining slots by score.

    Returns
    -------
    list of the original window dicts, enriched with a 'meta' key containing:
        scenario, cpa_m, tcpa_s, risk, obs_counts, score
    """
    multi = [w for w in windows if len(w["group"]) >= 2]

    scored = []
    for w in multi:
        rs    = w["rollout_start_idx"]
        score = _interestingness(w)
        alpha = float(w["alpha_nbr1_t"][0][rs])
        sc    = _classify_scenario(alpha)
        cpa   = float(w["cpa_nbr1_t"][0][rs])
        tcpa  = float(w["tcpa_nbr1_t"][0][rs])
        risk  = float(w["risk_win"].max())
        obs   = [int((w["group"][vi][:, COL_VALID] == 1).sum()) for vi in range(len(w["group"]))]
        w = dict(w)  # shallow copy so we don't mutate the original list
        w["meta"] = dict(scenario=sc, cpa_m=cpa, tcpa_s=tcpa,
                         risk=risk, obs_counts=obs, score=score)
        scored.append((score, w))

    scored.sort(key=lambda x: -x[0])
    valid = [(s, w) for s, w in scored if s > 0]

    # ensure diversity across scenario types
    seen_types: set[str] = set()
    selected = []
    remainder = []
    for _, w in valid:
        sc = w["meta"]["scenario"]
        if sc not in seen_types and len(selected) < n_cases:
            selected.append(w)
            seen_types.add(sc)
        else:
            remainder.append(w)

    for w in remainder:
        if len(selected) >= n_cases:
            break
        selected.append(w)

    return selected[:n_cases]


def print_test_cases(cases: list[dict]):
    """Print a ranked summary table of selected test cases."""
    print()
    print("=" * 80)
    print("SELECTED MULTI-VESSEL TEST CASES")
    print("=" * 80)
    header = ("  #  win_id  n  scenario    CPA[m]  TCPA[s]  risk    obs      score")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, w in enumerate(cases):
        m = w["meta"]
        print(
            "  %2d  %5d  %d  %-10s  %6.0f   %6.0f  %.3f  %-16s %.3f"
            % (
                i + 1,
                w["window_id"],
                len(w["group"]),
                m["scenario"],
                m["cpa_m"] if not np.isnan(m["cpa_m"]) else -1,
                m["tcpa_s"] if not np.isnan(m["tcpa_s"]) else -1,
                m["risk"],
                str(m["obs_counts"]),
                m["score"],
            )
        )
    print("=" * 80)


# ---------------------------------------------------------------------------
# AIS → VesselState conversion
# ---------------------------------------------------------------------------

def window_to_vessel_states(
    w: dict,
    ref_lat: float | None = None,
    ref_lon: float | None = None,
) -> tuple[list[dict], float, float]:
    """Convert one multi-vessel window to a sequence of VesselState dicts.

    Parameters
    ----------
    w : dict
        A window from the dataset (must have >= 2 vessels in 'group').
    ref_lat, ref_lon : float, optional
        Geographic reference for the local NED frame.  Defaults to the mean
        position of all vessels at the rollout_start_idx row.

    Returns
    -------
    frames : list[dict[str, VesselState]]
        One dict per valid timestep (timesteps where ALL vessels have data).
        Keys are string vessel ids "v0", "v1", … matching the group order.
    ref_lat, ref_lon : float
        The reference coordinates used (useful for geo-plotting later).
    """
    from src.vessel.state import VesselState

    rs = w["rollout_start_idx"]

    # use the average position at rollout start as NED origin
    if ref_lat is None or ref_lon is None:
        lats, lons = [], []
        for arr in w["group"]:
            row = arr[rs] if arr[rs, COL_VALID] == 1 else valid_rows(arr)[0]
            lats.append(float(row[COL_LAT]))
            lons.append(float(row[COL_LON]))
        ref_lat = float(np.mean(lats))
        ref_lon = float(np.mean(lons))

    cos_lat = math.cos(math.radians(ref_lat))

    def row_to_state(row: np.ndarray) -> "VesselState":
        north = (float(row[COL_LAT]) - ref_lat) * _LAT_TO_M
        east  = (float(row[COL_LON]) - ref_lon) * _LAT_TO_M * cos_lat
        psi   = math.atan2(float(row[COL_SINH]), float(row[COL_COSH]))
        speed = float(row[COL_SOG]) * _KNOTS_TO_MS
        return VesselState(north=north, east=east, psi=psi, speed=speed, r=0.0)

    # build per-vessel valid-row lookup: timestep index → row
    # we align on the shared timestamp grid
    vessel_ts: list[dict[float, np.ndarray]] = []
    for arr in w["group"]:
        valid = arr[arr[:, COL_VALID] == 1]
        vessel_ts.append({float(row[COL_TS]): row for row in valid})

    # find timestamps present in ALL vessels
    common_ts = set(vessel_ts[0].keys())
    for vts in vessel_ts[1:]:
        common_ts &= set(vts.keys())
    common_ts_sorted = sorted(common_ts)

    frames: list[dict] = []
    for ts in common_ts_sorted:
        frame = {}
        for vi, vts in enumerate(vessel_ts):
            vid = f"v{vi}"
            frame[vid] = row_to_state(vts[ts])
        frames.append(frame)

    return frames, ref_lat, ref_lon


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    data_path = Path(__file__).resolve().parents[2] / "data" / "test.pkl"
    out_path  = Path(__file__).resolve().parents[2] / "data" / "analysis.png"

    print(f"Loading {data_path} ...")
    windows = load(data_path)
    print(f"Loaded {len(windows)} windows.\n")

    print_summary(windows)
    plot_analysis(windows, out_path)

    cases = select_test_cases(windows, n_cases=10)
    print_test_cases(cases)

    # Show how many shared-timestamp frames each case yields
    print("\nFrame counts (shared timestamps across all vessels):")
    for w in cases:
        frames, _, _ = window_to_vessel_states(w)
        print(f"  win={w['window_id']:4d}  {len(frames)} frames")


if __name__ == "__main__":
    main()
