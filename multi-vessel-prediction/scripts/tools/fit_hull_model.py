"""scripts/tools/fit_hull_model.py

Fit HullModel (planner_api/manoeuvre_search.py) to the milliAmpere hull
following the planner's own-ship tracks through DP.

For every action sequence of the search and each start speed, the own ship's
reference track is rolled out as the planner does, and the milliAmpere
simulator follows it the way SearchDPPolicy drives it: re-anchored on the
hull every replan_s. The HullModel parameters are then fitted (Nelder-Mead,
mean squared position error) on two thirds of the tracks and reported on the
rest, against taking the reference as the hull.

Usage
-----
    python scripts/tools/fit_hull_model.py
    python scripts/tools/fit_hull_model.py --config scenarios/simulation/milliampere/give_way_stop.yaml
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_simulation as rs  # noqa: E402
from colregs_planner.motion_models.adapters import MilliAmpereAdapter  # noqa: E402
from colregs_planner.planner_api.manoeuvre_search import HullModel  # noqa: E402
from colregs_planner.prediction.vo_rollout import vo_rollout  # noqa: E402
from colregs_planner.vessel.state import DPAction, InitialState, VesselState  # noqa: E402


def follow(ref: np.ndarray, v0: np.ndarray, plan_dt: float, dt: float,
           duration: float, replan_s: float) -> np.ndarray:
    """Hull positions at plan_dt as the milliAmpere follows *ref* through DP,
    with the reference moved onto the hull every replan_s."""
    ref = ref.copy()
    speed = float(np.linalg.norm(v0))
    psi = float(np.arctan2(v0[1], v0[0])) if speed > 0 else 0.0
    adapter = MilliAmpereAdapter()
    s = adapter.init(InitialState(north=ref[0, 0], east=ref[0, 1], heading=psi, speed=speed))
    every = int(round(replan_s / dt))

    def at(t: float) -> np.ndarray:
        f = t / plan_dt
        j = min(int(f), len(ref) - 2)
        return ref[j] + (ref[j + 1] - ref[j]) * (f - j)

    out = [[s.north, s.east]]
    for k in range(int(round(duration / dt))):
        t = k * dt
        if k and k % every == 0:
            j = int(round(t / plan_dt))
            ref[j:] += np.array([s.north, s.east]) - ref[j]
        r0, r1 = at(t), at(t + dt)
        d = r1 - r0
        psi_ref = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-9 else s.psi
        s = adapter.step(DPAction(d_north=d[0], d_east=d[1], psi_ref=psi_ref,
                                  ref_start=(r0[0], r0[1], psi_ref)), dt)
        out.append([s.north, s.east])
    return np.array(out)[::int(round(plan_dt / dt))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "scenarios/simulation/milliampere/give_way_stop.yaml")
    parser.add_argument("--duration", type=float, default=40.0, help="Seconds of each track fitted")
    parser.add_argument("--speeds", type=float, nargs="+", default=[1.5, 1.0, 0.5, 0.0],
                        help="Start speeds [m/s]")
    args = parser.parse_args()

    config, vessel_cfgs, _, pred_cfg = rs.load_config(args.config)
    vi = config.vessels[0]
    vcfg = dict(vessel_cfgs[vi.vessel_id], policy="search_dp")
    policy = rs._build_policy(vcfg, vi, config, 0, pred_cfg, vessel_cfgs)
    planner, replan_s = policy.planner, policy.replan_s
    c = planner.cfg
    own_cfg = planner.configs[vi.vessel_id]
    H = int(round(c.horizon_s / c.plan_dt))
    n = int(round(args.duration / c.plan_dt)) + 1

    tracks = []
    for u0 in args.speeds:
        v0 = u0 * np.array([np.cos(own_cfg.nominal_psi), np.sin(own_cfg.nominal_psi)])
        start = VesselState(0.0, 0.0, own_cfg.nominal_psi, u0)
        for actions in itertools.product(c.actions, repeat=c.depth):
            cfg = replace(own_cfg, pref_vel_seq=planner.velocity_sequence(actions))
            ref = vo_rollout({"own": start}, c.plan_dt, H, {"own": cfg}, fixed_plan_ids={"own"})["own"]
            ref = np.array([[s.north, s.east] for s in ref])
            hull = follow(ref, v0, c.plan_dt, config.dt, args.duration, replan_s)
            tracks.append((ref[:n], v0, hull[:n], "stop" in (a.name for a in actions)))
    print(f"{len(tracks)} tracks, {args.duration:.0f} s each, re-anchored every {replan_s:.0f} s")

    def errors(model: HullModel, sel) -> np.ndarray:
        return np.array([
            np.linalg.norm(model.track(tracks[i][0][None], tracks[i][1], c.plan_dt)[0][0] - tracks[i][2], axis=1)
            for i in sel
        ])

    order = np.random.default_rng(0).permutation(len(tracks))
    train, test = order[: len(tracks) * 2 // 3], order[len(tracks) * 2 // 3:]

    def model(x) -> HullModel:
        kp, kd, cd, a_max = np.abs(x)
        return HullModel(kp=kp, kd=kd, c=cd, a_max=a_max, reanchor_s=replan_s)

    d = HullModel()
    res = minimize(lambda x: float(np.mean(errors(model(x), train) ** 2)),
                   [d.kp + 0.02, d.kd, d.c + 0.005, d.a_max], method="Nelder-Mead",
                   options=dict(maxiter=300, xatol=1e-4, fatol=1e-4))
    fit = model(res.x)
    print(f"kp={fit.kp:.4f}  kd={fit.kd:.4f}  c={fit.c:.4f}  a_max={fit.a_max:.4f}")

    stops = [i for i in test if tracks[i][3]]
    ref_err = np.array([np.linalg.norm(tracks[i][0] - tracks[i][2], axis=1) for i in test])
    for label, e, es in (("reference as hull", ref_err, ref_err[[list(test).index(i) for i in stops]]),
                         ("fitted model", errors(fit, test), errors(fit, stops))):
        print(f"  test, {label:17}: RMS {np.sqrt(np.mean(e ** 2)):.2f} m, worst {e.max():.1f} m"
              f" | plans with a stop: RMS {np.sqrt(np.mean(es ** 2)):.2f} m, worst {es.max():.1f} m")


if __name__ == "__main__":
    main()
