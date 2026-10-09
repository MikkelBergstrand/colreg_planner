"""scripts/tools/profile_vo_rollout.py

Profile and time vo_rollout for both policy types.

Usage
-----
    python scripts/tools/profile_vo_rollout.py                # timing + top-20 cProfile
    python scripts/tools/profile_vo_rollout.py --policy vo_kuwata
    python scripts/tools/profile_vo_rollout.py --policy vo_analytic
    python scripts/tools/profile_vo_rollout.py --snakeviz     # open snakeviz after profiling
    python scripts/tools/profile_vo_rollout.py --n-vessels 4 --n-steps 200

The script builds a synthetic head-on encounter (configurable number of
vessels), runs the rollout N_REPEATS times for a stable timing estimate, then
dumps the cProfile stats sorted by cumulative time.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import numpy as np

from colregs_planner.vessel.state import VesselState
from colregs_planner.prediction.vo_rollout import vo_rollout, VOAgentConfig


# ---------------------------------------------------------------------------
# Synthetic scenario builder
# ---------------------------------------------------------------------------

def _make_states(n_vessels: int, speed: float = 5.0) -> dict[str, VesselState]:
    """Place vessels in a spoke pattern converging toward the origin."""
    states: dict[str, VesselState] = {}
    for i in range(n_vessels):
        angle = 2 * np.pi * i / n_vessels          # evenly spaced bearings
        north = 400.0 * np.cos(angle)
        east  = 400.0 * np.sin(angle)
        # Heading toward origin
        psi   = np.arctan2(-east, -north)
        vid   = "ego" if i == 0 else f"target_{i}"
        states[vid] = VesselState(
            north=north, east=east,
            psi=psi, speed=speed,
        )
    return states


def _make_configs(
    vessel_ids:  list[str],
    policy_type: str,
) -> dict[str, VOAgentConfig]:
    return {
        vid: VOAgentConfig(
            policy_type    = policy_type
        )
        for vid in vessel_ids
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _time_rollout(
    states:      dict[str, VesselState],
    configs:     dict[str, VOAgentConfig],
    n_steps:     int,
    n_repeats:   int,
    dt:          float,
) -> tuple[float, float]:
    """Return (mean_ms, std_ms) over n_repeats runs."""
    times: list[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        vo_rollout(
            initial_states = states,
            dt             = dt,
            n_steps        = n_steps,
            agent_configs  = configs,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def _profile_rollout(
    states:  dict[str, VesselState],
    configs: dict[str, VOAgentConfig],
    n_steps: int,
    dt:      float,
    top_n:   int,
) -> str:
    pr = cProfile.Profile()
    pr.enable()
    vo_rollout(
        initial_states = states,
        dt             = dt,
        n_steps        = n_steps,
        agent_configs  = configs,
    )
    pr.disable()

    buf = io.StringIO()
    ps  = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(top_n)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Profile vo_rollout")
    parser.add_argument("--policy",    choices=["vo_kuwata", "vo_analytic", "both"],
                        default="both")
    parser.add_argument("--n-vessels", type=int,   default=2)
    parser.add_argument("--n-steps",   type=int,   default=120,
                        help="Rollout steps (default 120 = 60 s at dt=0.5)")
    parser.add_argument("--dt",        type=float, default=0.5)
    parser.add_argument("--repeats",   type=int,   default=10,
                        help="Timing repeats for stable mean/std")
    parser.add_argument("--top",       type=int,   default=20,
                        help="Top N functions in cProfile output")
    parser.add_argument("--snakeviz",  action="store_true",
                        help="Open snakeviz after profiling (requires pip install snakeviz)")
    args = parser.parse_args()

    states = _make_states(args.n_vessels)
    vids   = list(states.keys())

    policies = (
        ["vo_kuwata", "vo_analytic"] if args.policy == "both" else [args.policy]
    )

    print(
        f"\nvo_rollout profile  —  {args.n_vessels} vessels, "
        f"{args.n_steps} steps (dt={args.dt} s), "
        f"{args.repeats} timing repeats\n"
        + "=" * 60
    )

    for ptype in policies:
        configs = _make_configs(vids, ptype)

        mean_ms, std_ms = _time_rollout(
            states, configs, args.n_steps, args.repeats, args.dt,
        )
        per_step_us = mean_ms / args.n_steps * 1000.0
        print(
            f"\n[{ptype}]  "
            f"mean={mean_ms:.1f} ms  std={std_ms:.1f} ms  "
            f"({per_step_us:.0f} µs/step)"
        )

        profile_output = _profile_rollout(
            states, configs, args.n_steps, args.dt, args.top,
        )
        print(profile_output)

    if args.snakeviz:
        import tempfile, subprocess, sys
        configs = _make_configs(vids, policies[-1])
        with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as f:
            prof_path = f.name
        cProfile.run(
            "vo_rollout(initial_states=states, dt=args.dt, "
            "n_steps=args.n_steps, agent_configs=configs)",
            prof_path,
        )
        subprocess.Popen([sys.executable, "-m", "snakeviz", prof_path])
        print(f"snakeviz launched for: {prof_path}")


if __name__ == "__main__":
    main()
