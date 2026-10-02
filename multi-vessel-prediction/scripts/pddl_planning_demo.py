"""PDDL planning on top of the predictor, on a synthetic crossing (no data needed).

Every few seconds the planning problem is regenerated from the prediction.
When its facts change (a replanning event), it is written to PDDL and solved
with Tamer (or Aries) through Unified Planning. While a give-way target is
still trusted, a contingency plan for "it never gives way" is solved too.

    python scripts/pddl_planning_demo.py                      # ego is stand-on, target never gives way
    python scripts/pddl_planning_demo.py --scenario give_way  # ego must give way
    python scripts/pddl_planning_demo.py --engine aries

Needs the planning extra: pip install -e ".[planning]"
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.planner_api import PredictionService
from src.planner_api.pddl import build_problem

DT = 2.0

# vessel id -> (north0, east0, course_deg, speed); every vessel holds course.
SCENARIOS = {
    # Target crosses from ego's port side: target gives way, but never does.
    "stand_on": {"ego": (0.0, 0.0, 0.0, 5.0), "target": (1000.0, -1000.0, 90.0, 5.0)},
    # Target crosses from ego's starboard side: ego gives way.
    "give_way": {"ego": (0.0, 0.0, 0.0, 5.0), "target": (1000.0, 1000.0, 270.0, 5.0)},
}


def observe(vessels: dict, t: float) -> dict[str, dict]:
    obs = {}
    for vid, (n0, e0, crs, spd) in vessels.items():
        c = math.radians(crs)
        obs[vid] = {"north": n0 + spd * math.cos(c) * t, "east": e0 + spd * math.sin(c) * t,
                    "course_deg": crs, "speed": spd}
    return obs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="stand_on")
    ap.add_argument("--duration", type=float, default=240.0, help="simulated time [s]")
    ap.add_argument("--plan-every", type=float, default=10.0, help="regenerate the problem every N s")
    ap.add_argument("--out", default="outputs/pddl", help="where to write domain/problem files")
    ap.add_argument("--engine", choices=["tamer", "aries"], default="tamer")
    args = ap.parse_args()

    service = PredictionService.from_config(
        {
            "H": 150,                                  # 300 s horizon: covers manoeuvre + CPA
            "interaction": {
                "use_responding_filter": True,
                "top_k_hypotheses": 4,
                "agent": {"d_pass": 150.0, "tcpa_threshold": 600.0},
            },
        },
        dt=DT,
        own_id="ego",
    )

    vessels = SCENARIOS[args.scenario]
    plan_every = max(1, round(args.plan_every / DT))
    last_facts = None

    for k in range(int(args.duration / DT) + 1):
        t = k * DT
        service.step(observe(vessels, t), t=t)
        if k == 0 or k % plan_every:
            continue

        problem = build_problem(service, d_pass=150.0)
        # TIL times move with the clock, so only the static facts mark an event.
        facts = {k: v for k, v in problem.facts.items() if k != "til"}
        if facts == last_facts:
            continue                                   # nothing changed: keep the current plan
        last_facts = facts

        out_dir = Path(args.out) / f"t{int(t):04d}"
        problem.write_pddl(out_dir)
        p = service.get_responsiveness()
        beliefs = ", ".join(f"{v}={p[v]:.2f}" for v in p if v != "ego")
        print(f"t={t:5.0f} s  P(responsive): {beliefs}   at risk: {problem.targets_at_risk or '-'}"
              f"   -> {out_dir}")
        show("plan", problem.solve(engine=args.engine))

        # Plan B for every trusted give-way target: what if it never gives way?
        for target in problem.targets_at_risk:
            if [problem.own_id, target] in [[problem.vessels[a], problem.vessels[b]]
                                            for a, b in problem.facts.get("stand_on", [])] \
                    and target in [problem.vessels[v] for v, in problem.facts.get("trusted", [])]:
                contingency = build_problem(service, d_pass=150.0, assume_non_responsive=[target])
                show(f"if {target} never gives way", contingency.solve(engine=args.engine))


def show(label: str, result: dict) -> None:
    m = result["manoeuvre"]
    when = ""
    if m is not None and result["latest_start_s"] is not None:
        when = f", start at {result['start_s']:.0f} s (latest {result['latest_start_s']:.0f} s)"
    print(f"          {label}: {result['status']}" + (f" -> {m['name']}{when}" if m else ""))
    for step in result["plan"]:
        print(f"            {step['start']:7.1f}  ({step['action']} {' '.join(step['args'])})"
              + (f"  [{step['duration']:.0f} s]" if step["duration"] else ""))


if __name__ == "__main__":
    main()
