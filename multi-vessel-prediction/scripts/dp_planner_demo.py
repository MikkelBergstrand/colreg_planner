"""ColregsDPPlanner on a synthetic head-on encounter (no data needed).

The own ship follows the DP reference exactly; the target holds its course
and ignores COLREGs. Prints the reference at every replan.

    python scripts/dp_planner_demo.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from colregs_planner.planner_api import ColregsDPPlanner, Route

DT = 0.5


def target_state(t: float) -> dict:
    """Head-on target, 120 m ahead, holding course south at 1.2 m/s."""
    return {"north": 120.0 - 1.2 * t, "east": 0.0, "course_deg": 180.0, "speed": 1.2}


def main() -> None:
    planner = ColregsDPPlanner(
        own_id = "ego",
        route  = Route(north=0.0, east=0.0, course_deg=0.0, speed=1.5),
        dt     = DT,
        config = {"interaction": {"use_responding_filter": True, "top_k_hypotheses": 2,
                                  "reclassify_in_rollout": False,
                                  "agent": {"d_pass": 30.0, "tcpa_threshold": 100.0,
                                            "los_lookahead": 30.0}}},
        classifier = {"d_u": 45.0, "dcpa_max": 450.0, "d_a": 3000.0},
    )
    own = {"north": 0.0, "east": 0.0, "course_deg": 0.0, "speed": 1.5}
    closest = math.inf
    with planner:
        for k in range(int(90.0 / DT) + 1):
            t = k * DT
            tgt = target_state(t)
            closest = min(closest, math.hypot(tgt["north"] - own["north"], tgt["east"] - own["east"]))
            ref = planner.update(t, {"ego": own, "target": tgt})
            if ref.replanned:
                print(f"t={t:5.1f} s  plan {'/'.join(ref.plan):20s} reference "
                      f"N {ref.north:7.1f}  E {ref.east:6.1f}  heading {ref.heading_deg:5.1f} deg")
            # Stand-in for the DP controller and the ship: reach the next reference.
            nxt = planner.reference(t + DT)
            own = {"north": nxt.north, "east": nxt.east, "course_deg": nxt.heading_deg,
                   "speed": math.hypot(nxt.vn, nxt.ve)}
    print(f"closest approach to the target: {closest:.1f} m (d_pass 30 m)")


if __name__ == "__main__":
    main()
