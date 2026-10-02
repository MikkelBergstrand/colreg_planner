"""Demo of the planner API on a synthetic three-vessel encounter (no data needed).

ego heads north, a vessel crosses from ego's starboard side and a third
comes head-on. All three hold course, so the vessels with an obligation to
act are gradually inferred non-responsive.

    python scripts/planner_api_demo.py               # print query results
    python scripts/planner_api_demo.py --tools       # print the tool definitions
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.planner_api import TOOLS, PredictionService, call_tool

DT = 2.0

# vessel id -> (north0, east0, course_deg, speed)
VESSELS = {
    "ego":      (0.0,    0.0,    0.0,   5.0),
    "crosser":  (1000.0, 1000.0, 270.0, 5.0),
    "oncoming": (1600.0, 40.0,   180.0, 4.0),
}


def observations(t: float) -> dict[str, dict]:
    obs = {}
    for vid, (n0, e0, crs, spd) in VESSELS.items():
        c = math.radians(crs)
        obs[vid] = {
            "north": n0 + spd * math.cos(c) * t,
            "east":  e0 + spd * math.sin(c) * t,
            "course_deg": crs,
            "speed": spd,
        }
    return obs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", action="store_true", help="print the tool definitions and exit")
    ap.add_argument("--steps", type=int, default=40, help="observation steps to run")
    args = ap.parse_args()

    if args.tools:
        print(json.dumps(TOOLS, indent=2))
        return

    service = PredictionService.from_config(
        {
            "H": 90,
            "interaction": {
                "enabled": True,
                "use_responding_filter": True,
                "top_k_hypotheses": 2,
                "agent": {"d_pass": 150.0, "tcpa_threshold": 600.0},
            },
        },
        dt=DT,
    )

    for k in range(args.steps):
        service.step(observations(k * DT), t=k * DT)

    calls = [
        ("get_responsiveness", {}),
        ("get_encounters", {"vessel_id": "ego"}),
        ("get_hypotheses", {}),
        ("get_predicted_cpa", {"vessel_id": "ego"}),
        ("get_predicted_trajectory", {"vessel_id": "ego", "resolution_s": 30}),
        ("get_predicted_state", {"vessel_id": "crosser", "t": 60}),
        ("get_vessel", {"vessel_id": "nobody"}),          # error example
    ]
    print(f"t = {service.snapshot.t:.0f} s after {args.steps} steps\n")
    for name, arguments in calls:
        print(f"--- {name}({json.dumps(arguments)})")
        print(json.dumps(call_tool(service, name, arguments), indent=2))
        print()


if __name__ == "__main__":
    main()
