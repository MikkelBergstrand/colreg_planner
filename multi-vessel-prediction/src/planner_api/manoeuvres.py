"""planner_api/manoeuvres.py

Discrete own-ship manoeuvres, the actions a symbolic planner chooses between.

A manoeuvre alters course and/or speed relative to the vessel's nominal
velocity for duration_s, then returns to the nominal course (a parallel
offset track). It starts now or after a delay. PredictionService.
evaluate_manoeuvre() rolls one out against every responsiveness hypothesis,
with the other vessels reacting to it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Manoeuvre:
    name: str
    course_change_deg: float = 0.0   # positive = starboard
    speed_factor: float = 1.0        # multiplier on the nominal speed
    duration_s: float = 180.0        # time before returning to the nominal course

    @property
    def keeps_course_and_speed(self) -> bool:
        return self.course_change_deg == 0.0 and self.speed_factor == 1.0

    @property
    def alters_to_port(self) -> bool:
        return self.course_change_deg < 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "course_change_deg": self.course_change_deg,
            "speed_factor": self.speed_factor,
            "duration_s": self.duration_s,
        }


# In order of preference (see ColregsPlanningProblem.solve): no action, then
# course alterations large enough to be readily apparent (Rule 8(b)),
# smallest first, then speed, then port as a last resort.
DEFAULT_MANOEUVRES: tuple[Manoeuvre, ...] = (
    Manoeuvre("hold"),
    Manoeuvre("stbd_30", course_change_deg=30.0),
    Manoeuvre("stbd_45", course_change_deg=45.0),
    Manoeuvre("stbd_70", course_change_deg=70.0),
    Manoeuvre("slow_50", speed_factor=0.5),
    Manoeuvre("port_30", course_change_deg=-30.0),
)


def velocity_plan(
    nominal: np.ndarray, m: Manoeuvre, dt: float, start_s: float = 0.0,
) -> np.ndarray:
    """Apply manoeuvre *m*, starting after start_s, to a (n, 2) nominal [vn, ve] sequence."""
    plan = np.array(nominal, dtype=float, copy=True)
    k_start = min(len(plan), int(round(start_s / dt)))
    k_end = min(len(plan), k_start + int(round(m.duration_s / dt)))
    a = math.radians(m.course_change_deg)
    # NED, course clockwise from north: rotating by +a turns to starboard.
    rot = np.array([[math.cos(a), -math.sin(a)],
                    [math.sin(a),  math.cos(a)]])
    plan[k_start:k_end] = (plan[k_start:k_end] @ rot.T) * m.speed_factor
    return plan
