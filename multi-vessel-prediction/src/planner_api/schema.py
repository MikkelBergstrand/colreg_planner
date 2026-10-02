"""planner_api/schema.py

Plain, JSON-serializable output types for downstream planners.

Conventions used in every type here:
  - positions are NED metres relative to the session origin (north, east)
  - courses and bearings are degrees in [0, 360), 0 = north, clockwise
  - speeds are m/s, times are seconds
  - vessel ids are the strings passed to PredictionService.step()

Every type has ``to_dict()`` returning only built-in types (dict, list, str,
float, int, bool, None), so the result can go straight into json.dumps() or
an LLM tool result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def _round(x: float | None, nd: int = 2) -> float | None:
    return None if x is None else round(float(x), nd)


@dataclass
class TrajectoryPoint:
    t: float          # time relative to the snapshot [s]
    north: float      # [m]
    east: float       # [m]
    course_deg: float
    speed: float      # [m/s]

    def to_dict(self) -> dict:
        return {
            "t": _round(self.t, 1),
            "north": _round(self.north, 1),
            "east": _round(self.east, 1),
            "course_deg": _round(self.course_deg, 1),
            "speed": _round(self.speed, 2),
        }


@dataclass
class RoleAssignment:
    """One vessel's COLREGs obligation toward one counterpart."""
    counterpart: str
    role: str                     # give_way | stand_on | head_on | overtaking | avoid_collision
    encounter_type: str | None    # geometry from this vessel's perspective, e.g. crossing_starboard

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VesselStatus:
    """Current observed state and inferred behaviour of one vessel."""
    vessel_id: str
    north: float
    east: float
    course_deg: float
    speed: float
    yaw_rate_deg_s: float
    moving: bool                  # False -> below min speed, left out of the interaction rollout
    p_responsive: float | None    # P(vessel follows its COLREGs obligations); None when the filter is off
    roles: list[RoleAssignment] = field(default_factory=list)

    @property
    def in_encounter(self) -> bool:
        return bool(self.roles)

    @property
    def burdened(self) -> bool:
        """Holds an obligation to act (any role other than stand_on)."""
        return any(r.role != "stand_on" for r in self.roles)

    def to_dict(self) -> dict:
        return {
            "vessel_id": self.vessel_id,
            "north": _round(self.north, 1),
            "east": _round(self.east, 1),
            "course_deg": _round(self.course_deg, 1),
            "speed": _round(self.speed, 2),
            "yaw_rate_deg_s": _round(self.yaw_rate_deg_s, 3),
            "moving": self.moving,
            "p_responsive": _round(self.p_responsive, 4),
            "in_encounter": self.in_encounter,
            "burdened": self.burdened,
            "roles": [r.to_dict() for r in self.roles],
        }


@dataclass
class Encounter:
    """A classified COLREGs encounter between two vessels, with current CPA geometry.

    Roles and encounter_type are from the classifier at the current time
    (before any responsiveness override). CPA values assume both vessels
    hold their current velocity.
    """
    vessel_a: str
    vessel_b: str
    encounter_type: str | None    # from vessel_a's perspective
    role_a: str
    role_b: str
    range_m: float
    bearing_a_to_b_deg: float     # true bearing
    relative_bearing_a_to_b_deg: float  # relative to vessel_a's course, [0, 360)
    dcpa_m: float
    tcpa_s: float | None          # None -> no relative motion; negative -> diverging

    def to_dict(self) -> dict:
        return {
            "vessel_a": self.vessel_a,
            "vessel_b": self.vessel_b,
            "encounter_type": self.encounter_type,
            "role_a": self.role_a,
            "role_b": self.role_b,
            "range_m": _round(self.range_m, 1),
            "bearing_a_to_b_deg": _round(self.bearing_a_to_b_deg, 1),
            "relative_bearing_a_to_b_deg": _round(self.relative_bearing_a_to_b_deg, 1),
            "dcpa_m": _round(self.dcpa_m, 1),
            "tcpa_s": _round(self.tcpa_s, 1),
        }


@dataclass
class Hypothesis:
    """One responsiveness hypothesis and the multi-vessel future it implies."""
    rank: int                                   # 0 = most likely
    probability: float                          # joint prior weight of this assignment
    assumptions: dict[str, str]                 # vessel id -> "responsive" | "non_responsive"
    effective_roles: list[dict]                 # roles after the non-responsive override
    trajectories: dict[str, list[TrajectoryPoint]]

    def to_dict(self, include_trajectories: bool = True) -> dict:
        d = {
            "rank": self.rank,
            "probability": _round(self.probability, 4),
            "assumptions": dict(self.assumptions),
            "effective_roles": list(self.effective_roles),
        }
        if include_trajectories:
            d["trajectories"] = {
                vid: [p.to_dict() for p in pts] for vid, pts in self.trajectories.items()
            }
        return d


@dataclass
class PredictedCPA:
    """Closest approach between two vessels along their predicted trajectories."""
    vessel_id: str
    other_id: str
    hypothesis_rank: int
    min_distance_m: float
    t_min_s: float
    position_vessel: tuple[float, float]   # (north, east) of vessel_id at t_min
    position_other: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "vessel_id": self.vessel_id,
            "other_id": self.other_id,
            "hypothesis_rank": self.hypothesis_rank,
            "min_distance_m": _round(self.min_distance_m, 1),
            "t_min_s": _round(self.t_min_s, 1),
            "position_vessel": [_round(v, 1) for v in self.position_vessel],
            "position_other": [_round(v, 1) for v in self.position_other],
        }


@dataclass
class Snapshot:
    """Everything the predictor knows at one time step."""
    t: float
    step: int
    dt: float
    horizon_s: float
    vessels: list[VesselStatus]
    encounters: list[Encounter]
    hypotheses: list[Hypothesis]
    nominal: dict[str, list[TrajectoryPoint]] = field(default_factory=dict)

    def to_dict(
        self,
        include_trajectories: bool = True,
        include_nominal: bool = False,
    ) -> dict:
        d = {
            "t": _round(self.t, 2),
            "step": self.step,
            "dt": self.dt,
            "horizon_s": self.horizon_s,
            "units": {
                "position": "NED metres from session origin",
                "angles": "degrees, 0=north, clockwise",
                "speed": "m/s",
                "time": "s",
            },
            "vessels": [v.to_dict() for v in self.vessels],
            "encounters": [e.to_dict() for e in self.encounters],
            "hypotheses": [h.to_dict(include_trajectories) for h in self.hypotheses],
        }
        if include_nominal:
            d["nominal"] = {
                vid: [p.to_dict() for p in pts] for vid, pts in self.nominal.items()
            }
        return d
