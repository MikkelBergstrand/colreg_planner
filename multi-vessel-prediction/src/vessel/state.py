"""vessel/state.py

Standard vessel state and action types, independent of simulation model or policy.
All models and policies operate exclusively on these types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Union

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class InitialState:
    """Minimal kinematic specification used to initialise a simulation model.

    Intentionally simple — holds only what is needed to construct any model
    adapter. Decoupled from scenario geometry and visualization concerns.
    """

    north:   float = 0.0   # [m]
    east:    float = 0.0   # [m]
    speed:   float = 0.0   # [m/s]
    heading: float = 0.0   # [rad]  NED, 0 = north, positive clockwise

    @property
    def position(self) -> np.ndarray:
        return np.array([self.north, self.east])

    @property
    def vn(self) -> float:
        return self.speed * np.cos(self.heading)

    @property
    def ve(self) -> float:
        return self.speed * np.sin(self.heading)

    @classmethod
    def from_pos_speed_heading(
        cls, north: float, east: float, speed: float, heading: float,
    ) -> "InitialState":
        return cls(north=north, east=east, speed=speed, heading=heading)


@dataclass
class VesselState:
    """Model-agnostic vessel state in NED frame.

    north : NED north position [m]
    east  : NED east position  [m]
    psi   : heading [rad], 0 = north, positive clockwise
    speed : scalar speed [m/s]
    r     : yaw rate [rad/s], positive = clockwise
    """

    north: float
    east:  float
    psi:   float
    speed: float
    r:     float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.north, self.east])

    @property
    def heading(self) -> float:
        """Heading wrapped to (-pi, pi]."""
        return wrap_angle(self.psi)

    @property
    def vn(self) -> float:
        return self.speed * np.cos(self.psi)

    @property
    def ve(self) -> float:
        return self.speed * np.sin(self.psi)

@dataclass
class CTAction:
    """Action for constant-turn kinematic models.

    speed     : desired scalar speed [m/s]
    turn_rate : desired yaw rate [rad/s], positive = clockwise (NED)
    """

    speed:     float
    turn_rate: float


@dataclass
class CVAction:
    """Action for constant-velocity kinematic models.

    Directly commands a velocity vector in NED frame.
    vn : desired north velocity [m/s]
    ve : desired east  velocity [m/s]
    """

    vn: float
    ve: float

    @property
    def speed(self) -> float:
        return float(np.sqrt(self.vn ** 2 + self.ve ** 2))

    @property
    def heading(self) -> float:
        """Heading [rad] implied by the velocity vector, NED."""
        return float(np.arctan2(self.ve, self.vn))


@dataclass
class DPAction:
    """Action for dynamic positioning on a moving reference.

    Moves the DP reference by one step over the coming dt, in NED frame.
    d_north, d_east : reference displacement over the step [m]
    psi_ref         : reference heading at the end of the step [rad], NED
    ref_start       : optional (north, east, psi) the reference is at when the
                      step starts. Given, the reference follows that track
                      (e.g. a plan, re-anchored at each replan); omitted, it
                      continues from wherever the previous steps left it.
    """

    d_north:   float
    d_east:    float
    psi_ref:   float
    ref_start: Optional[tuple[float, float, float]] = None


Action = Union[CTAction, CVAction, DPAction]


# ---------------------------------------------------------------------------
# Vessel intent and initialisation (used by scenario builders)
# ---------------------------------------------------------------------------

class Intent(Enum):
    RESPONSIVE     = auto()
    NON_RESPONSIVE = auto()


@dataclass
class VesselInit:
    """Initial conditions and responsiveness of one vessel."""

    vessel_id:      str
    state:          InitialState
    intent:         Intent = Intent.RESPONSIVE

    @property
    def position(self) -> np.ndarray:
        return self.state.position

    @property
    def speed(self) -> float:
        return self.state.speed

    @property
    def heading(self) -> float:
        return self.state.heading
