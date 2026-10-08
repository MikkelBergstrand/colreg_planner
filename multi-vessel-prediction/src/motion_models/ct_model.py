"""models/ct_model.py

Two coordinated-turn kinematic models for maritime vessel motion:

  ModelA — COG-implicit:  state = [north, east, vn, ve]
      Heading is always aligned with velocity (course = heading).
      Simple, fewer states, good for tracking in open water.

  ModelB — Heading-explicit: state = [north, east, psi, v, r]
      Heading psi and speed v are independent states; turn rate r = psi_dot.
      Heading can differ from course (sideslip, current, drift).
      More physically faithful for maneuvering vessels.

Both expose the same interface:
    state  = Model.step(state, ..., dt)
    state  = State.from_pos_speed_heading(north, east, speed, heading)

Author: Johannes Robert Skarø
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _wrap(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


# ---------------------------------------------------------------------------
# CT-Coupled (course == heading):  [north, east, vn, ve]
# ---------------------------------------------------------------------------

@dataclass
class CTCoupledState:
    """Kinematic state for Model A.

    Heading is implicitly defined as atan2(ve, vn) — always aligned with
    the velocity vector (course over ground = heading).
    """

    north: float = 0.0
    east:  float = 0.0
    vn:    float = 0.0   # velocity north [m/s]
    ve:    float = 0.0   # velocity east  [m/s]

    # --- Derived quantities ---

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vn, self.ve))

    @property
    def heading(self) -> float:
        """Course = heading [rad], NED. 0=north, +ve clockwise."""
        return float(np.arctan2(self.ve, self.vn))

    @classmethod
    def from_pos_speed_heading(
        cls, north: float, east: float, speed: float, heading: float
    ) -> "CTCoupledState":
        return cls(
            north=north, east=east,
            vn=speed * np.cos(heading),
            ve=speed * np.sin(heading),
        )


class CTCoupledModel:
    """COG-implicit CT model.

    State:  [north, east, vn, ve]
    Action: u = [v, v*omega]  (speed, speed × turn-rate)

    Heading is derived from velocity; course == heading by construction.
    """

    @staticmethod
    def step(state: CTCoupledState, omega: float, dt: float) -> CTCoupledState:
        """Exact CT integration one step forward.

        Parameters
        ----------
        state : CTCoupledState
        omega : float   Turn rate [rad/s]. Positive = clockwise (NED).
        dt    : float   Timestep [s].
        """
        v   = state.speed
        psi = state.heading

        if abs(omega) < 1e-6:
            new_north = state.north + v * np.cos(psi) * dt
            new_east  = state.east  + v * np.sin(psi) * dt
            new_psi   = psi
        else:
            new_psi   = psi + omega * dt
            new_north = state.north + (v / omega) * (np.sin(new_psi) - np.sin(psi))
            new_east  = state.east  - (v / omega) * (np.cos(new_psi) - np.cos(psi))

        return CTCoupledState(
            north=new_north, east=new_east,
            vn=v * np.cos(new_psi),
            ve=v * np.sin(new_psi),
        )

# ---------------------------------------------------------------------------
# CT-Decoupled (heading independent of course):  [north, east, psi, v, r]
# ---------------------------------------------------------------------------

@dataclass
class CTDecoupledState:
    """
    Heading psi is an independent state — it can differ from the course
    (direction of velocity vector) due to sideslip, current, or drift.

    Turn rate r = psi_dot [rad/s] drives heading change directly.
    Speed v is along the heading direction (body-frame surge).
    """

    north: float = 0.0
    east:  float = 0.0
    psi:   float = 0.0   # heading [rad], NED, 0=north, +ve clockwise
    v:     float = 0.0   # speed [m/s]  (body-frame surge)
    r:     float = 0.0   # turn rate [rad/s]

    # --- Derived quantities ---

    @property
    def speed(self) -> float:
        return abs(self.v)

    @property
    def heading(self) -> float:
        return _wrap(self.psi)

    @classmethod
    def from_pos_speed_heading(
        cls, north: float, east: float, speed: float, heading: float
    ) -> "CTDecoupledState":
        return cls(north=north, east=east, psi=heading, v=speed, r=0.0)


class CTDecoupledModel:
    """Heading-explicit CT model.

    State:  [north, east, psi, v, r]
    Action: u = [v, v*r]  (same VO parameterization as Model A)

    Heading psi evolves independently of velocity. This allows heading to
    differ from course, which is physically relevant when:
      - Vessels are subject to currents or wind
      - Vessels use rudder/thruster to maintain heading while drifting
      - Comparing with tracker outputs that estimate heading separately

    The propagation assumes pure surge (no sway) — course == heading
    in the absence of external disturbances. The distinction is that
    heading is an explicit integrated state rather than derived from velocity.
    """

    @staticmethod
    def step(state: CTDecoupledState, dt: float) -> CTDecoupledState:
        """Exact CT integration using state's own turn rate r.

        Unlike CTCoupledModel.step(), this uses r from the state directly.
        """
        v   = state.v
        psi = state.psi
        r   = state.r

        if abs(r) < 1e-6:
            new_north = state.north + v * np.cos(psi) * dt
            new_east  = state.east  + v * np.sin(psi) * dt
            new_psi   = psi
        else:
            new_psi   = psi + r * dt
            new_north = state.north + (v / r) * (np.sin(new_psi) - np.sin(psi))
            new_east  = state.east  - (v / r) * (np.cos(new_psi) - np.cos(psi))

        return CTDecoupledState(
            north=new_north,
            east=new_east,
            psi=_wrap(new_psi),
            v=v,
            r=r,
        )
