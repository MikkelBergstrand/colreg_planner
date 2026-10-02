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
    state  = Model.step(state, action, dt)
    state  = Model.from_pos_speed_heading(north, east, speed, heading)
    traj   = Model.propagate(state, action, dt, n_steps)

The VO action parameterization u = [v, v*omega] is supported by both.

Author: Johannes Robert Skarø
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union
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
    def position(self) -> np.ndarray:
        return np.array([self.north, self.east])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vn, self.ve])

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vn, self.ve))

    @property
    def heading(self) -> float:
        """Course = heading [rad], NED. 0=north, +ve clockwise."""
        return float(np.arctan2(self.ve, self.vn))

    @property
    def heading_vec(self) -> np.ndarray:
        h = self.heading
        return np.array([np.cos(h), np.sin(h)])

    @property
    def heading_perp(self) -> np.ndarray:
        """Starboard-perpendicular unit vector."""
        h = self.heading
        return np.array([-np.sin(h), np.cos(h)])

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.north, self.east, self.vn, self.ve])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "CTCoupledState":
        return cls(north=x[0], east=x[1], vn=x[2], ve=x[3])

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

    @staticmethod
    def step_from_action(state: CTCoupledState, u: np.ndarray, dt: float) -> CTCoupledState:
        """Propagate from VO action u = [v, v*omega]."""
        v     = float(u[0])
        omega = float(u[1] / u[0]) if abs(u[0]) > 1e-6 else 0.0
        # Rebuild speed in current heading direction
        psi   = state.heading
        s     = CTCoupledState(north=state.north, east=state.east,
                         vn=v * np.cos(psi), ve=v * np.sin(psi))
        return CTCoupledModel.step(s, omega, dt)

    @staticmethod
    def action_from_state(state: CTCoupledState, omega: float = 0.0) -> np.ndarray:
        """Extract VO action u = [v, v*omega] from state."""
        v = state.speed
        return np.array([v, v * omega])

    @staticmethod
    def action_matrix(state: CTCoupledState, t: float) -> np.ndarray:
        """Linearized CT action matrix A(t): p(t) ≈ p0 + A(t) @ u."""
        psi = state.heading
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [t * c,  -(t**2 / 2) * s],
            [t * s,   (t**2 / 2) * c],
        ])

    @staticmethod
    def jacobian(state: CTCoupledState, omega: float, dt: float) -> np.ndarray:
        """Linearized Jacobian F = dx_{k+1}/dx_k (4x4)."""
        v   = state.speed
        psi = state.heading
        if abs(omega) < 1e-6:
            return np.array([
                [1, 0, dt,  0],
                [0, 1,  0, dt],
                [0, 0,  1,  0],
                [0, 0,  0,  1],
            ], dtype=float)
        else:
            psi_new = psi + omega * dt
            F = np.eye(4)
            F[0, 2] =  (np.sin(psi_new) - np.sin(psi)) / omega
            F[0, 3] = -(np.cos(psi_new) - np.cos(psi)) / omega
            F[1, 2] =  (np.cos(psi) - np.cos(psi_new)) / omega
            F[1, 3] =  (np.sin(psi_new) - np.sin(psi)) / omega
            return F

    @staticmethod
    def propagate(
        state: CTCoupledState, omega: float, dt: float, n_steps: int
    ) -> list[CTCoupledState]:
        traj = [state]
        for _ in range(n_steps):
            state = CTCoupledModel.step(state, omega, dt)
            traj.append(state)
        return traj


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
    def position(self) -> np.ndarray:
        return np.array([self.north, self.east])

    @property
    def speed(self) -> float:
        return abs(self.v)

    @property
    def heading(self) -> float:
        return _wrap(self.psi)

    @property
    def heading_vec(self) -> np.ndarray:
        return np.array([np.cos(self.psi), np.sin(self.psi)])

    @property
    def heading_perp(self) -> np.ndarray:
        return np.array([-np.sin(self.psi), np.cos(self.psi)])

    @property
    def ned_velocity(self) -> np.ndarray:
        """Velocity in NED frame (surge only, no sway)."""
        return self.v * self.heading_vec

    @property
    def course(self) -> float:
        """Course over ground [rad] — direction of NED velocity."""
        vned = self.ned_velocity
        return float(np.arctan2(vned[1], vned[0]))

    @property
    def as_array(self) -> np.ndarray:
        return np.array([self.north, self.east, self.psi, self.v, self.r])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "CTDecoupledState":
        return cls(north=x[0], east=x[1], psi=x[2], v=x[3], r=x[4])

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

    @staticmethod
    def step_from_action(state: CTDecoupledState, u: np.ndarray, dt: float) -> CTDecoupledState:
        """Propagate from VO action u = [v, v*r]."""
        v = float(u[0])
        r = float(u[1] / u[0]) if abs(u[0]) > 1e-6 else 0.0
        s = CTDecoupledState(north=state.north, east=state.east,
                         psi=state.psi, v=v, r=r)
        return CTDecoupledModel.step(s, dt)

    @staticmethod
    def action_from_state(state: CTDecoupledState) -> np.ndarray:
        """Extract VO action u = [v, v*r] from state."""
        return np.array([state.v, state.v * state.r])

    @staticmethod
    def action_matrix(state: CTDecoupledState, t: float) -> np.ndarray:
        """Linearized CT action matrix A(t): p(t) ≈ p0 + A(t) @ u.

        Uses heading psi directly (not course), which is the correct
        linearization point for a heading-explicit model.
        """
        psi  = state.psi
        c, s = np.cos(psi), np.sin(psi)
        return np.array([
            [t * c,  -(t**2 / 2) * s],
            [t * s,   (t**2 / 2) * c],
        ])

    @staticmethod
    def jacobian(state: CTDecoupledState, dt: float) -> np.ndarray:
        """Linearized Jacobian F = dx_{k+1}/dx_k (5x5)."""
        v, psi, r = state.v, state.psi, state.r
        if abs(r) < 1e-6:
            F = np.eye(5)
            F[0, 2] = -v * np.sin(psi) * dt
            F[0, 3] =  np.cos(psi) * dt
            F[1, 2] =  v * np.cos(psi) * dt
            F[1, 3] =  np.sin(psi) * dt
            F[2, 4] =  dt
        else:
            psi_new = psi + r * dt
            ir = 1.0 / r
            F = np.eye(5)
            # d(north)/d(psi)
            F[0, 2] = ir * (np.cos(psi_new) - np.cos(psi))
            # d(north)/d(v)
            F[0, 3] = ir * (np.sin(psi_new) - np.sin(psi))
            # d(north)/d(r)
            F[0, 4] = (v * ir) * (dt * np.cos(psi_new) - ir * (np.sin(psi_new) - np.sin(psi)))
            # d(east)/d(psi)
            F[1, 2] = ir * (np.sin(psi_new) - np.sin(psi))
            # d(east)/d(v)
            F[1, 3] = -ir * (np.cos(psi_new) - np.cos(psi))
            # d(east)/d(r)
            F[1, 4] = (v * ir) * (dt * np.sin(psi_new) + ir * (np.cos(psi_new) - np.cos(psi)))
            # d(psi)/d(r)
            F[2, 4] = dt
        return F

    @staticmethod
    def propagate(
        state: CTDecoupledState, dt: float, n_steps: int,
        r_schedule: list[float] | None = None,
    ) -> list[CTDecoupledState]:
        """Propagate forward. Optionally supply a turn-rate schedule."""
        traj = [state]
        for k in range(n_steps):
            if r_schedule is not None:
                state = CTDecoupledState(
                    north=state.north, east=state.east,
                    psi=state.psi, v=state.v,
                    r=r_schedule[k] if k < len(r_schedule) else state.r,
                )
            state = CTDecoupledModel.step(state, dt)
            traj.append(state)
        return traj


# ---------------------------------------------------------------------------
# Union type for type hints
# ---------------------------------------------------------------------------

CTModelState = Union[CTCoupledState, CTDecoupledState]