"""viknes_model.py

Self-contained 3DOF maneuvering model for the Viknes vessel, extracted and
simplified from the colav-simulator by Trym Tengesdal.

The model implements the standard Fossen (2011) maneuvering equations:

    eta_dot = R(psi) @ nu
    nu_dot  = M_inv @ (-C(nu_r) @ nu_r - D(nu_r) @ nu_r + tau)

with eta = [x, y, psi]^T  (NED position + heading)
     nu  = [u, v, r]^T    (surge, sway, yaw-rate in BODY frame)
     tau = [X, Y, N]^T    (generalized forces / moments)

Usage
-----
    sim = ViknesSimulator(dt=0.1)
    state = ViknesState(x=0.0, y=0.0, psi=0.0, u=3.0, v=0.0, r=0.0)
    tau   = np.array([100.0, 0.0, 0.0])          # surge force only
    for _ in range(100):
        state = sim.step(state, tau)

Author: Johannes Robert Skarø (adapted from Trym Tengesdal / colav-simulator)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class ViknesParams:
    """Physical parameters for the Viknes vessel (read-only / fixed)."""

    # Geometry
    length: float = 8.45      # [m]
    width:  float = 2.71      # [m]
    draft:  float = 0.50      # [m]
    l_r:    float = 4.00      # [m]  distance from CG to rudder

    # Speed / rate limits
    U_min:  float = 0.0                      # [m/s]
    U_max:  float = 10.0                     # [m/s]
    r_max:  float = float(np.deg2rad(15.0))  # [rad/s]

    # Force limits [N]
    Fx_min: float = -6_550.0
    Fx_max: float =  13_100.0
    Fy_min: float = -645.0
    Fy_max: float =  645.0

    # Inertia matrices  (3x3, body-fixed)
    M_rb: np.ndarray = field(
        default_factory=lambda: np.diag([3_980.0, 3_980.0, 19_703.0])
    )
    M_a: np.ndarray = field(
        default_factory=lambda: np.zeros((3, 3))
    )

    # Damping matrices
    D_l: np.ndarray = field(   # linear
        default_factory=lambda: np.diag([50.0, 200.0, 1_281.0])
    )
    D_q: np.ndarray = field(   # quadratic
        default_factory=lambda: np.diag([135.0, 2_000.0, 0.0])
    )
    D_c: np.ndarray = field(   # cubic
        default_factory=lambda: np.diag([0.0, 0.0, 3_224.0])
    )

    # Ship polygon (columns = vertices, in body frame) for visualization
    @property
    def ship_vertices(self) -> np.ndarray:
        L, W = self.length, self.width
        return np.array([
            [ L / 2,  W / 2],
            [ L / 2, -W / 2],
            [-L / 2, -W / 2],
            [-L / 2,  W / 2],
        ]).T  # (2, 4)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class ViknesState:
    """6-DOF (planar) state of the Viknes vessel."""

    x:   float = 0.0   # [m]   NED north
    y:   float = 0.0   # [m]   NED east
    psi: float = 0.0   # [rad] heading (0 = north, positive clockwise)
    u:   float = 0.0   # [m/s] surge  (body-fixed forward)
    v:   float = 0.0   # [m/s] sway   (body-fixed starboard)
    r:   float = 0.0   # [rad/s] yaw rate (positive = bow to starboard)

    # ------------------------------------------------------------------ #
    @property
    def eta(self) -> np.ndarray:
        """Position/heading vector [x, y, psi]."""
        return np.array([self.x, self.y, self.psi])

    @property
    def nu(self) -> np.ndarray:
        """Body-fixed velocity vector [u, v, r]."""
        return np.array([self.u, self.v, self.r])

    @property
    def speed(self) -> float:
        """Speed over ground [m/s]."""
        return float(np.hypot(self.u, self.v))

    @property
    def heading_vec(self) -> np.ndarray:
        """Unit vector in the heading direction (NED frame)."""
        return np.array([np.cos(self.psi), np.sin(self.psi)])

    @property
    def ned_velocity(self) -> np.ndarray:
        """Velocity in NED frame [vx, vy]."""
        R = _rot2d(self.psi)
        return R @ np.array([self.u, self.v])

    @classmethod
    def from_ned_speed_heading(
        cls,
        x: float,
        y: float,
        speed: float,
        heading: float,
    ) -> "ViknesState":
        """Convenience constructor: speed [m/s] and heading [rad], no sway/yaw."""
        return cls(x=x, y=y, psi=heading, u=speed, v=0.0, r=0.0)


# ---------------------------------------------------------------------------
# Helper math (no external deps beyond numpy)
# ---------------------------------------------------------------------------

def _rot2d(psi: float) -> np.ndarray:
    """2D rotation matrix R(psi) mapping body → NED."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s], [s, c]])


def _Rmtrx(psi: float) -> np.ndarray:
    """3x3 rotation/transformation matrix for [u,v,r] → [x_dot, y_dot, psi_dot]."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _coriolis_rb(M_rb: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Rigid-body Coriolis matrix C_RB(nu)."""
    m  = M_rb[0, 0]
    Iz = M_rb[2, 2]
    r  = nu[2]
    return np.array([
        [0.0,  -m * r,  0.0],
        [m * r, 0.0,    0.0],
        [0.0,   0.0,    0.0],
    ])


def _coriolis_a(M_a: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Added-mass Coriolis matrix C_A(nu). Valid for diagonal M_a."""
    a11, a22, a33 = M_a[0, 0], M_a[1, 1], M_a[2, 2]
    u, v, r = nu
    return np.array([
        [0.0,    0.0,   a22 * v],
        [0.0,    0.0,  -a11 * u],
        [-a22 * v, a11 * u, 0.0],
    ])


def _damping(D_l, D_q, D_c, nu: np.ndarray) -> np.ndarray:
    """Combined linear + quadratic + cubic damping matrix."""
    abs_nu = np.abs(nu)
    return D_l + D_q * abs_nu[np.newaxis, :] + D_c * (abs_nu ** 2)[np.newaxis, :]


def _wrap(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _sat(val: float, lo: float, hi: float) -> float:
    return float(np.clip(val, lo, hi))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class ViknesSimulator:
    """Discrete-time 3DOF simulator for the Viknes vessel.

    Integrates the maneuvering equations using 4th-order Runge-Kutta.

    Parameters
    ----------
    dt : float
        Timestep [s]. Keep ≤ 0.1 s for numerical stability with RK4.
    params : ViknesParams, optional
        Physical parameters. Defaults to stock Viknes values.
    """

    def __init__(
        self,
        dt: float = 0.1,
        params: ViknesParams | None = None,
    ) -> None:
        self.dt = dt
        self.p  = params or ViknesParams()
        self._Minv = np.linalg.inv(self.p.M_rb + self.p.M_a)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def step(
        self,
        state: ViknesState,
        tau: np.ndarray,
    ) -> ViknesState:
        """Advance the state by one timestep using RK4.

        Parameters
        ----------
        state : ViknesState
            Current state.
        tau : np.ndarray, shape (3,)
            Generalized forces [X, Y, N] in body frame [N, N, Nm].
            Will be saturated to actuator limits internally.

        Returns
        -------
        ViknesState
            Updated state at t + dt.
        """
        tau = self._saturate_tau(tau)
        xs  = np.array([state.x, state.y, state.psi, state.u, state.v, state.r])

        xs_new = self._rk4(xs, tau)

        return ViknesState(
            x=xs_new[0],
            y=xs_new[1],
            psi=_wrap(xs_new[2]),
            u=xs_new[3],
            v=xs_new[4],
            r=xs_new[5],
        )

    def ode(
        self,
        xs: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray:
        """Continuous-time ODE  x_dot = f(x, tau).

        Parameters
        ----------
        xs : np.ndarray, shape (6,)
            State [x, y, psi, u, v, r].
        tau : np.ndarray, shape (3,)
            Generalized forces [X, Y, N] (already saturated).

        Returns
        -------
        np.ndarray, shape (6,)
            Time derivative of the state.
        """
        eta = xs[0:3]
        nu  = xs[3:6]

        C_RB = _coriolis_rb(self.p.M_rb, nu)
        C_A  = _coriolis_a(self.p.M_a,   nu)
        D    = _damping(self.p.D_l, self.p.D_q, self.p.D_c, nu)

        eta_dot = _Rmtrx(eta[2]) @ nu
        nu_dot  = self._Minv @ (-(C_RB + C_A) @ nu - D @ nu + tau)

        xdot = np.empty(6)
        xdot[0:3] = eta_dot
        xdot[3:6] = nu_dot
        return xdot

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _rk4(self, xs: np.ndarray, tau: np.ndarray) -> np.ndarray:
        h = self.dt
        k1 = self.ode(xs,             tau)
        k2 = self.ode(xs + 0.5*h*k1, tau)
        k3 = self.ode(xs + 0.5*h*k2, tau)
        k4 = self.ode(xs +     h*k3, tau)
        return xs + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _saturate_tau(self, tau: np.ndarray) -> np.ndarray:
        tau = tau.copy().astype(float)
        tau[0] = _sat(tau[0], self.p.Fx_min, self.p.Fx_max)
        tau[1] = _sat(tau[1], self.p.Fy_min, self.p.Fy_max)
        tau[2] = _sat(tau[2],
                      self.p.Fy_min * self.p.l_r,
                      self.p.Fy_max * self.p.l_r)
        return tau


# ---------------------------------------------------------------------------
# Simple PD heading + speed controller (useful for scripted scenarios)
# ---------------------------------------------------------------------------

class ViknesPDController:
    """Decoupled PD controller that produces body-frame forces [X, Y, N].

    Useful for scripted scenarios where you want the vessel to follow
    a desired heading and speed, without implementing full autopilot logic.

    Tuning
    ------
    Kp_psi, Kd_psi : heading proportional / derivative gains
    Kp_u            : surge proportional gain
    """

    def __init__(
        self,
        Kp_psi: float = 5_000.0,
        Kd_psi: float = 2_000.0,
        Kp_u:   float = 1_000.0,
    ) -> None:
        self.Kp_psi = Kp_psi
        self.Kd_psi = Kd_psi
        self.Kp_u   = Kp_u

    def compute(
        self,
        state: ViknesState,
        desired_heading: float,
        desired_speed:   float,
    ) -> np.ndarray:
        """Compute generalized forces to reach desired heading and speed.

        Parameters
        ----------
        state : ViknesState
        desired_heading : float  [rad]
        desired_speed   : float  [m/s]

        Returns
        -------
        np.ndarray, shape (3,)  — [X, Y, N]
        """
        psi_err = _wrap(desired_heading - state.psi)
        u_err   = desired_speed - state.u

        X = self.Kp_u   * u_err
        Y = 0.0                          # no sway control
        N = self.Kp_psi * psi_err - self.Kd_psi * state.r

        return np.array([X, Y, N])


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dt  = 0.05
    sim = ViknesSimulator(dt=dt)
    ctl = ViknesPDController()

    # Initial state: heading north, 3 m/s
    state = ViknesState.from_ned_speed_heading(x=0.0, y=0.0, speed=3.0, heading=0.0)

    # Desired: turn to heading 45 deg, maintain speed
    desired_heading = np.deg2rad(45.0)
    desired_speed   = 3.0

    T_sim = 60.0   # seconds
    N     = int(T_sim / dt)

    xs_log   = np.zeros((N, 6))
    tau_log  = np.zeros((N, 3))

    for k in range(N):
        tau = ctl.compute(state, desired_heading, desired_speed)
        xs_log[k]  = [state.x, state.y, state.psi, state.u, state.v, state.r]
        tau_log[k] = tau
        state = sim.step(state, tau)

    t = np.arange(N) * dt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Viknes simulator — turn to 45° heading")

    axes[0, 0].plot(xs_log[:, 1], xs_log[:, 0])   # y=east, x=north
    axes[0, 0].set_xlabel("East [m]"); axes[0, 0].set_ylabel("North [m]")
    axes[0, 0].set_title("Trajectory (NED)"); axes[0, 0].set_aspect("equal")
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, np.rad2deg(xs_log[:, 2]))
    axes[0, 1].axhline(np.rad2deg(desired_heading), ls="--", c="r", label="desired")
    axes[0, 1].set_xlabel("Time [s]"); axes[0, 1].set_ylabel("Heading [deg]")
    axes[0, 1].set_title("Heading"); axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[1, 0].plot(t, xs_log[:, 3], label="u (surge)")
    axes[1, 0].plot(t, xs_log[:, 4], label="v (sway)")
    axes[1, 0].axhline(desired_speed, ls="--", c="r", label="desired")
    axes[1, 0].set_xlabel("Time [s]"); axes[1, 0].set_ylabel("Speed [m/s]")
    axes[1, 0].set_title("Body-frame velocities"); axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(t, np.rad2deg(xs_log[:, 5]))
    axes[1, 1].set_xlabel("Time [s]"); axes[1, 1].set_ylabel("Yaw rate [deg/s]")
    axes[1, 1].set_title("Yaw rate"); axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("outputs/viknes_smoketest.png", dpi=150)
    print("Smoke test done.")