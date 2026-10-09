"""milliampere_model.py

3DOF maneuvering model of the milliAmpere autonomous ferry, with the heading
and speed control loop used to drive it.

Ported from liteAmpere (src/models.py, src/course_controller.py,
src/speed_controller.py, src/math/pid.py, src/reference_filter.py,
src/dynamic_positioning.py, src/math/integrator.py). The hydrodynamic coefficients, controller gains and
the thruster mapping are unchanged. It is copied rather than imported, so the
package does not depend on liteAmpere.

The model implements the Fossen (2011) maneuvering equations

    eta_dot = R(psi) @ nu
    nu_dot  = M_inv @ (-(C(nu) + D(nu)) @ nu + tau)

with eta = [north, east, psi]^T, nu = [u, v, r]^T (BODY) and tau = [X, Y, N]^T.

The coefficients were identified for the ferry's operating envelope; liteAmpere
cruises at 0.5 m/s and the speed loop tracks up to about 2 m/s. Above that the
required surge force grows past what the real thrusters deliver, so scenarios
driving this model should keep speeds in that range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _ssa(angle: float) -> float:
    """Smallest signed angle, wrapped to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Kinetics  (liteAmpere src/models.py: KineticsModel)
# ---------------------------------------------------------------------------

class MilliAmpereKinetics:
    """milliAmpere 3DOF kinetics. State xx = [north, east, psi, u, v, r]."""

    def __init__(self) -> None:
        self.I_z = 3826.78
        self.m_11 = 2390.0
        self.m_12 = 0
        self.m_13 = 0
        self.m_21 = 0
        self.m_22 = 2448.0
        self.m_23 = 268.1
        self.m_31 = 0
        self.m_32 = 268.1
        self.m_33 = 4862.0
        self.X_u = -106.6
        self.X_v = 0
        self.X_r = 0
        self.X_auu = -21.39
        self.X_uuu = -37.43
        self.Y_u = 0
        self.Y_v = -29.44
        self.Y_avv = -172.9
        self.Y_vvv = -1.338
        self.Y_arv = -1517
        self.Y_r = 62.58
        self.Y_avr = 488.7
        self.Y_arr = -198.2
        self.N_v = 7.34
        self.N_avv = -4.352
        self.N_arv = 437.8
        self.N_r = -142.7
        self.N_arr = -831.7
        self.N_rrr = 0.0
        self.N_avr = -122
        self.Y_ur = 77.58
        self.N_uv = -90.97
        self.N_ur = 178.5
        self.Nr_dot = -5068.910 + self.I_z

        self.N_u = 0

        self.M = np.array(
            [
                [self.m_11, self.m_12, self.m_13],
                [self.m_21, self.m_22, self.m_23],
                [self.m_31, self.m_32, self.m_33],
            ]
        )
        self.M_inv = np.linalg.inv(self.M)

    def coriolis(self, nu: np.ndarray) -> np.ndarray:
        u, v, r = nu
        c_13 = (
            -0.5 * (self.m_21 + self.m_12) * u
            - self.m_22 * v
            - 0.5 * (self.m_23 + self.m_32) * r
        )
        c_23 = (
            self.m_11 * u
            + 0.5 * (self.m_21 + self.m_12) * v
            + 0.5 * (self.m_13 + self.m_31) * r
        )
        return np.array([
            [0.0,   0.0,   c_13],
            [0.0,   0.0,   c_23],
            [-c_13, -c_23, 0.0],
        ])

    def surge_damping(self, u: float) -> float:
        """d_11: the surge entry of D(nu)."""
        return -self.X_u - self.X_auu * abs(u) - self.X_uuu * u ** 2

    def damping(self, nu: np.ndarray) -> np.ndarray:
        u, v, r = nu
        d_22 = -self.Y_v - self.Y_avv * abs(v) - self.Y_arv * abs(r) - self.Y_vvv * v ** 2
        d_23 = -self.Y_r - self.Y_avr * abs(v) - self.Y_arr * abs(r) - self.Y_ur * u
        d_32 = -self.N_v - self.N_avv * abs(v) - self.N_arv * abs(r) - self.N_uv * u
        d_33 = -self.N_r - self.N_avr * abs(v) - self.N_arr * abs(r) - self.N_ur * u
        return np.array([
            [self.surge_damping(u), -self.X_v, -self.X_r],
            [-self.Y_u,             d_22,      d_23],
            [-self.N_u,             d_32,      d_33],
        ])

    def derivative(self, xx: np.ndarray, tau: np.ndarray) -> np.ndarray:
        psi = xx[2]
        nu  = xx[3:6]
        u, v, r = nu

        dnorth = np.cos(psi) * u - np.sin(psi) * v
        deast  = np.sin(psi) * u + np.cos(psi) * v

        tau_internal = -(self.damping(nu) + self.coriolis(nu)) @ nu
        dnu = self.M_inv @ (tau_internal + tau)
        return np.array([dnorth, deast, r, dnu[0], dnu[1], dnu[2]])


def rk4(xx: np.ndarray, x_dot, dt: float) -> np.ndarray:
    k1 = x_dot(xx)
    k2 = x_dot(xx + 0.5 * dt * k1)
    k3 = x_dot(xx + 0.5 * dt * k2)
    k4 = x_dot(xx + dt * k3)
    return xx + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# Control loop  (liteAmpere main.py, path-following branch)
# ---------------------------------------------------------------------------

@dataclass
class MilliAmpereControlParams:
    """Gains of liteAmpere's heading and speed loops."""

    course_bandwidth: float = 0.5                      # omega of the Nomoto pole placement [rad/s]
    course_zeta:      float = 1.0
    delta_max:        float = float(np.deg2rad(35.0))  # rudder-equivalent limit [rad]
    T_prop:           float = 500.0                    # [N] thrust per unit delta
    l_x:              float = 1.8                      # [m] lever arm of the steering thrust
    sway_ratio:       float = 0.1                      # Y = sway_ratio * N
    speed_kp:         float = 100.0
    speed_ki:         float = 10.0
    ref_omega_yaw:    float = 0.5                      # heading reference filter bandwidth [rad/s]


@dataclass
class MilliAmpereDPParams:
    """Gains of liteAmpere's DynamicPositioning (Ki = 0 there as well).

    Kp acts on the NED position/heading error; Kd on the NED velocity error.
    """

    Kp: tuple[float, float, float] = (35.0, 100.0, 160.0)
    Kd: tuple[float, float, float] = (10.0, 900.0, 1200.0)


def _rot(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class MilliAmpereState:
    """Full simulator state: hull state plus controller memory."""

    xx:             np.ndarray                          # [north, east, psi, u, v, r]
    speed_integral: float = 0.0                         # speed PI integrator [m]
    ref:            np.ndarray = field(                 # yaw reference filter [psi_d, r_d, r_dot_d]
        default_factory=lambda: np.zeros(3)
    )


class MilliAmpereSimulator:
    """Integrates the kinetics under liteAmpere's heading and speed control.

    step() holds a commanded heading and speed for one outer time step and
    integrates with fixed internal steps of ``inner_dt`` (liteAmpere runs at
    100 Hz). The outer dt must be a whole multiple of inner_dt.

    Differences from liteAmpere's loop, all at the interface:
      * Only the yaw axis of the 3-axis reference filter is kept; the other two
        axes are unused in path following.
      * The heading command is fed to the filter as the nearest equivalent
        angle to the filter's own state, so a command across +-180 deg turns
        the short way instead of unwinding a full revolution.
      * reset() pre-loads the speed integrator with the thrust that holds the
        initial speed, so a vessel starting at speed does not first decelerate
        while the integrator winds up from zero.
    """

    def __init__(
        self,
        inner_dt:  float = 0.01,
        params:    MilliAmpereControlParams | None = None,
        dp_params: MilliAmpereDPParams | None = None,
    ) -> None:
        self.inner_dt  = inner_dt
        self.params    = params or MilliAmpereControlParams()
        self.dp_params = dp_params or MilliAmpereDPParams()
        self.model     = MilliAmpereKinetics()
        self._dp_Kp    = np.diag(self.dp_params.Kp)
        self._dp_Kd    = np.diag(self.dp_params.Kd)

        p = self.params
        # Nomoto pole placement (liteAmpere CourseController)
        K = 1.0 / -self.model.N_r
        T = (self.model.I_z - self.model.Nr_dot) / -self.model.N_r
        self.Kp_psi = p.course_bandwidth ** 2 * T / K
        self.Kd_psi = (2 * p.course_zeta * p.course_bandwidth * T - 1) / K

        w = p.ref_omega_yaw
        self._ref_A = np.array([
            [0.0,       1.0,         0.0],
            [0.0,       0.0,         1.0],
            [-(w ** 3), -3 * w ** 2, -3 * w],
        ])
        self._ref_B = np.array([0.0, 0.0, w ** 3])

    def reset(
        self, north: float, east: float, psi: float, speed: float,
    ) -> MilliAmpereState:
        xx = np.array([north, east, psi, speed, 0.0, 0.0], dtype=float)
        X_trim = self.model.surge_damping(speed) * speed
        return MilliAmpereState(
            xx             = xx,
            speed_integral = X_trim / self.params.speed_ki,
            ref            = np.array([psi, 0.0, 0.0]),
        )

    def _n_inner(self, dt: float) -> int:
        n_inner = round(dt / self.inner_dt)
        if n_inner < 1 or abs(n_inner * self.inner_dt - dt) > 1e-9:
            raise ValueError(
                f"dt={dt} must be a positive multiple of inner_dt={self.inner_dt}"
            )
        return n_inner

    def step(
        self,
        s:         MilliAmpereState,
        psi_cmd:   float,
        speed_cmd: float,
        dt:        float,
    ) -> MilliAmpereState:
        """Heading/speed control (liteAmpere's path-following loop)."""
        n_inner = self._n_inner(dt)

        p    = self.params
        h    = self.inner_dt
        xx   = s.xx.copy()
        ref  = s.ref.copy()
        integ = s.speed_integral

        for _ in range(n_inner):
            # Heading reference filter, commanded to the nearest equivalent angle
            psi_target = ref[0] + _ssa(psi_cmd - ref[0])
            ref = ref + (self._ref_A @ ref + self._ref_B * psi_target) * h
            ref[0] = _ssa(ref[0])

            # PD course control (r reference not fed forward, as in liteAmpere)
            psi_err = _ssa(ref[0] - xx[2])
            delta = self.Kp_psi * psi_err + self.Kd_psi * -xx[5]
            delta = min(max(delta, -p.delta_max), p.delta_max)
            N = p.l_x * p.T_prop * delta

            # PI speed control on surge
            integ += (speed_cmd - xx[3]) * h
            X = p.speed_kp * (speed_cmd - xx[3]) + p.speed_ki * integ

            tau = np.array([X, p.sway_ratio * N, N])
            xx = rk4(xx, lambda x: self.model.derivative(x, tau), h)

        return MilliAmpereState(xx=xx, speed_integral=integ, ref=ref)

    def step_dp(
        self,
        s:       MilliAmpereState,
        eta_ref: np.ndarray,
        eta_dot_ref: np.ndarray,
        dt:      float,
    ) -> MilliAmpereState:
        """Dynamic positioning on a reference moving at constant rate.

        The reference over the step is eta_ref + eta_dot_ref * t (NED position
        and heading, and their rates), for t in [0, dt).

        liteAmpere's DP law is tau = -R^T Kp e - R^T Kd R nu: its damping acts
        on the absolute velocity, which is right for holding station but makes
        a moving reference trail by 10-17 m at 1.5 m/s. Here the damping acts
        on the velocity error instead, and the hull's own damping and Coriolis
        forces at the reference velocity are fed forward:

            tau = -R^T Kp e - R^T Kd (R nu - eta_dot_ref) + N(nu_ref) nu_ref

        with nu_ref = R^T eta_dot_ref. With eta_dot_ref = 0 this is exactly
        liteAmpere's law.

        The speed loop's integrator and the heading reference filter are
        carried through unchanged, and the filter is reset to the hull
        heading, so a later step() picks up from the current heading.
        """
        n_inner = self._n_inner(dt)
        h   = self.inner_dt
        xx  = s.xx.copy()
        eta_dot_ref = np.asarray(eta_dot_ref, dtype=float)

        for i in range(n_inner):
            eta_d = eta_ref + eta_dot_ref * (i * h)
            e = xx[:3] - eta_d
            e[2] = _ssa(e[2])
            Rm = _rot(xx[2])
            nu_ref = Rm.T @ eta_dot_ref
            tau_ff = (self.model.damping(nu_ref) + self.model.coriolis(nu_ref)) @ nu_ref
            tau = (
                -Rm.T @ self._dp_Kp @ e
                - Rm.T @ self._dp_Kd @ (Rm @ xx[3:6] - eta_dot_ref)
                + tau_ff
            )
            xx = rk4(xx, lambda x: self.model.derivative(x, tau), h)

        return MilliAmpereState(
            xx             = xx,
            speed_integral = s.speed_integral,
            ref            = np.array([xx[2], xx[5], 0.0]),
        )
