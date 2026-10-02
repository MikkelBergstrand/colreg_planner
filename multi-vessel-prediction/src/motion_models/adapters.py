"""models/adapters.py

Adapters that wrap the CT, Viknes, milliAmpere, and CV models behind the BaseSimModel interface.
Each adapter translates an Action to model-specific inputs and converts model-specific
state back to the standard VesselState.
"""

from __future__ import annotations

import numpy as np

from src.vessel.state import InitialState, VesselState, CTAction, CVAction, DPAction, wrap_angle
from src.vessel.base import BaseSimModel
from src.motion_models.ct_model import (
    CTCoupledState, CTCoupledModel,
    CTDecoupledState, CTDecoupledModel,
)
from src.motion_models.vikenes_model import ViknesState, ViknesSimulator, ViknesPDController
from src.motion_models.milliampere_model import MilliAmpereState, MilliAmpereSimulator


# ---------------------------------------------------------------------------
# CT-Coupled adapter
# ---------------------------------------------------------------------------

class CTCoupledAdapter(BaseSimModel):
    """Wraps CTCoupledModel. Accepts CTAction(speed, turn_rate)."""

    def __init__(self) -> None:
        self._s: CTCoupledState | None = None

    def init(self, initial: InitialState) -> VesselState:
        self._s = CTCoupledState.from_pos_speed_heading(
            initial.north, initial.east, initial.speed, initial.heading
        )
        return self._export(r=0.0)

    def step(self, action: CTAction, dt: float) -> VesselState:
        assert isinstance(action, CTAction), (
            f"CTCoupledAdapter requires CTAction, got {type(action).__name__}"
        )
        assert self._s is not None, "Call init() before step()."
        psi = self._s.heading
        self._s = CTCoupledState(
            north=self._s.north, east=self._s.east,
            vn=action.speed * np.cos(psi),
            ve=action.speed * np.sin(psi),
        )
        self._s = CTCoupledModel.step(self._s, action.turn_rate, dt)
        return self._export(r=action.turn_rate)

    def _export(self, r: float) -> VesselState:
        return VesselState(
            north=self._s.north, east=self._s.east,
            psi=self._s.heading,
            speed=self._s.speed,
            r=r,
        )


# ---------------------------------------------------------------------------
# CT-Decoupled adapter
# ---------------------------------------------------------------------------

class CTDecoupledAdapter(BaseSimModel):
    """Wraps CTDecoupledModel. Accepts CTAction(speed, turn_rate)."""

    def __init__(self) -> None:
        self._s: CTDecoupledState | None = None

    def init(self, initial: InitialState) -> VesselState:
        self._s = CTDecoupledState.from_pos_speed_heading(
            initial.north, initial.east, initial.speed, initial.heading
        )
        return self._export()

    def step(self, action: CTAction, dt: float) -> VesselState:
        assert isinstance(action, CTAction), (
            f"CTDecoupledAdapter requires CTAction, got {type(action).__name__}"
        )
        assert self._s is not None, "Call init() before step()."
        self._s = CTDecoupledState(
            north=self._s.north, east=self._s.east,
            psi=self._s.psi,
            v=action.speed,
            r=action.turn_rate,
        )
        self._s = CTDecoupledModel.step(self._s, dt)
        return self._export()

    def _export(self) -> VesselState:
        return VesselState(
            north=self._s.north, east=self._s.east,
            psi=self._s.heading,
            speed=self._s.speed,
            r=self._s.r,
        )


# ---------------------------------------------------------------------------
# Viknes adapter
# ---------------------------------------------------------------------------

class ViknesAdapter(BaseSimModel):
    """Wraps ViknesSimulator + ViknesPDController. Accepts CTAction or CVAction.

    - CTAction(speed, turn_rate): desired_heading = psi + turn_rate * dt
    - CVAction(vn, ve): desired_heading extracted directly from velocity vector

    Note: the Viknes simulator uses a fixed internal dt set at construction.
    The dt passed to step() is ignored; ensure they match at the call site.
    """

    def __init__(
        self,
        dt:     float = 0.1,
        Kp_psi: float = 5_000.0,
        Kd_psi: float = 2_000.0,
        Kp_u:   float = 1_000.0,
    ) -> None:
        self._sim = ViknesSimulator(dt=dt)
        self._ctl = ViknesPDController(Kp_psi=Kp_psi, Kd_psi=Kd_psi, Kp_u=Kp_u)
        self._s: ViknesState | None = None

    def init(self, initial: InitialState) -> VesselState:
        self._s = ViknesState.from_ned_speed_heading(
            x=initial.north, y=initial.east,
            speed=initial.speed, heading=initial.heading,
        )
        return self._export()

    def step(self, action: CTAction | CVAction, dt: float) -> VesselState:
        assert isinstance(action, (CTAction, CVAction)), (
            f"ViknesAdapter requires CTAction or CVAction, got {type(action).__name__}"
        )
        assert self._s is not None, "Call init() before step()."
        if isinstance(action, CVAction):
            desired_heading = action.heading if action.speed > 1e-9 else self._s.psi
        else:
            desired_heading = wrap_angle(self._s.psi + action.turn_rate * self._sim.dt)
        tau = self._ctl.compute(self._s, desired_heading, action.speed)
        self._s = self._sim.step(self._s, tau)
        return self._export()

    def _export(self) -> VesselState:
        return VesselState(
            north=self._s.x, east=self._s.y,
            psi=self._s.psi,
            speed=self._s.speed,
            r=self._s.r,
        )


# ---------------------------------------------------------------------------
# milliAmpere adapter
# ---------------------------------------------------------------------------

class MilliAmpereAdapter(BaseSimModel):
    """Wraps MilliAmpereSimulator (3DOF kinetics + heading/speed loops + DP).
    Accepts CVAction, CTAction or DPAction.

    - CVAction(vn, ve): commanded heading and speed taken from the velocity vector
    - CTAction(speed, turn_rate): the commanded heading advances by turn_rate * dt
      each step, starting from the vessel's heading
    - DPAction(d_north, d_east, psi_ref): dynamic positioning on a reference that
      moves by (d_north, d_east) over the step and turns to psi_ref. The
      reference starts at the initial pose and accumulates the steps, so it
      moves along the commanded path independently of where the hull is,
      unless the action sets ref_start, which places it explicitly.

    The simulator integrates internally at inner_dt; the step dt must be a
    whole multiple of it.
    """

    _MIN_HEADWAY = 0.05   # [m/s] surge below which the course is not reported

    def __init__(self, inner_dt: float = 0.01) -> None:
        self._sim = MilliAmpereSimulator(inner_dt=inner_dt)
        self._s: MilliAmpereState | None = None
        self._psi_cmd: float = 0.0
        self._dp_ref: np.ndarray | None = None   # [north, east, psi] DP reference

    def init(self, initial: InitialState) -> VesselState:
        self._s = self._sim.reset(
            initial.north, initial.east, initial.heading, initial.speed,
        )
        self._psi_cmd = initial.heading
        self._dp_ref = np.array([initial.north, initial.east, initial.heading])
        return self._export()

    @property
    def dp_reference(self) -> np.ndarray | None:
        """Current DP reference [north, east, psi], or None before init()."""
        return None if self._dp_ref is None else self._dp_ref.copy()

    def step(self, action: CTAction | CVAction | DPAction, dt: float) -> VesselState:
        assert isinstance(action, (CTAction, CVAction, DPAction)), (
            "MilliAmpereAdapter requires CTAction, CVAction or DPAction, "
            f"got {type(action).__name__}"
        )
        assert self._s is not None, "Call init() before step()."
        if isinstance(action, DPAction):
            ref = (self._dp_ref if action.ref_start is None
                   else np.array(action.ref_start, dtype=float))
            rate = np.array([
                action.d_north / dt,
                action.d_east / dt,
                wrap_angle(action.psi_ref - ref[2]) / dt,
            ])
            self._s = self._sim.step_dp(self._s, ref, rate, dt)
            self._dp_ref = np.array([
                ref[0] + action.d_north,
                ref[1] + action.d_east,
                wrap_angle(action.psi_ref),
            ])
            self._psi_cmd = float(self._s.xx[2])
            return self._export()
        # Heading/speed control: keep the DP reference on the hull so a later
        # switch to DP starts from where the vessel is.
        if isinstance(action, CVAction):
            if action.speed > 1e-9:
                self._psi_cmd = action.heading
        else:
            self._psi_cmd = wrap_angle(self._psi_cmd + action.turn_rate * dt)
        self._s = self._sim.step(self._s, self._psi_cmd, action.speed, dt)
        self._dp_ref = self._s.xx[:3].copy()
        return self._export()

    def _export(self) -> VesselState:
        # VesselState carries a single angle that consumers turn into the
        # velocity vector (vn = speed * cos(psi)), and a non-negative speed.
        # Making way ahead, export course over ground rather than hull
        # heading: the hull sideslips by tens of degrees in a turn and keeps a
        # few degrees after it. Stopped or going astern (u <= _MIN_HEADWAY,
        # e.g. DP backing onto a reference it overshot), the course points
        # off the bow or astern and would read as the ship having turned
        # round; planners started from it then plan a U-turn. There, export
        # the hull heading and only the headway.
        north, east, psi, u, v, r = self._s.xx
        if u > self._MIN_HEADWAY:
            speed  = float(np.hypot(u, v))
            course = psi + np.arctan2(v, u)
        else:
            speed  = max(0.0, float(u))
            course = psi
        return VesselState(
            north=float(north), east=float(east),
            psi=wrap_angle(course),
            speed=speed,
            r=float(r),
        )


# ---------------------------------------------------------------------------
# CV (constant-velocity) adapter
# ---------------------------------------------------------------------------

class CVAdapter(BaseSimModel):
    """Constant-velocity kinematic model. Accepts CVAction(vn, ve).

    Integrates the commanded velocity directly — no turn dynamics.
    Pair with VOPolicyAnalytic or any policy that returns CVAction.
    """

    def __init__(self) -> None:
        self._s: VesselState | None = None

    def init(self, initial: InitialState) -> VesselState:
        self._s = VesselState(
            north=initial.north, east=initial.east,
            psi=initial.heading, speed=initial.speed,
            r=0.0,
        )
        return self._s

    def step(self, action: CVAction, dt: float) -> VesselState:
        assert isinstance(action, CVAction), (
            f"CVAdapter requires CVAction, got {type(action).__name__}"
        )
        assert self._s is not None, "Call init() before step()."
        north = self._s.north + action.vn * dt
        east  = self._s.east  + action.ve * dt
        # Preserve heading when stopped: arctan2(0,0)=0 would corrupt psi otherwise
        psi = action.heading if action.speed > 1e-9 else self._s.psi
        self._s = VesselState(
            north=north, east=east,
            psi=psi,
            speed=action.speed,
            r=0.0,
        )
        return self._s


# ---------------------------------------------------------------------------
# Angular-rate wrapper
# ---------------------------------------------------------------------------

class AngularRateWrapper(BaseSimModel):
    """Bridges CVAction → CTAction via a proportional heading controller.

    Wraps any CT adapter (CTCoupledAdapter, CTDecoupledAdapter, ViknesAdapter),
    allowing CVAction-producing policies (e.g. VOPolicyAnalytic) to drive models with
    real turn dynamics.

    The wrapper converts the desired velocity vector (vn, ve) to a desired heading,
    computes turn_rate = k_psi * wrap(psi_des - psi_current), and forwards a
    CTAction to the inner adapter.

    Parameters
    ----------
    inner : BaseSimModel
        A CT adapter instance to wrap.
    k_psi : float
        Proportional gain [1/s].  Higher values track heading faster but may
        overshoot in models with turn-rate lag (e.g. Viknes).
    """

    def __init__(self, inner: BaseSimModel, k_psi: float = 0.1) -> None:
        self._inner = inner
        self.k_psi  = k_psi
        self._psi:  float = 0.0

    def init(self, initial: InitialState) -> VesselState:
        state      = self._inner.init(initial)
        self._psi  = state.psi
        return state

    def step(self, action: CVAction, dt: float) -> VesselState:
        assert isinstance(action, CVAction), (
            f"AngularRateWrapper requires CVAction, got {type(action).__name__}"
        )
        # When stopped, preserve current heading instead of snapping to arctan2(0,0)=0
        psi_des   = action.heading if action.speed > 1e-9 else self._psi
        turn_rate = self.k_psi * wrap_angle(psi_des - self._psi)
        ct_action = CTAction(speed=action.speed, turn_rate=turn_rate)
        state     = self._inner.step(ct_action, dt)
        self._psi = state.psi
        return state


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------

def build_adapter(model_type: str, dt: float = 1.0, *, k_psi: float) -> BaseSimModel:
    """Instantiate a BaseSimModel by name.

    model_type must be one of the strings accepted by run_simulation's sim_model
    field.  VO policies produce CVAction, so only 'cv' and the '*_ar' variants
    are compatible; plain CT/Viknes adapters expect CTAction.
    """
    if model_type == "cv":
        return CVAdapter()
    if model_type == "ct_coupled":
        return CTCoupledAdapter()
    if model_type == "ct_decoupled":
        return CTDecoupledAdapter()
    if model_type == "viknes":
        return ViknesAdapter(dt=dt)
    if model_type == "ct_coupled_ar":
        return AngularRateWrapper(CTCoupledAdapter(), k_psi=k_psi)
    if model_type == "ct_decoupled_ar":
        return AngularRateWrapper(CTDecoupledAdapter(), k_psi=k_psi)
    if model_type == "viknes_ar":
        return AngularRateWrapper(ViknesAdapter(dt=dt), k_psi=k_psi)
    if model_type == "milliampere":
        return MilliAmpereAdapter()
    raise ValueError(
        f"Unknown model_type '{model_type}'. "
        "Choose: cv | ct_coupled | ct_decoupled | viknes | "
        "ct_coupled_ar | ct_decoupled_ar | viknes_ar | milliampere"
    )
