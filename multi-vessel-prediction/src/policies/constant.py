"""policies/constant.py

Concrete vessel policies built on BasePolicy.

Action type guide
-----------------
CVAction policies  →  pair with CVAdapter or any *_ar adapter
CTAction policies  →  pair with ct_coupled, ct_decoupled, or viknes adapters

    ConstantVelocityPolicy   CVAction   hold current (vn, ve) — true constant velocity
    ConstantTurnPolicy       CTAction   hold current speed, fixed turn rate (0 = straight)
    FixedCommandPolicy       CTAction   fixed (speed, turn_rate) every step
"""

from __future__ import annotations

from src.vessel.state import VesselState, CTAction, CVAction
from src.vessel.base import BasePolicy


class ConstantVelocityPolicy(BasePolicy):
    """Hold the current velocity vector (vn, ve) unchanged each step.

    Returns CVAction — pair with CVAdapter or an *_ar adapter.
    """

    def select_action(
        self,
        own_state:     VesselState,
        _other_states: list[VesselState],
    ) -> CVAction:
        return CVAction(vn=own_state.vn, ve=own_state.ve)


class ConstantTurnPolicy(BasePolicy):
    """Hold current speed with a fixed turn rate (use 0.0 for straight-line).

    Returns CTAction — pair with ct_coupled, ct_decoupled, or viknes adapters.

    Parameters
    ----------
    turn_rate : float  [rad/s], positive = clockwise (NED)
    """

    def __init__(self, turn_rate: float) -> None:
        self.turn_rate = turn_rate

    def select_action(
        self,
        own_state:     VesselState,
        _other_states: list[VesselState],
    ) -> CTAction:
        return CTAction(speed=own_state.speed, turn_rate=self.turn_rate)


class FixedCommandPolicy(BasePolicy):
    """Output a fixed (speed, turn_rate) every step.

    Returns CTAction — pair with ct_coupled, ct_decoupled, or viknes adapters.
    """

    def __init__(self, speed: float, turn_rate: float = 0.0) -> None:
        self._action = CTAction(speed=speed, turn_rate=turn_rate)

    def select_action(
        self,
        _own_state:    VesselState,
        _other_states: list[VesselState],
    ) -> CTAction:
        return self._action
