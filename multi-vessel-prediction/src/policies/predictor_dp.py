"""policies/predictor_dp.py

Own-ship policy that follows the interaction-aware predictor's own forecast
for the vessel, through dynamic positioning.

Each step the simulation hands the policy the vessel's predicted state now
and one step ahead (top-1 responsiveness hypothesis, see
scripts/run_simulation.py). The policy returns that predicted step as a
DPAction; the sim model advances its DP reference by it, so the reference
moves along the predicted path at the predicted velocity.

Returns DPAction: pair with sim_model 'milliampere'.
"""

from __future__ import annotations

from typing import Optional

from src.vessel.base import BasePolicy
from src.vessel.state import DPAction, VesselState


class PredictorDPPolicy(BasePolicy):
    """Turn the predictor's next step for this vessel into a DP reference step."""

    def __init__(self, dt: float) -> None:
        self.dt = dt
        self._pred: Optional[tuple[VesselState, VesselState]] = None

    def set_predicted_step(self, now: VesselState, nxt: VesselState) -> None:
        """Predicted state at the current time and one dt later."""
        self._pred = (now, nxt)

    def select_action(
        self,
        own_state:     VesselState,
        _other_states: list[VesselState],
    ) -> DPAction:
        pred, self._pred = self._pred, None
        if pred is None:
            # No prediction this step: carry on at the current velocity.
            return DPAction(
                d_north=own_state.vn * self.dt,
                d_east=own_state.ve * self.dt,
                psi_ref=own_state.psi,
            )
        now, nxt = pred
        return DPAction(
            d_north=nxt.north - now.north,
            d_east=nxt.east - now.east,
            psi_ref=nxt.psi,
        )
