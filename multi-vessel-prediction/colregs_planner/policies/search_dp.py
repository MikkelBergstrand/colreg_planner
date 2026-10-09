"""policies/search_dp.py

Own-ship policy that plans with ManoeuvreSearchPlanner and follows the chosen
plan through dynamic positioning.

Every replan_s the simulation hands the policy the current situation; the
planner picks the best manoeuvre sequence, and the own ship's track along it
(rolled out with the prediction model) becomes the path the DP reference
steps along until the next replan. As with PredictorDPPolicy, each step is
returned as a DPAction: pair with sim_model 'milliampere'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from colregs_planner.planner_api.manoeuvre_search import (
    ManoeuvreSearchPlanner, SearchResult, interpolate_track,
)
from colregs_planner.vessel.base import BasePolicy
from colregs_planner.vessel.state import DPAction, VesselState


@dataclass
class PlanRecord:
    """One replan, kept for plotting and analysis."""

    t:            float              # sim time the plan was made [s]
    result:       SearchResult
    velocity_seq: np.ndarray         # (H, 2) planned [vn, ve] at result.plan_dt


class SearchDPPolicy(BasePolicy):
    """Follow the planner's chosen own-ship track, replanning every replan_s."""

    def __init__(self, dt: float, planner: ManoeuvreSearchPlanner, replan_s: float = 5.0) -> None:
        self.dt       = dt
        self.planner  = planner
        self.replan_s = replan_s
        self.time     = 0.0                       # sim time of the next select_action
        self.plans:   list[PlanRecord] = []

    @property
    def current(self) -> Optional[PlanRecord]:
        return self.plans[-1] if self.plans else None

    def needs_replan(self) -> bool:
        cur = self.current
        return cur is None or self.time - cur.t >= self.replan_s - 1e-9

    def set_plan(self, result: SearchResult) -> None:
        self.plans.append(PlanRecord(
            t=self.time, result=result,
            velocity_seq=self.planner.velocity_sequence(result.best.actions),
        ))

    def select_action(
        self,
        own_state:     VesselState,
        _other_states: list[VesselState],
    ) -> DPAction:
        cur, t = self.current, self.time
        self.time += self.dt
        if cur is None:
            return DPAction(d_north=own_state.vn * self.dt, d_east=own_state.ve * self.dt,
                            psi_ref=own_state.psi)
        traj, pdt = cur.result.own_traj, cur.result.plan_dt
        tau = t - cur.t
        if tau + self.dt > (len(traj) - 1) * pdt:
            # Past the planned horizon (only if replanning stopped): keep going.
            last = traj[-1]
            return DPAction(d_north=last.vn * self.dt, d_east=last.ve * self.dt, psi_ref=last.psi)
        # The reference tracks the plan itself, so each replan re-anchors it
        # at the hull (where the new plan starts) instead of leaving it on the
        # old plan, e.g. at a stop point the hull overshot.
        n0, e0, psi0 = interpolate_track(traj, pdt, tau)
        n1, e1, psi1 = interpolate_track(traj, pdt, tau + self.dt)
        return DPAction(d_north=n1 - n0, d_east=e1 - e0, psi_ref=psi1,
                        ref_start=(n0, e0, psi0))

    def planned_velocities(self, t: float, dt: float, n: int) -> Optional[np.ndarray]:
        """(n, 2) planned [vn, ve] from sim time t at spacing dt, from the plan
        in force at t (nominal velocity beyond its horizon). None before the
        first plan."""
        rec = next((p for p in reversed(self.plans) if p.t <= t + 1e-9), None)
        if rec is None:
            return None
        seq, pdt = rec.velocity_seq, rec.result.plan_dt
        idx = np.floor((t - rec.t + np.arange(n) * dt) / pdt + 1e-9).astype(int)
        return seq[np.clip(idx, 0, len(seq) - 1)]
