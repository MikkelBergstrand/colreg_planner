"""vessel/base.py

Abstract interfaces for simulation models and policies, and the Vessel class
that combines them into a single step-able entity.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from colregs_planner.vessel.state import InitialState, VesselState, Action


class BaseSimModel(ABC):
    """Stateful simulation model adapter.

    Implementations maintain their own internal representation.
    The public interface always speaks VesselState / Action.
    """

    @abstractmethod
    def init(self, initial: InitialState) -> VesselState:
        """Initialise from a scenario InitialState; return the standard state."""
        ...

    @abstractmethod
    def step(self, action: Action, dt: float) -> VesselState:
        """Advance internal state one timestep; return new standard state."""
        ...


class BasePolicy(ABC):
    """Abstract vessel policy."""

    @abstractmethod
    def select_action(
        self,
        own_state:    VesselState,
        other_states: list[VesselState],
    ) -> Action:
        """Select an action given own state and observed other vessel states."""
        ...


@dataclass
class Vessel:
    """A vessel that couples a simulation model with a policy.

    Attributes
    ----------
    vessel_id : str
    state     : current standard VesselState
    sim_model : BaseSimModel  — dynamics adapter
    policy    : BasePolicy    — decision-making agent
    """

    vessel_id: str
    state:     VesselState
    sim_model: BaseSimModel
    policy:    BasePolicy

    @classmethod
    def create(
        cls,
        vessel_id: str,
        initial:   InitialState,
        sim_model: BaseSimModel,
        policy:    BasePolicy,
    ) -> "Vessel":
        """Construct a Vessel, initialising the model from an InitialState."""
        state = sim_model.init(initial)
        return cls(vessel_id=vessel_id, state=state,
                   sim_model=sim_model, policy=policy)

    def step(self, other_states: list[VesselState], dt: float) -> None:
        """Advance the vessel one timestep in-place."""
        t0 = time.perf_counter()
        action = self.policy.select_action(self.state, other_states)
        elapsed = time.perf_counter() - t0
        if not hasattr(self.policy, "_call_times"):
            self.policy._call_times = []
        self.policy._call_times.append(elapsed)
        self.state = self.sim_model.step(action, dt)
