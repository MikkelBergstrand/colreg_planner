"""Planner-facing interface to the multi-vessel predictor.

PredictionService runs the predictor once per observation step and exposes
the results through cheap get_* queries returning JSON-serializable data.
TOOLS / call_tool expose the same queries for function-calling planners.
ColregsDPPlanner turns observed vessel states into a COLREGs-aware DP
reference for the own ship.
"""

from src.planner_api.schema import (
    Encounter,
    Hypothesis,
    PredictedCPA,
    RoleAssignment,
    Snapshot,
    TrajectoryPoint,
    VesselStatus,
)
from src.planner_api.dp_planner import ColregsDPPlanner, DPReference, Route
from src.planner_api.service import PredictionService, build_predictor, to_vessel_state
from src.planner_api.tools import TOOLS, call_tool, tool_specs

__all__ = [
    "ColregsDPPlanner",
    "DPReference",
    "Route",
    "PredictionService",
    "build_predictor",
    "to_vessel_state",
    "TOOLS",
    "call_tool",
    "tool_specs",
    "Snapshot",
    "VesselStatus",
    "RoleAssignment",
    "Encounter",
    "Hypothesis",
    "TrajectoryPoint",
    "PredictedCPA",
]
