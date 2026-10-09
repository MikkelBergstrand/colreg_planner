"""planner_api/tools.py

Function-calling interface for PredictionService.

TOOLS is a list of tool definitions ({"name", "description", "input_schema"}),
the format the Anthropic Messages API takes in ``tools=``; other frameworks
use the same JSON Schema under a different key (e.g. "parameters"), see
tool_specs(). call_tool() dispatches a tool call by name to the service and
always returns a JSON-serializable dict, with errors reported as
{"error": "..."} instead of raised, so a planner can recover from a bad call.

    from colregs_planner.planner_api import PredictionService, TOOLS, call_tool

    service.step(observations)
    result = call_tool(service, "get_encounters", {"vessel_id": "ego"})
"""

from __future__ import annotations

from typing import Any

from colregs_planner.planner_api.service import PredictionService


_VESSEL_ID = {"type": "string", "description": "Vessel id as given in the observations."}
_HYPOTHESIS = {
    "type": "integer", "minimum": 0, "default": 0,
    "description": "Rank of the responsiveness hypothesis; 0 = most likely.",
}
_RESOLUTION = {
    "type": "number", "exclusiveMinimum": 0,
    "description": "Spacing between returned trajectory points [s]. Omit for the full resolution.",
}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_snapshot",
        "description": (
            "Full situation picture at the current time: every vessel's state, "
            "probability of being COLREGs-responsive and roles, all active COLREGs "
            "encounters with range/bearing/DCPA/TCPA, and the responsiveness "
            "hypotheses. Positions are NED metres from the session origin, angles "
            "in degrees (0 = north, clockwise), speeds in m/s."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "include_trajectories": {"type": "boolean", "default": False,
                                         "description": "Include predicted trajectories per hypothesis."},
                "include_nominal": {"type": "boolean", "default": False,
                                    "description": "Include traffic-free nominal trajectories."},
                "resolution_s": _RESOLUTION,
            },
        },
    },
    {
        "name": "list_vessels",
        "description": "Current state, p_responsive and COLREGs roles of every observed vessel.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_vessel",
        "description": "Current state, p_responsive and COLREGs roles of one vessel.",
        "input_schema": {
            "type": "object",
            "properties": {"vessel_id": _VESSEL_ID},
            "required": ["vessel_id"],
        },
    },
    {
        "name": "get_responsiveness",
        "description": (
            "Probability that each vessel is responsive, i.e. manoeuvres according to "
            "its COLREGs obligations. Near 0 means the vessel is holding its course "
            "regardless of traffic. Null when the responding filter is disabled."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"vessel_id": {**_VESSEL_ID, "description": "Only this vessel; omit for all."}},
        },
    },
    {
        "name": "get_encounters",
        "description": (
            "Active COLREGs encounters (head_on, crossing_starboard, crossing_port, "
            "overtaking_ahead, overtaking_astern) with each vessel's role "
            "(give_way, stand_on, head_on, overtaking, avoid_collision), range, "
            "bearing and straight-line DCPA/TCPA. With vessel_id, only encounters "
            "involving that vessel, seen from its perspective (it is vessel_a)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"vessel_id": {**_VESSEL_ID, "description": "Perspective vessel; omit for all encounters."}},
        },
    },
    {
        "name": "get_hypotheses",
        "description": (
            "Responsiveness hypotheses, most likely first: which vessels are assumed "
            "responsive or non-responsive, the joint probability, and the COLREGs roles "
            "after re-assigning obligations around non-responsive vessels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "include_trajectories": {"type": "boolean", "default": False},
                "resolution_s": _RESOLUTION,
            },
        },
    },
    {
        "name": "get_predicted_trajectory",
        "description": (
            "Interaction-aware predicted trajectory of one vessel under one hypothesis: "
            "a list of {t, north, east, course_deg, speed} with t in seconds from now."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vessel_id": _VESSEL_ID,
                "hypothesis": _HYPOTHESIS,
                "resolution_s": _RESOLUTION,
                "horizon_s": {"type": "number", "exclusiveMinimum": 0,
                              "description": "Cut the trajectory at this time [s]."},
            },
            "required": ["vessel_id"],
        },
    },
    {
        "name": "get_predicted_state",
        "description": "Predicted state of one vessel t seconds from now under one hypothesis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vessel_id": _VESSEL_ID,
                "t": {"type": "number", "minimum": 0, "description": "Seconds from now."},
                "hypothesis": _HYPOTHESIS,
            },
            "required": ["vessel_id", "t"],
        },
    },
    {
        "name": "get_nominal_trajectory",
        "description": (
            "Where a vessel would go if there were no other traffic (nominal layer, "
            "from historical tracks when a library is loaded, else its mission course)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"vessel_id": _VESSEL_ID, "resolution_s": _RESOLUTION},
            "required": ["vessel_id"],
        },
    },
    {
        "name": "get_predicted_cpa",
        "description": (
            "Closest point of approach between a vessel and every other vessel (or one) "
            "along the predicted trajectories of one hypothesis, including the predicted "
            "COLREGs manoeuvres. Sorted by minimum distance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vessel_id": _VESSEL_ID,
                "other_id": {**_VESSEL_ID, "description": "Only this counterpart; omit for all."},
                "hypothesis": _HYPOTHESIS,
            },
            "required": ["vessel_id"],
        },
    },
    {
        "name": "evaluate_manoeuvre",
        "description": (
            "Predict the outcome if a vessel starts a manoeuvre start_s seconds from now: "
            "alter course by course_change_deg (positive = starboard) and scale speed by speed_factor "
            "for duration_s, then return to the nominal course. The other vessels react "
            "according to their COLREGs roles. Returns the closest approach to every "
            "other vessel under each hypothesis and the worst case."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "vessel_id": _VESSEL_ID,
                "course_change_deg": {"type": "number", "minimum": -180, "maximum": 180, "default": 0},
                "speed_factor": {"type": "number", "minimum": 0, "default": 1},
                "duration_s": {"type": "number", "exclusiveMinimum": 0, "default": 180},
                "start_s": {"type": "number", "minimum": 0, "default": 0,
                            "description": "Delay before the manoeuvre starts [s]."},
            },
            "required": ["vessel_id"],
        },
    },
]

_TOOL_NAMES = {t["name"] for t in TOOLS}


def tool_specs(schema_key: str = "input_schema") -> list[dict[str, Any]]:
    """TOOLS with the JSON Schema under *schema_key* (e.g. "parameters")."""
    return [
        {"name": t["name"], "description": t["description"], schema_key: t["input_schema"]}
        for t in TOOLS
    ]


def call_tool(service: PredictionService, name: str, arguments: dict[str, Any] | None = None) -> dict:
    """Run tool *name* with *arguments* and wrap the result as {"result": ...}.

    Errors (unknown tool, unknown vessel, bad arguments, no data yet) come
    back as {"error": "..."}.
    """
    if name not in _TOOL_NAMES:
        return {"error": f"unknown tool '{name}'; available: {sorted(_TOOL_NAMES)}"}
    try:
        return {"result": getattr(service, name)(**(arguments or {}))}
    except (KeyError, IndexError, ValueError, TypeError, RuntimeError) as exc:
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return {"error": f"{type(exc).__name__}: {msg}"}
