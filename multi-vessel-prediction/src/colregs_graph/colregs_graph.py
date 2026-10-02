from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field

from src.vessel.encounter import EncounterType
from src.vessel.state import Intent


class VesselRole(Enum):
    GIVE_WAY = "give_way"
    STAND_ON = "stand_on"
    HEAD_ON = "head_on"
    OVERTAKING = "overtaking"
    AVOID_COLLISION = "avoid_collision"


@dataclass
class ColregsEdge:
    vessel_a: int
    vessel_b: int
    role_a: VesselRole
    role_b: VesselRole
    has_non_responsive: bool = False
    # Encounter geometry from vessel_a's perspective (None when unknown).
    encounter_type: EncounterType | None = None


@dataclass
class ColregsGraph:
    edges: list[ColregsEdge] = field(default_factory=list)

    def get_roles(self, vessel_id: int) -> list[tuple[int, VesselRole]]:
        """Return [(other_idx, own_role), ...] for all edges involving vessel_id."""
        roles = []
        for edge in self.edges:
            if edge.vessel_a == vessel_id:
                roles.append((edge.vessel_b, edge.role_a))
            elif edge.vessel_b == vessel_id:
                roles.append((edge.vessel_a, edge.role_b))
        return roles

    def add_edge(self, edge: ColregsEdge) -> None:
        self.edges.append(edge)


# ---------------------------------------------------------------------------
# Encounter → COLREGs role mapping
# ---------------------------------------------------------------------------
# (ego_role, target_role) from ego's perspective (ego = vessel 0).

ENCOUNTER_ROLES: dict[EncounterType, tuple[VesselRole, VesselRole]] = {
    EncounterType.HEAD_ON:            (VesselRole.HEAD_ON,    VesselRole.HEAD_ON),
    EncounterType.CROSSING_STARBOARD: (VesselRole.GIVE_WAY,   VesselRole.STAND_ON),
    EncounterType.CROSSING_PORT:      (VesselRole.STAND_ON,   VesselRole.GIVE_WAY),
    EncounterType.OVERTAKING_AHEAD:   (VesselRole.OVERTAKING, VesselRole.STAND_ON),
    EncounterType.OVERTAKING_ASTERN:  (VesselRole.STAND_ON,   VesselRole.OVERTAKING),
}


# Role a responsive vessel adopts when its counterpart is non-responsive.
# The non-responsive vessel is effectively frozen — the responsive vessel must act.
#   HEAD_ON    → GIVE_WAY        (still turn starboard, assertively)
#   STAND_ON   → AVOID_COLLISION (other not giving way — free to manoeuvre)
#   GIVE_WAY   → GIVE_WAY        (already acting, no change)
#   OVERTAKING → AVOID_COLLISION (other not clearing — free to manoeuvre)
_NON_RESPONSIVE_FACING: dict[VesselRole, VesselRole] = {
    VesselRole.HEAD_ON:         VesselRole.GIVE_WAY,
    VesselRole.STAND_ON:        VesselRole.AVOID_COLLISION,
    VesselRole.GIVE_WAY:        VesselRole.GIVE_WAY,
    VesselRole.OVERTAKING:      VesselRole.AVOID_COLLISION,
    VesselRole.AVOID_COLLISION: VesselRole.AVOID_COLLISION,
}


def apply_responsiveness(
    graph: ColregsGraph,
    intents: dict[int, Intent],
) -> ColregsGraph:
    """
    Post-process a classifier-produced ColregsGraph to account for
    non-responsive vessels.

    Parameters
    ----------
    graph   : ColregsGraph from ColregsClassifier.update()
    intents : dict mapping vessel_id → Intent for all tracked vessels

    Returns a new ColregsGraph with adjusted roles:
      - Non-responsive vessel → STAND_ON  (it will not manoeuvre)
      - Responsive vessel facing non-responsive → _NON_RESPONSIVE_FACING[nominal_role]
    """
    new_graph = ColregsGraph()

    for edge in graph.edges:
        a_nr = intents.get(edge.vessel_a) == Intent.NON_RESPONSIVE
        b_nr = intents.get(edge.vessel_b) == Intent.NON_RESPONSIVE

        if not (a_nr or b_nr):
            new_graph.add_edge(edge)
            continue

        role_a = VesselRole.STAND_ON if a_nr else _NON_RESPONSIVE_FACING[edge.role_a]
        role_b = VesselRole.STAND_ON if b_nr else _NON_RESPONSIVE_FACING[edge.role_b]

        new_graph.add_edge(ColregsEdge(
            vessel_a=edge.vessel_a,
            vessel_b=edge.vessel_b,
            role_a=role_a,
            role_b=role_b,
            has_non_responsive=True,
            encounter_type=edge.encounter_type,
        ))

    return new_graph