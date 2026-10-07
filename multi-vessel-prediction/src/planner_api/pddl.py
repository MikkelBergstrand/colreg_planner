"""planner_api/pddl.py

Build a temporal PDDL planning problem for the own ship from the current
prediction, and solve it with a Unified Planning engine (Tamer by default,
or Aries).

The planner decides which COLREGs-compliant manoeuvre the own ship executes
and when. The geometry stays in the predictor: every candidate manoeuvre is
rolled out at several start delays against the considered responsiveness
hypotheses (other vessels reacting by COLREGs), and the outcomes enter the
problem as facts and timed initial literals (TILs).

Belief: a vessel is *trusted* when p_responsive >= p_doubt (and it is not in
assume_non_responsive). The considered hypotheses are those in which every
trusted vessel is responsive and every assumed vessel non-responsive, kept
when their conditional probability is >= p_min.

Facts (all about the own ship o):

    risk(o, t)            straight-line DCPA < d_pass within the horizon
    give_way / stand_on(o, t)  o's COLREGs role toward t
    trusted(t)            t is believed to follow COLREGs
    may_act(o, t)         Rule 17(a)(ii) permission: true now if t is doubted;
                          for an assumed non-responsive t, a TIL makes it true
                          once t should visibly have acted (predicted give-way
                          time + act_grace_s)
    window(m, o, t)       m passes t at >= d_pass if started now; a TIL closes
                          it after the latest verified start delay
    safe(m, o)            m keeps >= d_min to every vessel if started now; a
                          TIL closes it the same way
    close_quarters(o, t)  no manoeuvre started now passes t at d_pass
    best_effort(m, o)     m started now maximizes the worst-case distance
    keeps_course(m), to_port(m)

Actions (COLREGs rules as preconditions):

    execute(o, m)               durative, duration(m); one per plan
    pass_give_way(o, t, m)      Rules 8, 14-16: substantial action, not to port
    pass_stand_on_hold(o, t, m) Rule 17(a)(i): keep course and speed, t trusted
    decide_act(o, t)            Rule 17(a)(ii): t has not acted, decide to act
                                (before the manoeuvre starts)
    pass_stand_on_act(o, t, m)  Rule 17(a)(ii), 17(c): act, not to port
    pass_emergency(o, t, m)     Rule 17(b) / Rule 2: too close for a compliant pass

The compliant pass actions also require safe(m, o); the emergency pass does
not, since a last-moment manoeuvre may no longer keep d_min.

Windows (and safe) are checked when a pass action fires, which is at or after the
manoeuvre start; since TILs only close windows, open then means open at the
start. Starts between the verified delays are assumed to behave like their
neighbours.

Goal: passed_clear(o, t) for every t with risk(o, t).

The facts describe the situation now, so regenerate and re-solve at every
replanning point (new encounter, role change, responsiveness threshold
crossed, plan precondition broken).

Requires the ``planning`` extra: pip install -e ".[planning]"
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import unified_planning.shortcuts as ups
from unified_planning.io import PDDLWriter
from unified_planning.model.timing import GlobalStartTiming, StartTiming

from src.planner_api.manoeuvres import DEFAULT_MANOEUVRES, Manoeuvre
from src.planner_api.service import PredictionService

with warnings.catch_warnings():
    # The environment imports every installed engine, some of which warn on import.
    warnings.simplefilter("ignore", UserWarning)
    ups.get_environment().factory
ups.get_environment().credits_stream = None

_GIVE_WAY_ROLES = {"give_way", "head_on", "overtaking"}

# Per-solve timeout [s] when none is given. Tamer takes no timeout. Aries
# cannot prove a temporal problem unsolvable and runs until the timeout, which
# the ordered solve hits on purpose; it solves these problems in ~0.1 s.
_DEFAULT_TIMEOUT = {"aries": 2.0}


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

def _build_domain() -> tuple[ups.Problem, dict]:
    """Problem with types, fluents and actions but no objects."""
    Vessel = ups.UserType("vessel")
    Mano   = ups.UserType("manoeuvre")
    B      = ups.BoolType()
    Not    = ups.Not

    f = {
        "own":            ups.Fluent("own", B, o=Vessel),
        "started":        ups.Fluent("started", B, o=Vessel),
        "executing":      ups.Fluent("executing", B, m=Mano, o=Vessel),
        "give_way":       ups.Fluent("give_way", B, o=Vessel, t=Vessel),
        "stand_on":       ups.Fluent("stand_on", B, o=Vessel, t=Vessel),
        "risk":           ups.Fluent("risk", B, o=Vessel, t=Vessel),
        "passed_clear":   ups.Fluent("passed_clear", B, o=Vessel, t=Vessel),
        "trusted":        ups.Fluent("trusted", B, t=Vessel),
        "may_act":        ups.Fluent("may_act", B, o=Vessel, t=Vessel),
        "acting_for":     ups.Fluent("acting_for", B, o=Vessel, t=Vessel),
        "window":         ups.Fluent("window", B, m=Mano, o=Vessel, t=Vessel),
        "safe":           ups.Fluent("safe", B, m=Mano, o=Vessel),
        "close_quarters": ups.Fluent("close_quarters", B, o=Vessel, t=Vessel),
        "best_effort":    ups.Fluent("best_effort", B, m=Mano, o=Vessel),
        "keeps_course":   ups.Fluent("keeps_course", B, m=Mano),
        "to_port":        ups.Fluent("to_port", B, m=Mano),
        "allowed":        ups.Fluent("allowed", B, m=Mano),     # see ColregsPlanningProblem.solve
        # Integer seconds: Aries does not support real-valued fluents.
        "duration":       ups.Fluent("duration", ups.IntType(), m=Mano),
    }

    problem = ups.Problem("colregs")
    for name, fl in f.items():
        problem.add_fluent(fl, default_initial_value=0 if name == "duration" else False)

    # execute(o, m): the single course/speed alteration of this plan.
    act = ups.DurativeAction("execute", o=Vessel, m=Mano)
    o, m = act.parameters
    act.set_fixed_duration(f["duration"](m))
    act.add_condition(StartTiming(), f["own"](o))
    act.add_condition(StartTiming(), f["allowed"](m))
    act.add_condition(StartTiming(), Not(f["started"](o)))
    act.add_effect(StartTiming(), f["started"](o), True)
    act.add_effect(StartTiming(), f["executing"](m, o), True)
    problem.add_action(act)

    def _pass(name: str, conds) -> None:
        a = ups.InstantaneousAction(name, o=Vessel, t=Vessel, m=Mano)
        o, t, m = a.parameters
        for c in (f["own"](o), f["risk"](o, t), f["executing"](m, o), *conds(o, t, m)):
            a.add_precondition(c)
        a.add_effect(f["risk"](o, t), False)
        a.add_effect(f["passed_clear"](o, t), True)
        problem.add_action(a)

    # Rules 8, 14-16: give-way vessel takes early, substantial action, not to port.
    _pass("pass_give_way", lambda o, t, m: (
        f["give_way"](o, t), f["window"](m, o, t), f["safe"](m, o),
        Not(f["keeps_course"](m)), Not(f["to_port"](m))))
    # Rule 17(a)(i): stand-on vessel keeps course and speed while t gives way.
    _pass("pass_stand_on_hold", lambda o, t, m: (
        f["stand_on"](o, t), f["trusted"](t), f["window"](m, o, t), f["safe"](m, o),
        f["keeps_course"](m)))

    # Rule 17(a)(ii): once t has not acted, decide to act, before manoeuvring.
    decide = ups.InstantaneousAction("decide_act", o=Vessel, t=Vessel)
    o, t = decide.parameters
    for c in (f["own"](o), f["stand_on"](o, t), f["may_act"](o, t), Not(f["started"](o))):
        decide.add_precondition(c)
    decide.add_effect(f["acting_for"](o, t), True)
    problem.add_action(decide)

    # Rule 17(a)(ii) + 17(c): act, not to port.
    _pass("pass_stand_on_act", lambda o, t, m: (
        f["stand_on"](o, t), f["acting_for"](o, t), f["window"](m, o, t), f["safe"](m, o),
        Not(f["keeps_course"](m)), Not(f["to_port"](m))))
    # Rule 17(b) / Rule 2: no compliant pass is left -> whatever keeps the most distance.
    _pass("pass_emergency", lambda o, t, m: (
        f["close_quarters"](o, t), f["best_effort"](m, o)))

    return problem, {"Vessel": Vessel, "Manoeuvre": Mano, **f}


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------

def _pddl_name(prefix: str, raw: str) -> str:
    return f"{prefix}_{re.sub(r'[^A-Za-z0-9_]', '_', raw)}".lower()


@dataclass
class ColregsPlanningProblem:
    """A generated planning problem plus the evidence behind its facts."""
    problem: ups.Problem
    own_id: str
    manoeuvres: dict[str, Manoeuvre]          # pddl name -> manoeuvre
    vessels: dict[str, str]                   # pddl name -> vessel id
    facts: dict[str, list]                    # human-readable list of asserted facts and TILs
    latest_start: dict[tuple[str, str], float] = field(default_factory=dict)  # (manoeuvre, target) -> [s]
    # "<manoeuvre>@<delay>" -> evaluate_manoeuvre(); delays after the one at
    # which a manoeuvre has failed every fact are not evaluated.
    evaluations: dict[str, dict] = field(default_factory=dict)
    targets_at_risk: list[str] = field(default_factory=list)
    _fluents: dict = field(default_factory=dict, repr=False)

    def write_pddl(self, out_dir: str | Path) -> tuple[Path, Path]:
        """Write domain.pddl and problem.pddl into out_dir."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        writer = PDDLWriter(self.problem)
        domain, prob = out / "domain.pddl", out / "problem.pddl"
        writer.write_domain(str(domain))
        writer.write_problem(str(prob))
        return domain, prob

    def solve(
        self,
        engine: str = "tamer",
        timeout: float | None = None,
        ordered: bool = True,
    ) -> dict:
        """Solve with a Unified Planning engine ("tamer" or "aries").

        timeout : per solve call [s]; None -> 2 s for Aries, none for Tamer
            (which does not support one).

        ordered : the engines do not optimize action costs, so by default the
            problem is solved with only the first manoeuvre allowed, then the
            first two, and so on: the result uses the earliest valid
            manoeuvre in the order given to build_problem(). False -> one
            solve, the planner picks any.

        Returns {"status", "plan": [{start, duration, action, args}],
        "manoeuvre": {...} | None, "start_s", "latest_start_s"}. An empty goal
        (no risk of collision) returns status "no_action_needed" without
        calling the planner.
        """
        none = {"plan": [], "manoeuvre": None, "start_s": None, "latest_start_s": None}
        if not self.targets_at_risk:
            return {"status": "no_action_needed", **none}

        names = list(self.manoeuvres)
        subsets = [names[: i + 1] for i in range(len(names))] if ordered else [names]
        for allowed in subsets:
            problem = self.problem.clone()
            for name in names[len(allowed):]:
                problem.set_initial_value(self._fluents["allowed"](problem.object(name)), False)
            limit = timeout if timeout is not None else _DEFAULT_TIMEOUT.get(engine)
            with ups.OneshotPlanner(name=engine) as planner:
                result = (planner.solve(problem) if limit is None
                          else planner.solve(problem, timeout=limit))
            if result.plan is not None:
                break

        status = result.status.name.lower()
        if result.plan is None:
            return {"status": status, **none}

        steps, chosen, start = [], None, None
        # Simultaneous happenings are listed in causal order: decide, execute, pass.
        causal = {"decide_act": 0, "execute": 1}
        for s, ai, d in sorted(result.plan.timed_actions,
                               key=lambda x: (x[0], causal.get(x[1].action.name, 2))):
            args = [self.vessels.get(str(p), str(p)) for p in ai.actual_parameters]
            steps.append({
                "start": round(float(s), 2),
                "duration": None if d is None else round(float(d), 2),
                "action": ai.action.name,
                "args": args,
            })
            if ai.action.name == "execute":
                chosen, start = str(ai.actual_parameters[1]), round(float(s), 2)
        targets = [st["args"][1] for st in steps
                   if st["action"].startswith("pass_") and st["action"] != "pass_emergency"]
        deadlines = [self.latest_start[(chosen, t)] for t in targets
                     if (chosen, t) in self.latest_start]
        return {
            "status": status,
            "plan": steps,
            "manoeuvre": self.manoeuvres[chosen].to_dict() if chosen else None,
            "start_s": start,
            "latest_start_s": min(deadlines) if deadlines else None,
        }


def _expected_action_time(service: PredictionService, vid: str, threshold_deg: float = 10.0) -> float | None:
    """When vid is predicted to start manoeuvring if it is responsive [s from now]."""
    for h in service.snapshot.hypotheses:
        if h.assumptions.get(vid, "responsive") != "responsive" or vid not in h.trajectories:
            continue
        pts = h.trajectories[vid]
        for p in pts:
            if abs((p.course_deg - pts[0].course_deg + 180.0) % 360.0 - 180.0) >= threshold_deg:
                return p.t
        return None
    return None


def build_problem(
    service: PredictionService,
    own_id: str | None = None,
    *,
    manoeuvres: tuple[Manoeuvre, ...] = DEFAULT_MANOEUVRES,
    d_pass: float = 150.0,
    d_min: float | None = None,
    p_min: float = 0.05,
    p_doubt: float = 0.5,
    delays_s: Iterable[float] = (0, 30, 60, 90, 120),
    assume_non_responsive: Iterable[str] = (),
    act_grace_s: float = 30.0,
) -> ColregsPlanningProblem:
    """Generate the planning problem for the own ship at the current step.

    Parameters
    ----------
    service : PredictionService
        After step() for the current observations. Use own_id on the service
        so the own ship's responsiveness is fixed rather than estimated, and
        top_k >= 2 so alternative hypotheses are rolled out.
    own_id : str, optional
        Defaults to service.own_id.
    manoeuvres : candidate own-ship manoeuvres, in order of preference.
    d_pass : desired passing distance [m]; also the risk-of-collision DCPA threshold.
    d_min : hard minimum distance to any vessel [m]; default d_pass / 2.
    p_min : hypotheses with a lower conditional probability are ignored.
    p_doubt : a vessel with p_responsive below this is doubted; at or above
        it, it is trusted to follow COLREGs.
    delays_s : start delays [s] at which every course/speed alteration is
        evaluated; they define the TIL deadlines.
    assume_non_responsive : vessel ids to plan for as if they will not
        follow COLREGs, e.g. a contingency plan for a trusted give-way
        vessel. A stand-on own ship then gets permission to act (may_act)
        act_grace_s after the vessel was predicted to start giving way.
    act_grace_s : see assume_non_responsive [s].
    """
    own_id = own_id or service.own_id
    if own_id is None:
        raise ValueError("own_id is required (pass it here or to PredictionService)")
    d_min = d_pass / 2 if d_min is None else d_min
    delays = sorted({float(d) for d in delays_s} | {0.0})
    assumed_nr = set(assume_non_responsive)
    snap = service.snapshot

    problem, f = _build_domain()
    facts: dict[str, list] = {}

    def assert_(name: str, *args) -> None:
        problem.set_initial_value(f[name](*args), True)
        facts.setdefault(name, []).append([str(a) for a in args])

    def til(at_s: float, value: bool, name: str, *args) -> None:
        at = max(1, int(math.ceil(at_s)))
        problem.add_timed_effect(GlobalStartTiming(at), f[name](*args), value)
        facts.setdefault("til", []).append([at, name, value, *[str(a) for a in args]])

    # Objects
    vessels = {_pddl_name("v", v.vessel_id): v.vessel_id for v in snap.vessels if v.moving}
    vobj = {vid: ups.Object(name, f["Vessel"]) for name, vid in vessels.items()}
    problem.add_objects(vobj.values())
    mobj = {m.name: ups.Object(_pddl_name("m", m.name), f["Manoeuvre"]) for m in manoeuvres}
    problem.add_objects(mobj.values())
    if own_id not in vobj:
        raise ValueError(f"own ship '{own_id}' is not observed or not moving")
    own = vobj[own_id]
    assert_("own", own)

    # Manoeuvre attributes
    for m in manoeuvres:
        mo = mobj[m.name]
        problem.set_initial_value(f["duration"](mo), int(round(m.duration_s)))
        assert_("allowed", mo)
        if m.keeps_course_and_speed:
            assert_("keeps_course", mo)
        if m.alters_to_port:
            assert_("to_port", mo)

    # Belief and considered hypotheses
    p_resp = service.get_responsiveness()
    trusted = set()
    for vid, o in vobj.items():
        if vid == own_id:
            continue
        p = p_resp.get(vid)
        if vid not in assumed_nr and (p is None or p >= p_doubt):
            trusted.add(vid)
            assert_("trusted", o)
    consistent = [
        h for h in snap.hypotheses
        if all(h.assumptions.get(v, "responsive") == "responsive" for v in trusted)
        and all(h.assumptions.get(v) == "non_responsive" for v in assumed_nr)
    ]
    if not consistent:
        raise ValueError("no rolled-out hypothesis matches the assumptions; "
                         "increase the service's top_k")
    total = sum(h.probability for h in consistent) or 1.0
    considered = {h.rank for h in consistent if h.probability / total >= p_min} or {consistent[0].rank}

    # Roles, risk of collision (straight-line CPA, as in COLREGs Rule 7) and Rule 17 permission
    at_risk = []
    for e in service.get_encounters(own_id):
        t = e["vessel_b"]
        if t not in vobj:
            continue
        if e["role_a"] in _GIVE_WAY_ROLES:
            assert_("give_way", own, vobj[t])
        elif e["role_a"] == "stand_on":
            assert_("stand_on", own, vobj[t])
            if t in assumed_nr:
                t_act = _expected_action_time(service, t)
                if t_act is None:
                    assert_("may_act", own, vobj[t])
                else:
                    til(t_act + act_grace_s, True, "may_act", own, vobj[t])
            elif t not in trusted:
                assert_("may_act", own, vobj[t])
        tcpa = e["tcpa_s"]
        if e["dcpa_m"] < d_pass and tcpa is not None and 0.0 < tcpa <= snap.horizon_s:
            assert_("risk", own, vobj[t])
            at_risk.append(t)

    # Manoeuvre outcomes per start delay: worst case over the considered
    # hypotheses. Delays are evaluated in order, every manoeuvre still open at
    # once (in parallel when the predictor has a worker pool). A manoeuvre's
    # window and safe facts only use its run of passing delays from 0, so once
    # safe and every pass have failed, its later delays are not evaluated.
    evaluations: dict[str, dict] = {}
    latest_start: dict[tuple[str, str], float] = {}
    resolvable: set[str] = set()
    best_now: dict[str, float] = {}
    passes: dict[str, dict[str, list[bool]]] = {m.name: {} for m in manoeuvres}
    safe: dict[str, list[bool]] = {m.name: [] for m in manoeuvres}
    open_ = list(manoeuvres)
    for d in delays:
        # Holding course is the same whenever it "starts".
        batch = [m for m in open_ if d == 0.0 or not m.keeps_course_and_speed]
        if not batch:
            break
        evs = service.evaluate_manoeuvres(
            own_id,
            [dict(course_change_deg=m.course_change_deg, speed_factor=m.speed_factor,
                  duration_s=m.duration_s, start_s=d) for m in batch],
            hypotheses=considered,
        )
        for m, ev in zip(batch, evs):
            evaluations[f"{m.name}@{d:g}"] = ev
            worst = {oid: c["min_distance_m"] for oid, c in ev["worst_case_cpa"].items()}
            safe[m.name].append(min(worst.values(), default=math.inf) >= d_min)
            for oid, dist in worst.items():
                passes[m.name].setdefault(oid, []).append(dist >= d_pass)
            if d == 0.0:
                best_now[m.name] = min(worst.values(), default=math.inf)
        open_ = [m for m in batch
                 if all(safe[m.name]) or any(all(ok) for ok in passes[m.name].values())]

    def last_passing(ok: list[bool]) -> float | None:
        """Latest delay of the run of passing delays from 0 (None: fails now)."""
        last = None
        for d, good in zip(delays, ok):
            if not good:
                break
            last = d
        return last

    for m in manoeuvres:
        mo = mobj[m.name]
        last_safe = last_passing(safe[m.name])
        if last_safe is not None:
            assert_("safe", mo, own)
            if not m.keeps_course_and_speed:
                til(last_safe + 1, False, "safe", mo, own)
        for oid, ok in passes[m.name].items():
            last = last_passing(ok)
            if last is None or oid not in vobj:
                continue
            assert_("window", mo, own, vobj[oid])
            resolvable.add(oid)
            if not m.keeps_course_and_speed:
                latest_start[(mo.name, oid)] = last
                til(last + 1, False, "window", mo, own, vobj[oid])

    assert_("best_effort", mobj[max(best_now, key=best_now.get)], own)
    for t in at_risk:
        if t not in resolvable:
            assert_("close_quarters", own, vobj[t])
        problem.add_goal(f["passed_clear"](own, vobj[t]))

    return ColregsPlanningProblem(
        problem         = problem,
        own_id          = own_id,
        manoeuvres      = {mobj[m.name].name: m for m in manoeuvres},
        vessels         = vessels,
        facts           = facts,
        latest_start    = latest_start,
        evaluations     = evaluations,
        targets_at_risk = at_risk,
        _fluents        = f,
    )
