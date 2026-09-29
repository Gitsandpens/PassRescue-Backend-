"""Discrete contact scheduling with SciPy/HiGHS mixed-integer optimisation.

Binary variables select complete contact candidates. Resource constraints apply to
exact half-open reservation intervals, not merely their rounded display positions.
The search grid is explicit; OPTIMAL means optimal only on this candidate grid.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import time
from typing import Iterable
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csc_matrix, vstack
from .models import Assignment, Scenario

PRIORITY_ORDER = {"critical": 0, "high": 1, "routine": 2}
SECONDARY_VALUE = {"critical": 0, "high": 3, "routine": 1}
MAX_CANDIDATES = 12000

@dataclass(frozen=True)
class Candidate:
    request_id: str
    satellite_id: str
    station_id: str
    window_id: str
    reserved_start: int
    start: int
    end: int
    reserved_end: int
    preserved: bool = False


def seconds(scenario: Scenario, dt: datetime) -> float:
    return (dt - scenario.start).total_seconds()


def same_contact(a: Assignment, b: Assignment) -> bool:
    return (a.request_id, a.satellite_id, a.station_id, a.start, a.end,
            a.reserved_start, a.reserved_end) == (b.request_id, b.satellite_id,
            b.station_id, b.start, b.end, b.reserved_start, b.reserved_end)


def assignment(scenario: Scenario, candidate: Candidate) -> Assignment:
    values = {k: scenario.start + timedelta(seconds=getattr(candidate, k))
              for k in ("reserved_start", "start", "end", "reserved_end")}
    return Assignment(request_id=candidate.request_id,
                      satellite_id=candidate.satellite_id,
                      station_id=candidate.station_id, window_id=candidate.window_id,
                      **values)


def conflicts(a: Candidate, b: Candidate) -> bool:
    return (a.station_id == b.station_id or a.satellite_id == b.satellite_id) and (
        a.reserved_start < b.reserved_end and b.reserved_start < a.reserved_end)


def generate_candidates(s: Scenario, previous: list[Assignment]):
    old = {a.request_id: a for a in previous}
    if len(old) != len(previous):
        raise ValueError("Previous schedule contains duplicate request IDs.")
    if any(a.reserved_start < s.decision_time for a in previous):
        raise ValueError("This MVP replans future contacts only. Decision time must be before "
                         "every previous reservation begins; archive started contacts first.")
    stations = {a.id: a for a in s.stations}
    sats = {a.id: a for a in s.satellites}
    by_satellite = defaultdict(list)
    for win in s.windows:
        by_satellite[win.satellite_id].append(win)
    cutoff = seconds(s, s.decision_time) + s.booking_lead_s
    result: list[Candidate] = []
    diagnostics = {}
    required = set()
    warnings = []
    for req in s.requests:
        allowed = set(sats[req.satellite_id].allowed_station_ids)
        diag = {"compatible_windows": 0, "fitting_windows": 0,
                "lead_time_rejections": 0, "outage_rejections": 0, "candidate_count": 0}
        seen = set()
        preserved_count = 0
        prev = old.get(req.id)
        for win in sorted(by_satellite[req.satellite_id], key=lambda w: (w.start, w.id)):
            if win.station_id not in allowed:
                continue
            diag["compatible_windows"] += 1
            st = stations[win.station_id]
            lo = math.ceil(max(seconds(s, win.start) + st.acquisition_s,
                               seconds(s, req.earliest)))
            hi = math.floor(min(seconds(s, win.end) - st.turnaround_s - req.duration_s,
                                seconds(s, req.deadline) - req.duration_s))
            if hi < lo:
                continue
            diag["fitting_windows"] += 1
            starts = set(range(math.ceil(lo / s.slot_seconds) * s.slot_seconds,
                               hi + 1, s.slot_seconds))
            # A valid previous booking is retainable even off the regular grid.
            if prev and prev.station_id == st.id:
                old_start = seconds(s, prev.start)
                if old_start.is_integer() and lo <= old_start <= hi:
                    starts.add(int(old_start))
            for start in sorted(starts):
                key = (st.id, start)
                if key in seen:
                    continue
                c = Candidate(req.id, req.satellite_id, st.id, win.id,
                              start-st.acquisition_s, start, start+req.duration_s,
                              start+req.duration_s+st.turnaround_s)
                preserved = bool(prev and same_contact(assignment(s, c), prev))
                if c.reserved_start < cutoff and not preserved:
                    diag["lead_time_rejections"] += 1
                    continue
                if any(o.station_id == st.id and
                       c.reserved_start < seconds(s, o.end) and
                       seconds(s, o.start) < c.reserved_end for o in s.outages):
                    diag["outage_rejections"] += 1
                    continue
                c = Candidate(**{**c.__dict__, "preserved": preserved})
                seen.add(key)
                result.append(c)
                preserved_count += int(preserved)
                diag["candidate_count"] += 1
                if len(result) > MAX_CANDIDATES:
                    raise ValueError(f"More than {MAX_CANDIDATES:,} candidates. Shorten the horizon, "
                                     "reduce requests, or increase slot_seconds (maximum 120).")
        if prev and (req.locked or seconds(s, prev.reserved_start) < cutoff):
            if preserved_count:
                required.add(req.id)
            else:
                warnings.append(f"{req.id}: the old locked/lead-time-protected contact is no longer "
                                "eligible. Its lock is explicitly released in this proposal. "
                                "Operator approval is required; no external booking was changed.")
        if not diag["candidate_count"]:
            if not allowed:
                reason = "No station is authorised for this satellite in the scenario."
            elif not diag["compatible_windows"]:
                reason = "No visibility window is listed at an authorised station."
            elif not diag["fitting_windows"]:
                reason = "No window fits the duration, both buffers, earliest start and deadline."
            elif diag["outage_rejections"]:
                reason = "All remaining grid candidates are blocked by outages or booking lead time."
            elif diag["lead_time_rejections"]:
                reason = "The booking lead time rules out all candidate start times."
            else:
                reason = "The eligible interval contains no start on the configured time grid."
        else:
            reason = "Eligible options exist, but compete with other requests under the selected plan."
        diagnostics[req.id] = {**diag, "reason": reason}
    return result, required, diagnostics, warnings


def greedy_schedule(s: Scenario, candidates: list[Candidate], required: set[str]):
    """Priority-aware, earliest-deadline-first baseline under identical constraints."""
    by_req = defaultdict(list)
    for c in candidates:
        by_req[c.request_id].append(c)
    selected = []
    for rid in sorted(required):
        choices = [c for c in by_req[rid] if c.preserved]
        if not choices or any(conflicts(choices[0], c) for c in selected):
            raise ValueError("Existing protected bookings conflict. Resolve the input commitments first.")
        selected.append(choices[0])
    for req in sorted(s.requests, key=lambda r: (PRIORITY_ORDER[r.priority], r.deadline, r.id)):
        if req.id in required:
            continue
        # During repair, preserve the old contact first, then choose the earliest slot.
        for c in sorted(by_req[req.id], key=lambda c: (not c.preserved, c.start, c.station_id)):
            if not any(conflicts(c, other) for other in selected):
                selected.append(c)
                break
    return selected


def constraint_matrix(candidates: list[Candidate], required: set[str]):
    rows, cols, vals, lower, upper = [], [], [], [], []
    def add(indices: Iterable[int], lo=0, hi=1):
        indices = list(indices)
        if not indices:
            return
        row = len(lower)
        rows.extend([row]*len(indices)); cols.extend(indices); vals.extend([1.0]*len(indices))
        lower.append(lo); upper.append(hi)
    by_req = defaultdict(list)
    resources = defaultdict(list)
    for i, c in enumerate(candidates):
        by_req[c.request_id].append(i)
        resources[("station", c.station_id)].append(i)
        resources[("satellite", c.satellite_id)].append(i)
    for rid, indices in by_req.items():
        add(indices)
        if rid in required:
            add([i for i in indices if candidates[i].preserved], 1, 1)
    # At every event boundary, constrain the active clique. End events precede
    # starts, so back-to-back half-open reservations are allowed.
    known = set()
    for indices in resources.values():
        events = defaultdict(lambda: {"start": [], "end": []})
        for i in indices:
            c = candidates[i]
            events[c.reserved_start]["start"].append(i)
            events[c.reserved_end]["end"].append(i)
        active = set()
        for t in sorted(events):
            active.difference_update(events[t]["end"])
            active.update(events[t]["start"])
            if len(active) > 1:
                key = tuple(sorted(active))
                if key not in known:
                    add(key); known.add(key)
        if len(vals) > 2000000:
            raise ValueError("Scenario is too dense for this local MVP. Reduce requests or windows.")
    matrix = coo_matrix((vals, (rows, cols)), shape=(len(lower), len(candidates))).tocsc()
    return matrix, np.asarray(lower, float), np.asarray(upper, float)


def optimise(s: Scenario, candidates: list[Candidate], required: set[str], limit: float,
             fallback: list[Candidate]):
    if not candidates:
        return [], {"status": "OPTIMAL", "grid_optimal": True, "stages": [],
                    "message": "No eligible candidate contacts.", "runtime_s": 0}
    started = time.perf_counter()
    matrix, lower, upper = constraint_matrix(candidates, required)
    reqs = {r.id: r for r in s.requests}
    n = len(candidates)
    critical = np.array([int(reqs[c.request_id].priority == "critical") for c in candidates], float)
    utility = np.array([SECONDARY_VALUE[reqs[c.request_id].priority] for c in candidates], float)
    preserved = np.array([int(c.preserved) for c in candidates], float)
    earliest = np.array([c.start for c in candidates], float)
    stages = [("critical requests", -critical), ("other-request utility", -utility),
              ("unchanged contacts", -preserved), ("earlier payload starts", earliest)]
    logs = []
    best = None
    optimal = True
    status = "OPTIMAL"
    for name, objective in stages:
        if not np.any(objective):
            continue
        remaining = limit - (time.perf_counter()-started)
        if remaining < 0.05:
            optimal = False; status = "FEASIBLE_TIME_LIMIT"; break
        result = milp(c=objective, integrality=np.ones(n, dtype=int),
                      bounds=Bounds(np.zeros(n), np.ones(n)),
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={"time_limit": remaining, "mip_rel_gap": 0.0, "presolve": True})
        logs.append({"objective": name, "status_code": int(result.status),
                     "message": str(result.message)})
        if result.x is not None:
            rounded = np.rint(result.x)
            # Independently check binary feasibility before trusting an incumbent.
            ax = matrix @ rounded
            if np.max(np.abs(result.x-rounded)) < 1e-4 and np.all(ax >= lower-1e-6) and np.all(ax <= upper+1e-6):
                best = rounded
        if result.status != 0:
            optimal = False
            status = "FEASIBLE_TIME_LIMIT" if best is not None else "HEURISTIC_FALLBACK"
            if result.status == 2:
                raise ValueError("No feasible schedule can honour the protected existing contacts.")
            break
        if best is None:
            optimal = False; status = "HEURISTIC_FALLBACK"; break
        value = round(float(objective @ best))
        logs[-1]["attained_value"] = value
        matrix = vstack([matrix, csc_matrix(objective.reshape(1, -1))], format="csc")
        lower = np.append(lower, value); upper = np.append(upper, value)
    if best is None:
        selected = fallback
        optimal = False
        status = "HEURISTIC_FALLBACK"
    else:
        selected = [c for i, c in enumerate(candidates) if best[i] > 0.5]
    def quality(items):
        return (sum(reqs[c.request_id].priority == "critical" for c in items),
                sum(SECONDARY_VALUE[reqs[c.request_id].priority] for c in items),
                sum(c.preserved for c in items), -sum(c.start for c in items))
    if not optimal and quality(fallback) > quality(selected):
        selected = fallback; status = "HEURISTIC_FALLBACK"
    return selected, {"status": status, "grid_optimal": optimal, "stages": logs,
                      "runtime_s": round(time.perf_counter()-started, 4),
                      "message": (f"Optimal on the {s.slot_seconds}-second candidate grid; not a continuous-time optimum."
                                  if optimal else "Feasible plan; the full ordered objective was not proved optimal.")}
