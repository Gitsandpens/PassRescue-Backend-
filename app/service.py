from __future__ import annotations
import time
from .models import Scenario, Assignment, PlanInput
from .planner import generate_candidates, greedy_schedule, optimise, assignment, same_contact, SECONDARY_VALUE
from .validator import validate_schedule, contact_errors


def metrics(s: Scenario, schedule: list[Assignment]):
    reqs = {r.id: r for r in s.requests}
    chosen = [reqs[a.request_id] for a in schedule if a.request_id in reqs]
    return {"scheduled": len(chosen), "requested": len(s.requests),
            "unserved": len(s.requests)-len(chosen),
            "critical_scheduled": sum(r.priority == "critical" for r in chosen),
            "critical_requested": sum(r.priority == "critical" for r in s.requests),
            "other_utility": sum(SECONDARY_VALUE[r.priority] for r in chosen),
            "planned_payload_minutes": round(sum(r.duration_s for r in chosen)/60, 2)}


def plan(payload: PlanInput):
    started = time.perf_counter()
    s, previous = payload.scenario, payload.previous
    candidates, protected, diagnostics, warnings = generate_candidates(s, previous)
    baseline_candidates = greedy_schedule(s, candidates, protected)
    selected, solver = optimise(s, candidates, protected, payload.time_limit_s, baseline_candidates)
    schedule = sorted([assignment(s, c) for c in selected], key=lambda a: (a.start, a.station_id))
    baseline = sorted([assignment(s, c) for c in baseline_candidates], key=lambda a: (a.start, a.station_id))
    validation = validate_schedule(s, schedule, previous)
    baseline_validation = validate_schedule(s, baseline, previous)
    if not validation["valid"] or not baseline_validation["valid"]:
        raise RuntimeError("Internal schedule validation failed; no invalid schedule will be published.")
    old = {a.request_id: a for a in previous}
    new = {a.request_id: a for a in schedule}
    reqs = {r.id: r for r in s.requests}
    changes = []
    affected = {a.request_id for a in previous if contact_errors(s, a, previous)}
    for req in s.requests:
        before, after = old.get(req.id), new.get(req.id)
        if before and after and same_contact(before, after):
            status = "unchanged"
            reason = "Original contact is still eligible and has been preserved."
        elif before and after:
            status = "recovered" if req.id in affected else "moved"
            reason = ("Original contact is no longer eligible. This replacement fits the declared window, "
                      "deadline, compatibility and reservation rules." if req.id in affected else
                      "Moved as part of the selected overall plan. The replacement satisfies all declared constraints.")
        elif after:
            status = "scheduled"
            reason = "Fits an authorised window, the deadline and both preparation buffers without a resource conflict."
        else:
            status = "unserved"
            reason = diagnostics[req.id]["reason"]
            if diagnostics[req.id]["candidate_count"] and not solver["grid_optimal"]:
                reason += " Search was time-limited; infeasibility is not proved."
        changes.append({"request_id": req.id, "status": status, "reason": reason,
                        "before": before, "after": after, "protected": req.id in protected,
                        "diagnostics": diagnostics[req.id]})
    surviving = [a for a in previous if not contact_errors(s, a, previous)]
    unserved = [c for c in changes if c["status"] == "unserved"]
    base_metrics, final_metrics = metrics(s, baseline), metrics(s, schedule)
    final_metrics["unchanged"] = sum(c["status"] == "unchanged" for c in changes)
    base_metrics["unchanged"] = sum(a.request_id in old and same_contact(a, old[a.request_id]) for a in baseline)
    return {"schedule": schedule, "baseline_schedule": baseline, "metrics": final_metrics,
            "baseline_metrics": base_metrics, "baseline_validation": baseline_validation,
            "solver": solver, "validation": validation, "changes": changes, "unserved": unserved,
            "warnings": warnings, "candidate_count": len(candidates),
            "recovery": {"is_recovery": bool(previous), "affected": len(affected),
                         "recovered": sum(c["status"] == "recovered" for c in changes),
                         "moved_unaffected": sum(c["status"] == "moved" for c in changes),
                         "unrepaired_metrics": metrics(s, surviving)},
            "objectives": ["Maximise number of critical requests",
                           "Maximise other utility (high=3, routine=1)",
                           "Maximise exactly unchanged existing contacts", "Prefer earlier total payload starts"],
            "runtime_s": round(time.perf_counter()-started, 4),
            "notice": "Proposed schedule only. No reservation, spacecraft command or data transfer is performed."}
