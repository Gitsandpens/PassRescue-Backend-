"""Independent output checker. Does NOT reuse candidate generation or solver constraints."""
from __future__ import annotations
from datetime import timedelta
from .models import Scenario, Assignment


def contact_errors(s: Scenario, a: Assignment, previous: list[Assignment] | None = None) -> list[str]:
    reqs = {r.id: r for r in s.requests}
    stations = {r.id: r for r in s.stations}
    satellites = {r.id: r for r in s.satellites}
    windows = {r.id: r for r in s.windows}
    errors = []
    if a.request_id not in reqs:
        return [f"{a.request_id}: unknown request"]
    if a.station_id not in stations or a.window_id not in windows:
        return [f"{a.request_id}: unknown station/window"]
    r, station, window = reqs[a.request_id], stations[a.station_id], windows[a.window_id]
    if a.satellite_id != r.satellite_id:
        errors.append(f"{r.id}: wrong satellite")
    if a.station_id not in satellites[r.satellite_id].allowed_station_ids:
        errors.append(f"{r.id}: station not authorised")
    if (window.satellite_id, window.station_id) != (a.satellite_id, a.station_id):
        errors.append(f"{r.id}: wrong contact window")
    if (a.end-a.start).total_seconds() != r.duration_s:
        errors.append(f"{r.id}: wrong payload duration")
    if (a.start-a.reserved_start).total_seconds() != station.acquisition_s:
        errors.append(f"{r.id}: incorrect acquisition reservation")
    if (a.reserved_end-a.end).total_seconds() != station.turnaround_s:
        errors.append(f"{r.id}: incorrect turnaround reservation")
    if not window.start <= a.reserved_start <= a.start < a.end <= a.reserved_end <= window.end:
        errors.append(f"{r.id}: reservation outside visibility window")
    if a.start < r.earliest or a.end > r.deadline:
        errors.append(f"{r.id}: earliest/deadline violation")
    if a.reserved_start < s.start or a.reserved_end > s.end:
        errors.append(f"{r.id}: reservation outside planning horizon")
    old_matches = any((b.request_id, b.satellite_id, b.station_id, b.start, b.end,
                      b.reserved_start, b.reserved_end) ==
                     (a.request_id, a.satellite_id, a.station_id, a.start, a.end,
                      a.reserved_start, a.reserved_end) for b in (previous or []))
    if a.reserved_start < s.decision_time + timedelta(seconds=s.booking_lead_s) and not old_matches:
        errors.append(f"{r.id}: booking lead time violated")
    for o in s.outages:
        if o.station_id == a.station_id and a.reserved_start < o.end and o.start < a.reserved_end:
            errors.append(f"{r.id}: outage overlap ({o.id})")
    return errors


def validate_schedule(s: Scenario, schedule: list[Assignment], previous: list[Assignment] | None = None):
    errors = []
    req_ids = [a.request_id for a in schedule]
    if len(req_ids) != len(set(req_ids)):
        errors.append("A request is assigned more than once")
    for a in schedule:
        errors.extend(contact_errors(s, a, previous))
    for i, a in enumerate(schedule):
        for b in schedule[i+1:]:
            overlapping = a.reserved_start < b.reserved_end and b.reserved_start < a.reserved_end
            if overlapping and a.station_id == b.station_id:
                errors.append(f"Antenna double-booking: {a.request_id} / {b.request_id}")
            if overlapping and a.satellite_id == b.satellite_id:
                errors.append(f"Satellite link double-booking: {a.request_id} / {b.request_id}")
    reqs = {r.id: r for r in s.requests}
    cutoff = s.decision_time + timedelta(seconds=s.booking_lead_s)
    for old in previous or []:
        req = reqs.get(old.request_id)
        if req and (req.locked or old.reserved_start < cutoff) and not contact_errors(s, old, previous):
            if not any((a.request_id, a.station_id, a.start, a.end, a.reserved_start, a.reserved_end) ==
                       (old.request_id, old.station_id, old.start, old.end, old.reserved_start, old.reserved_end)
                       for a in schedule):
                errors.append(f"{old.request_id}: eligible protected contact was not preserved")
    return {"valid": not errors, "violation_count": len(errors), "errors": errors,
            "checks": ["request uniqueness", "explicit compatibility", "window containment",
                       "payload duration", "preparation buffers", "earliest start and deadline",
                       "station outages", "antenna exclusivity", "satellite exclusivity",
                       "booking lead time", "eligible protected contacts"],
            "scope": "Checks the declared scenario, not radio-link performance or operational certification."}
