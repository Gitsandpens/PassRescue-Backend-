"""Input contracts. All internal times are timezone-aware UTC; all intervals [start, end)."""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

class Station(Model):
    id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=100)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    altitude_m: float = Field(default=0, ge=-500, le=9000)
    minimum_elevation_deg: float = Field(default=10, ge=0, le=80)
    acquisition_s: int = Field(default=30, ge=0, le=900)
    turnaround_s: int = Field(default=30, ge=0, le=900)

class Satellite(Model):
    id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=100)
    allowed_station_ids: list[str] = Field(default_factory=list, max_length=10)
    orbit: dict[str, Any] | None = None

class Timed(Model):
    @field_validator("*", mode="after", check_fields=False)
    @classmethod
    def utc_datetimes(cls, value):
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("Include a timezone: use ISO 8601 with Z or an explicit offset.")
            return value.astimezone(timezone.utc)
        return value

class ContactWindow(Timed):
    id: str = Field(min_length=1, max_length=80)
    satellite_id: str
    station_id: str
    start: datetime
    end: datetime
    peak_elevation_deg: float | None = Field(default=None, ge=0, le=90)
    @model_validator(mode="after")
    def valid_interval(self):
        if self.end <= self.start:
            raise ValueError("Contact window end must follow start.")
        return self

class Request(Timed):
    id: str = Field(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")
    label: str = Field(min_length=1, max_length=120)
    satellite_id: str
    duration_s: int = Field(ge=30, le=7200)
    priority: Literal["critical", "high", "routine"] = "routine"
    earliest: datetime
    deadline: datetime
    locked: bool = False
    @model_validator(mode="after")
    def valid_interval(self):
        if self.deadline <= self.earliest:
            raise ValueError("Request deadline must follow earliest start.")
        return self

class Outage(Timed):
    id: str = Field(min_length=1, max_length=60)
    station_id: str
    start: datetime
    end: datetime
    reason: str = Field(default="Operator-declared outage", max_length=160)
    @model_validator(mode="after")
    def valid_interval(self):
        if self.end <= self.start:
            raise ValueError("Outage end must follow start.")
        return self

class Scenario(Timed):
    name: str = Field(min_length=1, max_length=100)
    mode: Literal["synthetic", "orbit_replay"] = "synthetic"
    start: datetime
    end: datetime
    decision_time: datetime
    slot_seconds: int = Field(default=30, ge=10, le=120)
    booking_lead_s: int = Field(default=900, ge=0, le=7200)
    stations: list[Station] = Field(min_length=1, max_length=10)
    satellites: list[Satellite] = Field(min_length=1, max_length=30)
    windows: list[ContactWindow] = Field(default_factory=list, max_length=3000)
    requests: list[Request] = Field(default_factory=list, max_length=100)
    outages: list[Outage] = Field(default_factory=list, max_length=40)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_references(self):
        if not 0 < (self.end-self.start).total_seconds() <= 172800:
            raise ValueError("Planning horizon must be positive and at most 48 hours.")
        if self.decision_time > self.end:
            raise ValueError("Decision time must not follow the planning horizon.")
        for name in ("stations", "satellites", "windows", "requests", "outages"):
            objects = getattr(self, name)
            ids = [o.id for o in objects]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate IDs in {name}.")
        stations = {s.id for s in self.stations}
        sats = {s.id for s in self.satellites}
        for sat in self.satellites:
            if not set(sat.allowed_station_ids) <= stations:
                raise ValueError(f"Unknown authorised station on satellite {sat.id}.")
        for win in self.windows:
            if win.station_id not in stations or win.satellite_id not in sats:
                raise ValueError(f"Unknown satellite/station in window {win.id}.")
            if win.start < self.start or win.end > self.end:
                raise ValueError(f"Window {win.id} lies outside the planning horizon.")
        for req in self.requests:
            if req.satellite_id not in sats:
                raise ValueError(f"Unknown satellite in request {req.id}.")
            if req.earliest < self.start or req.deadline > self.end:
                raise ValueError(f"Request {req.id} lies outside the planning horizon.")
        if any(o.station_id not in stations for o in self.outages):
            raise ValueError("Outage refers to an unknown station.")
        return self

class Assignment(Timed):
    request_id: str
    satellite_id: str
    station_id: str
    window_id: str
    reserved_start: datetime
    start: datetime
    end: datetime
    reserved_end: datetime

class PlanInput(Model):
    scenario: Scenario
    previous: list[Assignment] = Field(default_factory=list, max_length=100)
    time_limit_s: float = Field(default=8, ge=1, le=20)

class OrbitInput(Model):
    scenario: Scenario
    max_epoch_distance_days: float = Field(default=7, ge=0.01, le=14)
    omm_records: list[dict[str, Any]] | None = Field(default=None, max_length=100)

class OrbitImport(Model):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=8)
    stations: list[Station] = Field(min_length=1, max_length=5)
    simulate_all_pairings: bool = False
    start: datetime | None = None
    horizon_hours: int = Field(default=24, ge=1, le=24)
