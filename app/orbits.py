"""Optional Skyfield adapter; isolated from the tested offline scheduling core.

No approximate substitute is used when Skyfield is missing. The endpoint reports
that the optional dependency must be installed. Imports are OMM JSON, not commands.
"""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import math
from .models import Scenario, Satellite, ContactWindow, Request, OrbitImport


def parse_epoch(value: str) -> datetime:
    value = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    # In OMM UTC records, EPOCH conventionally has no literal timezone suffix.
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def build_orbit_scenario(payload: OrbitImport) -> Scenario:
    try:
        from skyfield.api import EarthSatellite, load, wgs84
    except ImportError as exc:
        raise ImportError("Orbit calculations need the optional package. In your project environment, run: "
                          "python -m pip install -r requirements-orbits.txt ; then restart the server.") from exc
    if not payload.simulate_all_pairings:
        raise ValueError("Explicitly acknowledge simulated station compatibility. Orbital records do not grant access.")
    required = {'NORAD_CAT_ID','OBJECT_NAME','EPOCH','MEAN_MOTION','ECCENTRICITY','INCLINATION',
                'RA_OF_ASC_NODE','ARG_OF_PERICENTER','MEAN_ANOMALY','BSTAR',
                'MEAN_MOTION_DOT','MEAN_MOTION_DDOT','REV_AT_EPOCH','EPHEMERIS_TYPE'}
    records=[]
    epochs=[]
    for raw in payload.records:
        missing=required-set(raw)
        if missing:
            raise ValueError('Missing OMM fields: '+', '.join(sorted(missing)))
        record=dict(raw)
        for key in ('MEAN_MOTION','ECCENTRICITY','INCLINATION','RA_OF_ASC_NODE',
                    'ARG_OF_PERICENTER','MEAN_ANOMALY','BSTAR','MEAN_MOTION_DOT','MEAN_MOTION_DDOT'):
            if not math.isfinite(float(record[key])):
                raise ValueError(f'Non-finite orbit value: {key}')
        # This adapter is deliberately scoped to low-eccentricity LEO examples.
        if not 11 <= float(record['MEAN_MOTION']) <= 17.5 or not 0 <= float(record['ECCENTRICITY']) <= .1:
            raise ValueError('This LEO prototype requires mean motion 11–17.5 rev/day and eccentricity 0–0.1.')
        if not 0 <= float(record['INCLINATION']) <= 180:
            raise ValueError('Invalid orbit inclination.')
        if str(record.get('CENTER_NAME','EARTH')).upper()!='EARTH' or str(record.get('REF_FRAME','TEME')).upper()!='TEME':
            raise ValueError('Only Earth-centred TEME OMM is supported.')
        if str(record.get('TIME_SYSTEM','UTC')).upper()!='UTC' or str(record.get('MEAN_ELEMENT_THEORY','SGP4')).upper()!='SGP4':
            raise ValueError('Only UTC / SGP4 OMM is supported.')
        record.update(CENTER_NAME='EARTH', REF_FRAME='TEME', TIME_SYSTEM='UTC', MEAN_ELEMENT_THEORY='SGP4')
        record.setdefault('OBJECT_ID',''); record.setdefault('CLASSIFICATION_TYPE','U'); record.setdefault('ELEMENT_SET_NO',0)
        epoch=parse_epoch(record['EPOCH'])
        epochs.append(epoch)
        record['EPOCH']=epoch.strftime('%Y-%m-%dT%H:%M:%S.%f')
        records.append(record)
    if len({int(r['NORAD_CAT_ID']) for r in records}) != len(records):
        raise ValueError('One OMM record per catalogue ID is required.')
    start=payload.start or max(epochs).replace(microsecond=0)
    if start.tzinfo is None:
        raise ValueError('Planning start must include a timezone.')
    start=start.astimezone(timezone.utc)
    end=start+timedelta(hours=payload.horizon_hours)
    if any(max(abs((start-e).total_seconds()),abs((end-e).total_seconds()))>7*86400 for e in epochs):
        raise ValueError('Planning period is more than 7 days from an orbit epoch. Use a near-epoch replay or fresher records.')
    ts=load.timescale(builtin=True)  # no network access / planetary ephemeris required
    satellites=[]; windows=[]
    for record in records:
        sat=EarthSatellite.from_omm(ts,record)
        sat_id=f"NORAD-{int(record['NORAD_CAT_ID'])}"
        satellites.append(Satellite(id=sat_id,name=str(record['OBJECT_NAME']),orbit=record,
                                    allowed_station_ids=[st.id for st in payload.stations]))
        for st in payload.stations:
            site=wgs84.latlon(st.latitude,st.longitude,elevation_m=st.altitude_m)
            relative=sat-site
            a,b=ts.from_datetime(start),ts.from_datetime(end)
            def altitude(dt):
                position=relative.at(ts.from_datetime(dt))
                if position.message is not None:
                    raise ValueError(f'Propagation failed for {sat_id}: {position.message}')
                value=float(position.altaz()[0].degrees)
                if not math.isfinite(value):
                    raise ValueError(f'Non-finite propagation for {sat_id}')
                return value
            opened=start if altitude(start)>=st.minimum_elevation_deg else None
            peak=altitude(start) if opened else 0.0
            times, events=sat.find_events(site,a,b,altitude_degrees=st.minimum_elevation_deg)
            def append_window(left,right,peak_value):
                # Clip to horizon and round inwards to integer seconds, never outwards.
                left=max(start,left); right=min(end,right)
                left=start+timedelta(seconds=math.ceil((left-start).total_seconds()))
                right=start+timedelta(seconds=math.floor((right-start).total_seconds()))
                if right>left:
                    windows.append(ContactWindow(id=f'ORB-{len(windows)+1:04}',satellite_id=sat_id,
                                                 station_id=st.id,start=left,end=right,
                                                 peak_elevation_deg=min(90,max(0,peak_value))))
            for moment,event in zip(times,events):
                dt=moment.utc_datetime()
                if event==0:
                    opened=dt; peak=st.minimum_elevation_deg
                elif event==1 and opened is not None:
                    peak=max(peak,altitude(dt))
                elif event==2 and opened is not None:
                    append_window(opened,dt,peak); opened=None
            if opened is not None:
                append_window(opened,end,max(peak,altitude(end)))
    requests=[]
    # A deliberately fictional workload, built only to exercise imported geometry.
    for sat in satellites:
        for part in range(3):
            lo=start+(end-start)*part/3
            hi=start+(end-start)*(part+1)/3
            requests.append(Request(id=f'Q-{len(requests)+1:02}',satellite_id=sat.id,
                label=f'Simulated batch {part+1}',duration_s=120,earliest=lo,deadline=hi,
                priority='critical' if part==0 else 'routine'))
    return Scenario(name='OMM orbital replay · Simulated workload',mode='orbit_replay',start=start,end=end,
        decision_time=start-timedelta(hours=1),stations=payload.stations,satellites=satellites,
        windows=windows,requests=requests,metadata={
            'provenance':'Geometry calculated from user-supplied OMM using Skyfield/SGP4. Requests, stations and all compatibility pairings are simulated.',
            'orbit_epochs':{f"NORAD-{int(r['NORAD_CAT_ID'])}":str(r['EPOCH']) for r in records},
            'station_status':'Not real ground-station inventory or permission to communicate.',
            'assumptions':['Constant minimum-elevation mask; no terrain or atmospheric refraction.',
                           'Single antenna and link; fixed buffers and request durations.',
                           'Predicted geometry is not proof of a usable radio link.',
                           'Not flight-qualified; verify against operator ephemerides before operational use.']})
