from __future__ import annotations
import importlib.util
import json
import logging
from pathlib import Path
from threading import Lock
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from . import __version__
from .models import Scenario, PlanInput, OrbitImport
from .service import plan
from .orbits import build_orbit_scenario

ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title='PassRescue API',version=__version__,description='Local decision-support prototype. No live spacecraft or station control.')
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)
STATIC_DIR = ROOT / "static"

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=STATIC_DIR),
        name="static"
    )
solve_lock=Lock()

@app.middleware('http')
async def headers(request: Request,call_next):
    content_length=request.headers.get('content-length','0')
    if content_length.isdigit() and int(content_length)>2_000_000:
        return JSONResponse(status_code=413,content={'detail':'Payload too large; keep scenario JSON below 2 MB.'})
    response=await call_next(request)
    response.headers['X-Content-Type-Options']='nosniff'
    response.headers['X-Frame-Options']='DENY'
    response.headers['Referrer-Policy']='no-referrer'
    if request.url.path=='/':
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
    return response

@app.get("/", include_in_schema=False)
def index():
    return {
        "status": "online",
        "service": "PassRescue Backend"
    }
@app.get('/api/health')
def health():
    return {'status':'ok','version':__version__,'solver':'SciPy / HiGHS MILP',
            'skyfield_available':importlib.util.find_spec('skyfield') is not None,
            'live_control':False}

@app.get('/api/demo',response_model=Scenario)
def demo():
    return Scenario.model_validate_json((ROOT/'data'/'demo_scenario.json').read_text(encoding='utf-8'))

@app.get('/api/sample-orbits')
def sample_orbits():
    return json.loads((ROOT/'data'/'sample_omm_2024.json').read_text(encoding='utf-8'))

@app.post('/api/validate-scenario',response_model=Scenario)
def validate_scenario(scenario:Scenario):
    return scenario

@app.post('/api/plan')
def make_plan(payload:PlanInput):
    if not solve_lock.acquire(blocking=False):
        raise HTTPException(409,'Another calculation is running. Run one calculation at a time.')
    try:
        return plan(payload)
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc
    except RuntimeError as exc:
        logging.exception('Planner validation failure')
        raise HTTPException(500,str(exc)) from exc
    finally:
        solve_lock.release()

@app.post('/api/orbits',response_model=Scenario)
def import_orbits(payload:OrbitImport):
    if not solve_lock.acquire(blocking=False):
        raise HTTPException(409,'Another calculation is running.')
    try:
        return build_orbit_scenario(payload)
    except ImportError as exc:
        raise HTTPException(503,str(exc)) from exc
    except (ValueError,KeyError,TypeError,OverflowError) as exc:
        raise HTTPException(422,str(exc)) from exc
    finally:
        solve_lock.release()
