"""
HTTP API.   Run:  python -m app            (serves API + built frontend on :8000)

All quantities are SI; every response that carries numbers says so.
"""
import csv
import io
import json
import os
import subprocess

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import config as C
from app import APP_VERSION, cache, geometry, pipeline
from app import params as P
from app.jobs import JobManager, SUCCEEDED
from app.schemas import FutureSimRequest, SimConfig, SubmitRequest
from app import verification as V

api = FastAPI(title="emg_twin digital phantom", version=APP_VERSION)
JOBS = JobManager()


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=C.ROOT, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


GIT_COMMIT = _git_commit()

STAGES = [
    dict(stage=1, name="Geometry, mesh, tissue conductivities", status="integrated",
         code=["mesh.py", "config.py"]),
    dict(stage=2, name="FEM matrix, selection matrix, CG, adjoint", status="integrated", code=["fem.py"]),
    dict(stage=3, name="Single-fibre action potential, volume conduction, electrode array",
         status="integrated", code=["fibre.py"]),
    dict(stage=4, name="Motor units, recruitment, combined muscle activity",
         status="implemented as scripts, not integrated in the app",
         code=["muscle.py", "stage4_muscle.py"],
         integration_point="POST /api/simulations/future with kind='motor_unit_pool' (returns 501 for now); "
                           "result schema reserves fibre.motor_unit and per-MU fields"),
    dict(stage=5, name="Timing and experiment C", status="implemented as scripts, not integrated in the app",
         code=["stage5_timing.py", "stage5_plot.py"],
         integration_point="POST /api/simulations/future with kind='timing' (returns 501 for now)"),
]


# --------------------------------------------------------------------------- model / parameters
@api.get("/api/health")
def health():
    return dict(ok=True, model_version=cache.MODEL_VERSION, app_version=APP_VERSION)


@api.get("/api/model")
def model():
    return dict(
        model_version=cache.MODEL_VERSION, app_version=APP_VERSION, git_commit=GIT_COMMIT,
        reference="Maksymenko et al., Nat. Commun. 14:1600 (2023), simplified re-implementation",
        coordinate_system="right-handed (x, y, z) in metres; z = arm long axis = fibre direction; origin on the "
                          "axis at mid-length; theta = atan2(y, x) with theta = 0 on +x (default electrode side)",
        units=dict(length="m", time="s", conductivity="S/m", current="A", voltage="V", angle="rad"),
        formulation=dict(
            equation="quasi-static Poisson  div(sigma grad phi) = -I_source",
            boundary="insulating (homogeneous Neumann) on the whole outer surface incl. end caps",
            reference="zero-mean potential (constant null space removed); lone unit sources balanced by a "
                      "uniform sink (fem.cg_solve)",
            sources="fibre membrane current, Eq. (8); current entering tissue is positive",
            electrodes="point electrodes, barycentric P1 interpolation (fem.selection_matrix)",
            differential="V_EMG(t) = phi_1(t) - phi_2(t) for consecutive electrodes",
            time="quasi-static: each time sample is an independent static solution superposed via Eq. (10)",
        ),
        stages=STAGES,
        tissues=[{k: v for k, v in t.items() if k != "label"} for t in geometry.TISSUES],
    )


@api.get("/api/parameters")
def parameters():
    return dict(parameters=P.PARAMETERS, unsupported=P.UNSUPPORTED, recompute_levels=P.RECOMPUTE_LEVELS)


@api.get("/api/config/default")
def default_config():
    return SimConfig().model_dump(mode="json")


@api.post("/api/config/diff")
def config_diff(body: dict):
    a = SimConfig.model_validate(body["a"]).model_dump(mode="json")
    b = SimConfig.model_validate(body["b"]).model_dump(mode="json")
    changes = P.diff_configs(a, b)
    return dict(changes=[dict(key=k, a=x, b=y) for k, x, y in changes],
                recompute=P.recompute_level([k for k, _, _ in changes]) if changes else None)


# --------------------------------------------------------------------------- preview / geometry
@api.post("/api/preview")
def preview(cfg: SimConfig):
    return pipeline.preview(cfg)


@api.get("/api/geometry/{mesh_key}")
def geometry_for(mesh_key: str):
    payload = cache.SURFACES.get(mesh_key)
    if payload is None:
        m = cache.MESHES.get(mesh_key)
        if m is None:
            raise HTTPException(404, "unknown mesh key; call POST /api/preview first")
        payload = geometry.geometry_payload(m)
        cache.SURFACES.put(mesh_key, payload)
    return payload


# --------------------------------------------------------------------------- simulations
def _job_view(job):
    out = job.summary()
    out["latest_seq"] = JOBS.latest_seq()
    if job.status == SUCCEEDED:
        out["result"] = job.result
    return out


@api.post("/api/simulations", status_code=202)
def submit(req: SubmitRequest):
    job = JOBS.submit(req.config, req.label)
    return _job_view(job)


@api.post("/api/simulations/future", status_code=501)
def submit_future(req: FutureSimRequest):
    raise HTTPException(501, f"'{req.kind}' simulations (Stage 4/5) are implemented as scripts "
                             f"(stage4_muscle.py, stage5_timing.py) but not yet integrated into the app")


@api.get("/api/simulations")
def list_jobs():
    return dict(jobs=JOBS.list(), latest_seq=JOBS.latest_seq())


@api.get("/api/simulations/{job_id}")
def get_job(job_id: str, include_result: bool = True):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    v = _job_view(job)
    if not include_result:
        v.pop("result", None)
    return v


def _result_or_404(job_id):
    job = JOBS.get(job_id)
    if job is None or job.status != SUCCEEDED:
        raise HTTPException(404, "no completed result for this job")
    return job.result


@api.get("/api/simulations/{job_id}/export.json")
def export_json(job_id: str):
    r = _result_or_404(job_id)
    return Response(json.dumps(r, indent=1), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="emg_{job_id}.json"'})


@api.get("/api/simulations/{job_id}/export.csv")
def export_csv(job_id: str):
    r = _result_or_404(job_id)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow([f"# emg_twin single-fibre result job={job_id} model_version={r['model_version']} "
                f"config_hash={r['config_hash']} units: time s, voltage V"])
    n_e, n_c = len(r["mono_V"]), len(r["diff_V"])
    pairs = r["electrodes"]["pairs"]
    w.writerow(["time_s"] + [f"mono_e{k+1}_V" for k in range(n_e)] +
               [f"diff_ch{j+1}_e{a+1}-e{b+1}_V" for j, (a, b) in enumerate(pairs)])
    for i, t in enumerate(r["time_s"]):
        w.writerow([repr(t)] + [repr(r["mono_V"][k][i]) for k in range(n_e)] +
                   [repr(r["diff_V"][j][i]) for j in range(n_c)])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="emg_{job_id}.csv"'})


@api.get("/api/simulations/{job_id}/config.json")
def export_config(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return Response(json.dumps(dict(model_version=cache.MODEL_VERSION, config=job.config.model_dump(mode="json")),
                               indent=1), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="config_{job_id}.json"'})


# --------------------------------------------------------------------------- verification
@api.get("/api/verification")
def list_verification():
    return dict(experiments=V.registry.describe())


@api.post("/api/verification/{name}")
def run_verification(name: str, params: dict | None = None):
    try:
        return V.run_experiment(name, params or {})
    except KeyError:
        raise HTTPException(404, f"unknown experiment {name}")
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e))


@api.post("/api/cache/clear")
def clear_cache(disk: bool = False):
    cache.clear_all(disk=disk)
    return dict(cleared=True, disk=disk)


# --------------------------------------------------------------------------- frontend
DIST = os.path.join(C.ROOT, "frontend", "dist")
if os.path.isdir(DIST):
    api.mount("/assets", StaticFiles(directory=os.path.join(DIST, "assets")), name="assets")

    @api.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = os.path.join(DIST, path)
        if path and os.path.isfile(f):
            return FileResponse(f)
        return FileResponse(os.path.join(DIST, "index.html"))
else:
    @api.get("/", include_in_schema=False)
    def root():
        return JSONResponse(dict(message="API running. Build the frontend (cd frontend && npm run build) or "
                                         "run the Vite dev server on :5173.", docs="/docs"))
