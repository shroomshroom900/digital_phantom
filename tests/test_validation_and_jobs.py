"""Invalid inputs, failed simulations, stale responses and the HTTP API."""
import math
import threading
import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app import jobs, pipeline
from app.schemas import SimConfig
from tests.conftest import coarse


# ----------------------------------------------------------------------------- schema-level
@pytest.mark.parametrize("key,value", [
    ("conductivity.skin", -1.0), ("conductivity.fat", 0.0), ("conductivity.bone", float("nan")),
    ("fibre.cv", float("inf")), ("geometry.bone_radius", 0.03), ("timing.fs", 10.0),
    ("electrodes.array.n", 1), ("electrodes.overrides", {"20": {"theta": 0, "z": 0}}),
])
def test_schema_rejects(key, value):
    with pytest.raises(ValidationError):
        coarse(**{key: value})


# ----------------------------------------------------------------------------- physics-level
@pytest.mark.parametrize("key,value,fragment", [
    ("geometry.skin_thickness", 1e-3, "thinner than one mesh cell"),
    ("fibre.depth", 14e-3, "outside the muscle"),
    ("fibre.l1", 0.09, "stored fibre"),
    ("electrodes.array.z_center", 0.03, "beyond the usable skin"),
    ("electrodes.overrides", {"3": {"theta": 0.0, "z": -0.0175}, "4": {"theta": 0.0, "z": -0.0175}},
     "apart"),
    ("mesh.h_xy", 0.5e-3, "vertices"),
])
def test_physical_validation(key, value, fragment):
    cfg = coarse(**{key: value}) if key != "mesh.h_xy" else coarse(**{key: value, "mesh.h_z": 1e-3,
                                                                       "geometry.skin_thickness": 2e-3})
    errors, _ = pipeline.static_checks(cfg)
    assert any(fragment in e for e in errors), errors
    with pytest.raises(pipeline.SimulationError) as e:
        pipeline.run(cfg)
    assert fragment in str(e.value)


def test_fibre_must_be_in_muscle_cells():
    """0.2 mm below the nominal muscle surface is inside the muscle annulus, but at
    some angles the 2 mm staircase puts those points in fat cells: must be rejected."""
    rejected = None
    for th in [i * math.pi / 90 for i in range(46)]:
        cfg = coarse(**{"fibre.depth": 0.2e-3, "fibre.theta": th})
        assert not pipeline.static_checks(cfg)[0]
        p = pipeline.preview(cfg)
        if not p["valid"]:
            rejected = cfg
            assert any("outside muscle cells" in e for e in p["errors"])
            break
    assert rejected is not None
    with pytest.raises(pipeline.SimulationError, match="outside muscle cells"):
        pipeline.run(rejected)


def test_numerical_failure_is_reported(monkeypatch):
    import fem

    def boom(*a, **k):
        raise RuntimeError("CG did not converge (info=5000) in 5000 iterations")
    monkeypatch.setattr(fem, "adjoint_K", boom)
    with pytest.raises(pipeline.SimulationError) as e:
        pipeline.run(coarse(**{"conductivity.skin": 0.77}))       # new A -> needs fresh solves
    assert e.value.stage == "adjoint" and "did not converge" in str(e.value)


# ----------------------------------------------------------------------------- jobs
def test_supersede_and_order():
    gate = threading.Event()
    seen = []

    def runner(cfg, progress):
        gate.wait(5)
        seen.append(cfg.fibre.cv)
        return dict(cv=cfg.fibre.cv)
    jm = jobs.JobManager(runner=runner)
    a = jm.submit(coarse(**{"fibre.cv": 3.0}))
    time.sleep(0.1)                                   # a is now running (blocked on the gate)
    b = jm.submit(coarse(**{"fibre.cv": 3.5}))
    c = jm.submit(coarse(**{"fibre.cv": 4.5}))        # b still queued -> superseded
    assert b.status == jobs.SUPERSEDED
    gate.set()
    jm.wait(c.id); jm.wait(a.id)
    assert a.status == c.status == jobs.SUCCEEDED
    assert seen == [3.0, 4.5]                         # b never ran
    assert a.seq < b.seq < c.seq and jm.latest_seq() == c.seq
    assert a.result["seq"] < c.result["seq"]          # client keeps the higher seq


def test_failed_job_has_message():
    def runner(cfg, progress):
        raise pipeline.SimulationError("electrode 3 does not land on skin", stage="validation")
    jm = jobs.JobManager(runner=runner)
    j = jm.wait(jm.submit(coarse()).id)
    assert j.status == jobs.FAILED and "land on skin" in j.error["message"] and j.result is None


def test_job_config_is_immutable():
    cfg = coarse()
    with pytest.raises(ValidationError):
        cfg.fibre.cv = 5.0


# ----------------------------------------------------------------------------- HTTP
@pytest.fixture(scope="module")
def client():
    from app.server import api
    return TestClient(api)


def test_api_workflow(client):
    params = client.get("/api/parameters").json()
    assert {p["provenance"] for p in params["parameters"]} <= {"[PAPER]", "[PROMPT]", "[ASSUMED]"}
    assert all(p["unit"] for p in params["parameters"]) and params["unsupported"]
    cfg = coarse().model_dump(mode="json")
    pv = client.post("/api/preview", json=cfg).json()
    assert pv["valid"], pv["errors"]
    g = client.get(f"/api/geometry/{pv['mesh_key']}").json()
    assert [t["key"] for t in g["tissues"]] == ["skin", "fat", "muscle", "bone"]
    assert all(t["in_fem"] for t in g["tissues"]) and not any(i["in_fem"] for i in g["illustrations"])

    job = client.post("/api/simulations", json=dict(config=cfg, label="api test")).json()
    for _ in range(600):
        j = client.get(f"/api/simulations/{job['id']}").json()
        if j["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.1)
    assert j["status"] == "succeeded", j
    r = j["result"]
    assert r["config"] == cfg and r["label"] == "api test"
    csv_text = client.get(f"/api/simulations/{job['id']}/export.csv").text
    lines = csv_text.splitlines()
    assert lines[1].startswith("time_s,mono_e1_V") and len(lines) == 2 + len(r["time_s"])
    assert client.get(f"/api/simulations/{job['id']}/export.json").json()["config_hash"] == r["config_hash"]


def test_api_rejects_invalid(client):
    cfg = coarse().model_dump(mode="json")
    cfg["conductivity"]["fat"] = -0.1
    assert client.post("/api/simulations", json=dict(config=cfg)).status_code == 422
    cfg = coarse().model_dump(mode="json")
    cfg["fibre"]["depth"] = 0.019
    pv = client.post("/api/preview", json=cfg).json()
    assert not pv["valid"] and pv["errors"]
    job = client.post("/api/simulations", json=dict(config=cfg)).json()
    for _ in range(100):
        j = client.get(f"/api/simulations/{job['id']}").json()
        if j["status"] == "failed":
            break
        time.sleep(0.05)
    assert j["status"] == "failed" and "outside the muscle" in j["error"]["message"]
    assert client.post("/api/simulations/future", json=dict(kind="motor_unit_pool")).status_code == 501


def test_config_diff(client):
    a = coarse().model_dump(mode="json")
    b = coarse(**{"fibre.cv": 3.3, "conductivity.fat": 0.05}).model_dump(mode="json")
    d = client.post("/api/config/diff", json=dict(a=a, b=b)).json()
    assert {c["key"] for c in d["changes"]} == {"fibre.cv", "conductivity.fat"} and d["recompute"] == "stiffness"
