"""Physical changes reach the numerical model, and caches are invalidated (or
reused) exactly as the dependency analysis says.

Sensitivity is checked as "the signal changes by more than X", never as a
direction: not every physical change must change amplitude monotonically.
"""
import numpy as np
import pytest

from app import cache, pipeline
from app import params as P
from tests.conftest import coarse

FIBRE_CH = 8   # differential channel over the propagating part of the default fibre


def _rel_change(a, b):
    a, b = np.array(a), np.array(b)
    return np.abs(a - b).max() / np.abs(a).max()


@pytest.fixture(scope="module")
def base():
    return pipeline.run(coarse())


def test_conductivity_change_reaches_fem(base):
    cfg = coarse(**{"conductivity.skin": 0.5})
    assert pipeline.mesh_key(cfg) == pipeline.mesh_key(coarse())
    assert pipeline.stiffness_key(cfg) != pipeline.stiffness_key(coarse())
    r = pipeline.run(cfg)
    assert r["cache"]["mesh"] == "reused" and r["cache"]["stiffness"] == "assembled"
    assert r["cache"]["adjoint_rows_solved"] == 16
    assert _rel_change(base["diff_V"], r["diff_V"]) > 0.01
    assert r["config"]["conductivity"]["skin"] == 0.5


def test_anisotropy_change_reaches_fem(base):
    r = pipeline.run(coarse(**{"conductivity.muscle_longitudinal": 0.3}))
    assert r["cache"]["stiffness"] == "assembled"
    assert _rel_change(base["diff_V"], r["diff_V"]) > 0.01


def test_geometry_change_remeshes(base):
    cfg = coarse(**{"geometry.fat_thickness": 6e-3})
    assert pipeline.mesh_key(cfg) != pipeline.mesh_key(coarse())
    r = pipeline.run(cfg)
    assert r["cache"]["mesh"] == "built"
    assert r["solver"]["n_vertices"] > base["solver"]["n_vertices"]
    # electrodes moved outwards with the skin
    r_el = np.hypot(*np.array(r["electrodes"]["positions_m"])[:, :2].T)
    r_b = np.hypot(*np.array(base["electrodes"]["positions_m"])[:, :2].T)
    assert (r_el > r_b + 2e-3).all()
    assert _rel_change(base["diff_V"], r["diff_V"]) > 0.05


def test_signal_only_change_reuses_volume_conductor(base):
    pipeline.run(coarse())          # make sure the base A is cached (LRU may have evicted it)
    r = pipeline.run(coarse(**{"fibre.cv": 3.0}))
    assert r["cache"] == dict(mesh="reused", stiffness="reused", adjoint_rows_reused=16, adjoint_rows_solved=0,
                              basis_points="recomputed", signal="recomputed")
    assert _rel_change(base["diff_V"], r["diff_V"]) > 0.05
    assert r["conduction_velocity"]["configured_m_s"] == 3.0
    est = r["conduction_velocity"]["estimated_m_s"]
    assert est is not None and est == pytest.approx(3.0, rel=0.1)
    assert P.recompute_level(["fibre.cv"]) == "signal"


def test_fibre_depth_change_reuses_rows(base):
    r = pipeline.run(coarse(**{"fibre.depth": 6e-3}))
    assert r["cache"]["adjoint_rows_solved"] == 0
    assert r["fibre"]["x_m"] == pytest.approx(0.020 - 6e-3)
    assert _rel_change(base["diff_V"], r["diff_V"]) > 0.05


def test_moving_one_electrode_solves_one_row(base):
    cfg = coarse(**{"electrodes.overrides": {"5": {"theta": 0.3, "z": 0.004}}})
    r = pipeline.run(cfg)
    assert r["cache"]["adjoint_rows_solved"] == 1 and r["cache"]["adjoint_rows_reused"] == 15
    assert r["cache"]["stiffness"] == "reused"
    mono, mono0 = np.array(r["mono_V"]), np.array(base["mono_V"])
    others = [k for k in range(16) if k != 5]
    assert np.array_equal(mono[others], mono0[others])          # same rows, same P -> identical
    assert _rel_change(mono0[5], mono[5]) > 0.01
    pos = np.array(r["electrodes"]["positions_m"])[5]
    assert np.arctan2(pos[1], pos[0]) == pytest.approx(0.3, abs=1e-9) and pos[2] == pytest.approx(0.004)
    assert r["electrodes"]["tissue"][5] == "skin"


def test_array_spacing_and_orientation(base):
    r = pipeline.run(coarse(**{"electrodes.array.spacing": 4e-3, "electrodes.array.orientation": 0.5}))
    pos = np.array(r["electrodes"]["positions_m"])
    th = np.arctan2(pos[:, 1], pos[:, 0])
    assert np.all(np.diff(pos[:, 2]) > 0) and np.all(np.diff(th) > 0)   # helical line on the skin
    assert np.diff(pos[:, 2]) == pytest.approx(4e-3 * np.cos(0.5))
    assert all(t == "skin" for t in r["electrodes"]["tissue"])


def test_cg_tolerance_is_part_of_row_key():
    pos = np.zeros((1, 3))
    assert pipeline.electrode_row_keys(coarse(), pos) != pipeline.electrode_row_keys(
        coarse(**{"solver.cg_rtol": 1e-10}), pos)


def test_keys_include_model_version(monkeypatch):
    k = pipeline.mesh_key(coarse())
    monkeypatch.setattr(cache, "MODEL_VERSION", "something-else")
    assert pipeline.mesh_key(coarse()) != k


def test_recompute_levels():
    assert P.recompute_level(["geometry.fat_thickness", "fibre.cv"]) == "mesh"
    assert P.recompute_level(["conductivity.fat"]) == "stiffness"
    assert P.recompute_level(["electrodes.overrides.3.theta"]) == "adjoint"
    assert P.recompute_level(["fibre.theta"]) == "basis"
