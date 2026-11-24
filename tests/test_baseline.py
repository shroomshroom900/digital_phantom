"""The app must reproduce the stage 2 + stage 3 scripts with default settings.

Tolerances:
  * same machine, same code path -> identical floating-point operations, so we
    require bit-for-bit equality of electrode positions, time axis and signals
    (asserted as max relative difference <= 1e-12 to allow a BLAS change);
  * against cache/cache_linear_array.npz committed from another machine: CG with
    rtol 1e-8 bounds the lead-field error to ~1e-8 relative; we measured 1e-14,
    and require 1e-8.
"""
import os

import numpy as np
import pytest

import config as C
import fem
import fibre
from mesh import build_cylinder_mesh
from app import pipeline
from app.schemas import SimConfig


@pytest.fixture(scope="module")
def default_run():
    return pipeline.run(SimConfig())


@pytest.fixture(scope="module")
def stage3_reference():
    """Literally the stage2_forward.py / stage3_single_fibre.py sequence."""
    m = build_cylinder_mesh()
    A = fem.assemble_stiffness(m)
    z = (np.arange(C.N_ELEC_LINEAR) - (C.N_ELEC_LINEAR - 1) / 2) * C.ELEC_SPACING
    elec = fem.skin_electrode_positions(m, 0.0, z)
    K, _ = fem.adjoint_K(A, fem.selection_matrix(m, elec))
    geom = fibre.FibreGeometry(x=C.R_MUSCLE - C.FIBRE_DEPTH, y=0.0)
    t = np.arange(int(0.030 * C.FS_SINGLE)) / C.FS_SINGLE
    mono, _ = fibre.simulate(fibre.precompute_basis_points(fem.BasisSolutions(m, K), geom), geom, t,
                             fibre.FibreParams())
    return dict(m=m, elec=elec, K=K, t=t, mono=mono, geom=geom)


def test_default_matches_stage3(default_run, stage3_reference):
    r, ref = default_run, stage3_reference
    assert np.array_equal(np.array(r["electrodes"]["positions_m"]), ref["elec"])
    assert np.array_equal(np.array(r["time_s"]), ref["t"])
    mono = np.array(r["mono_V"])
    assert np.abs(mono - ref["mono"]).max() / np.abs(ref["mono"]).max() <= 1e-12
    diff = np.array(r["diff_V"])
    assert np.allclose(diff, ref["mono"][:-1] - ref["mono"][1:], rtol=0, atol=1e-18)


def test_default_matches_committed_cache(default_run, stage3_reference):
    path = os.path.join(C.CACHE_DIR, "cache_linear_array.npz")
    if not os.path.exists(path):
        pytest.skip("committed cache not present")
    with np.load(path) as f:
        K_ref, elec_ref = f["K"], f["elec"]
    assert np.array_equal(elec_ref, stage3_reference["elec"])
    ref = stage3_reference
    mono_ref, _ = fibre.simulate(fibre.precompute_basis_points(fem.BasisSolutions(ref["m"], K_ref), ref["geom"]),
                                 ref["geom"], ref["t"], fibre.FibreParams())
    mono = np.array(default_run["mono_V"])
    assert np.abs(mono - mono_ref).max() / np.abs(mono_ref).max() <= 1e-8


def test_stage3_published_numbers(default_run):
    """README / stage 3 printout: NMJ channel ~0.087 of its neighbours; velocity
    estimate 3.95 m/s from channels 8-10, 8-11, 7-10 (input 4.0)."""
    r = default_run
    p2p = np.array(r["metrics"]["diff_p2p_V"])
    assert p2p[3] / p2p[[2, 4]].mean() == pytest.approx(0.087, abs=0.002)
    t, diff = np.array(r["time_s"]), np.array(r["diff_V"])
    v = np.mean([(b - a) * C.ELEC_SPACING / pipeline._xcorr_lag(t, diff[a - 1], diff[b - 1])
                 for a, b in ((8, 10), (8, 11), (7, 10))])
    assert v == pytest.approx(3.95, abs=0.01)
    est = r["conduction_velocity"]
    assert est["configured_m_s"] == 4.0
    assert est["estimated_m_s"] == pytest.approx(4.0, rel=0.05)


def test_solver_diagnostics(default_run):
    s = default_run["solver"]
    assert s["max_rel_residual"] <= 100 * s["cg_rtol"]
    assert s["current_conservation"] < 1e-12
    assert len(s["rows"]) == 16 and all(r["iterations"] > 0 for r in s["rows"])
    assert default_run["model_version"] and default_run["config_hash"]
    assert SimConfig.model_validate(default_run["config"]) == SimConfig()
