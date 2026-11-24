"""
stage5_timing.py -- experiment C: where does the time go, and what can be reused?
Run:  python3 stage5_timing.py        (about 4-5 minutes)

Idea (paper, "Implementation remarks"): every pipeline step depends only on the output
of earlier steps.  So when one parameter changes, we redo that step and the steps AFTER
it, and take everything before it from memory.  Here we do exactly that, for real:
for each kind of change we (i) run the SHORT path with cached intermediates,
(ii) run the FULL pipeline from scratch with the same new parameters, and check that
both give the SAME sEMG while the short path is much faster.
"""
import gc
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

import config as C
import fem
import fibre
import muscle as M
from mesh import build_cylinder_mesh

FS, T_RAMP = C.FS, 2.0
n_samp = int(T_RAMP * FS)
t_muap = np.arange(int(0.030 * FS)) / FS
t_grid = np.arange(n_samp) / FS
geom = fibre.FibreGeometry(0.0, 0.0, dz=M.MUSCLE_DZ)

# every random ingredient gets its OWN seed, so changing one leaves the others alone
BASE = dict(ring_z=M.RING_Z, geom_seed=1, mu_seed=2, prop_seed=3, spike_seed=4, peak=100.0)

STEPS = ["mesh", "A", "forward", "place", "mu", "basis_pts", "response", "muap", "spikes", "semg"]
LABEL = {"mesh": "mesh", "A": "assemble A", "forward": "general forward solution\n(8 CG solves)",
         "place": "place fibres", "mu": "MU distribution", "basis_pts": "fibres basis points",
         "response": "fibres EMG response", "muap": "MUAPs assembling",
         "spikes": "recruitment + firing", "semg": "raw sEMG assembling"}
# which steps are taken from memory (reused) for each kind of change
SCENARIOS = {
    "new arm / electrodes\n(everything changes)": set(),
    "new fibre positions": {"mesh", "A", "forward", "spikes"},
    "new fibre properties\n(velocity, NMJ, tendons)": {"mesh", "A", "forward", "place", "mu", "basis_pts", "spikes"},
    "new MU distribution\n(sizes, territories)": {"mesh", "A", "forward", "place", "basis_pts", "response", "spikes"},
    "new excitation /\nfiring model": {"mesh", "A", "forward", "place", "mu", "basis_pts", "response", "muap"},
}
CHANGE = {  # the parameter change made in each scenario (applied to BASE)
    "new arm / electrodes\n(everything changes)": dict(ring_z=(10e-3, 30e-3)),
    "new fibre positions": dict(geom_seed=11),
    "new fibre properties\n(velocity, NMJ, tendons)": dict(prop_seed=13),
    "new MU distribution\n(sizes, territories)": dict(mu_seed=12),
    "new excitation /\nfiring model": dict(peak=60.0),
}


def run(cfg, cache=None, reuse=frozenset()):
    """Run the whole pipeline; steps named in `reuse` are taken from `cache`.
    Returns (outputs dict, seconds-per-step dict, wall-clock seconds)."""
    out, T = {}, {}
    t_all = time.perf_counter()

    def step(name, fn):
        if name in reuse:
            out[name], T[name] = cache[name], 0.0
        else:
            t0 = time.perf_counter()
            out[name] = fn()
            T[name] = time.perf_counter() - t0

    step("mesh", build_cylinder_mesh)
    step("A", lambda: fem.assemble_stiffness(out["mesh"]))

    def forward():
        M.RING_Z = cfg["ring_z"]                              # electrode heights
        pos = M.ring_electrodes(out["mesh"])
        K, basis = M.bipolar_adjoint(out["mesh"], out["A"], pos)
        return basis
    step("forward", forward)

    def place():
        uv = M.uniform_in_unit_circle(M.N_FIBRES, np.random.default_rng(cfg["geom_seed"]))
        return uv, M.place_fibres(uv)
    step("place", place)
    step("mu", lambda: M.make_motor_units(np.random.default_rng(cfg["mu_seed"]), uv=out["place"][0]))
    step("basis_pts", lambda: M.fibre_basis_points(out["forward"], out["place"][1], geom))

    def response():
        props = M.make_fibre_properties(np.random.default_rng(cfg["prop_seed"]), M.N_FIBRES)
        return M.fibre_responses(out["basis_pts"], geom, t_muap, props)
    step("response", response)
    step("muap", lambda: M.assemble_muaps(out["response"], out["mu"].mu_of, out["mu"].n_mu))

    def spikes():
        E = M.excitation(t_grid, T_RAMP, cfg["peak"])
        return M.spike_trains(E, FS, out["mu"].threshold, np.random.default_rng(cfg["spike_seed"]))
    step("spikes", spikes)
    step("semg", lambda: M.assemble_semg(out["muap"], out["spikes"], FS, n_samp))
    return out, T, time.perf_counter() - t_all


def nmse(x, ref):
    return np.sum((x - ref) ** 2) / np.sum(ref ** 2)


# ======================================================================
# PART 1: baseline timings (3 repeats, median of each step)
# ======================================================================
print("=== PART 1: time of every pipeline step (median of 3 runs) ===")
reps = []
for r in range(3):
    out, T, wall = run(BASE)
    reps.append(T)
    print(f"  run {r + 1}: {wall:.1f} s")
    if r < 2:
        del out; gc.collect()
base_out = out
Tb = {s: float(np.median([rp[s] for rp in reps])) for s in STEPS}
Tb_spread = {s: float(np.ptp([rp[s] for rp in reps])) for s in STEPS}
total_b = sum(Tb.values())
print(f"{'step':<30s} {'median (s)':>10s} {'spread (s)':>11s}")
for s in STEPS:
    print(f"{LABEL[s].replace(chr(10), ' '):<30s} {Tb[s]:10.2f} {Tb_spread[s]:11.2f}")
print(f"{'TOTAL':<30s} {total_b:10.2f}")
n_s = len(base_out["forward"].muscle_nodes)
t_solve = Tb["forward"] / 8
print(f"\none CG solve = {t_solve:.2f} s.  Brute force (one solve per muscle basis vertex, "
      f"n_s = {n_s:,}) would take {n_s * t_solve / 3600:.1f} h just for the forward solution,"
      f" vs {Tb['forward']:.1f} s with the adjoint method ({n_s * t_solve / Tb['forward']:,.0f}x).")

# ======================================================================
# PART 2: change one thing, redo only what depends on it (measured, not estimated)
# ======================================================================
print("\n=== PART 2: partial recomputation vs full recomputation (measured) ===")
print(f"{'what changed':<44s} {'short path':>10s} {'full run':>9s} {'speed-up':>9s}  {'NMSE(short, full)':>18s}")
meas = {}
for name, reuse in SCENARIOS.items():
    cfg = {**BASE, **CHANGE[name]}
    flat = name.replace("\n", " ")
    if reuse:
        o_part, T_part, w_part = run(cfg, cache=base_out, reuse=reuse)
        o_full, T_full, w_full = run(cfg)
        err = nmse(o_part["semg"], o_full["semg"])
        del o_part, o_full; gc.collect()
        meas[name] = (w_part, w_full, err, T_part, T_full)
        print(f"{flat:<44s} {w_part:9.2f}s {w_full:8.2f}s {w_full / w_part:8.1f}x  {err:18.1e}")
        assert err < 1e-12, "short path does not reproduce the full recomputation!"
    else:
        o_full, T_full, w_full = run(cfg)
        # a changed electrode height must actually change the signal:
        diff = nmse(o_full["semg"], base_out["semg"])
        del o_full; gc.collect()
        meas[name] = (w_full, w_full, 0.0, T_full, T_full)
        print(f"{flat:<44s} {w_full:9.2f}s {w_full:8.2f}s {1.0:8.1f}x  "
              f"(signal differs from baseline: NMSE {diff:.2f})")
print("all short paths reproduce the full recomputation exactly: PASSED")

# ======================================================================
# PART 3: scaling  (how would this behave with the paper's much bigger problem?)
# ======================================================================
print("\n=== PART 3: scaling ===")
# (a) one CG solve vs mesh size
print("one electrode, one CG solve, for finer and finer meshes:")
rows = []
for hxy, hz in ((4e-3, 8e-3), (2.5e-3, 5e-3), (2e-3, 4e-3), (1.4e-3, 2.8e-3), (1e-3, 2e-3), (0.7e-3, 1.4e-3)):
    mm = build_cylinder_mesh(hxy, hz)
    t0 = time.perf_counter(); Am = fem.assemble_stiffness(mm); ta = time.perf_counter() - t0
    Sm = fem.selection_matrix(mm, fem.skin_electrode_positions(mm, 0.0, np.array([0.0])))
    t0 = time.perf_counter()
    _, it = fem.cg_solve(Am, Sm[0].toarray().ravel())
    ts = time.perf_counter() - t0
    rows.append((len(mm.nodes), ta, ts, it))
    print(f"  h = {hxy * 1e3:3.1f}/{hz * 1e3:3.1f} mm: {len(mm.nodes):9,} vertices   assemble {ta:5.1f} s   "
          f"solve {ts:6.2f} s ({it} iterations)")
    del mm, Am, Sm; gc.collect()
rows = np.array(rows)
nv, ts_ = rows[:, 0], rows[:, 2]
slope, icpt = np.polyfit(np.log(nv[2:]), np.log(ts_[2:]), 1)
t_paper_mesh = float(np.exp(icpt) * 2.1e6 ** slope)
print(f"  fit over the 4 largest meshes: solve time ~ n_v^{slope:.2f}  ->  extrapolated to the paper's "
      f"2.1 M vertices: ~{t_paper_mesh:.0f} s per electrode (paper: 13 s/electrode, with FEniCS, "
      f"a different solver/preconditioner -- so only a loose comparison)")

# (b) number of fibres
Pf, geo_xy = base_out["basis_pts"], base_out["place"][1]
props = M.make_fibre_properties(np.random.default_rng(BASE["prop_seed"]), M.N_FIBRES)
print("\nfibre-dependent steps vs number of fibres:")
nf_list = [6250, 12500, 25000, 50000]
t_bp, t_rs = [], []
for nf in nf_list:
    t0 = time.perf_counter(); M.fibre_basis_points(base_out["forward"], geo_xy[:nf], geom); t_bp.append(time.perf_counter() - t0)
    sub = {k: v[:nf] for k, v in props.items()}
    t0 = time.perf_counter(); M.fibre_responses(Pf[:nf], geom, t_muap, sub); t_rs.append(time.perf_counter() - t0)
    print(f"  {nf:6d} fibres: basis points {t_bp[-1]:5.2f} s   EMG response {t_rs[-1]:5.2f} s")

# (c) number of electrodes (channels) in the forward solution
print("\ngeneral forward solution vs number of channels (adjoint: one solve per channel):")
mesh_b, A_b = base_out["mesh"], base_out["A"]
M.RING_Z = BASE["ring_z"]
pos = M.ring_electrodes(mesh_b)
n_ch_list, t_fw = [1, 2, 4, 8], []
for nc in n_ch_list:
    t0 = time.perf_counter(); M.bipolar_adjoint(mesh_b, A_b, pos[:nc]); t_fw.append(time.perf_counter() - t0)
    print(f"  {nc} channels: {t_fw[-1]:5.1f} s")

# ======================================================================
# FIGURE  (drawn by stage5_plot.py from the saved results)
# ======================================================================
import pickle
import stage5_plot
R = dict(Tb=Tb, LABEL=LABEL, SCENARIOS=SCENARIOS, meas=meas, nv=nv, ts=ts_, slope=slope, icpt=icpt,
         n_vertices_main=len(base_out["mesh"].nodes), nf_list=nf_list, t_bp=t_bp, t_rs=t_rs,
         n_ch_list=n_ch_list, t_fw=t_fw, n_s=n_s, t_solve=t_solve)
with open(os.path.join(C.CACHE_DIR, "cache_stage5.pkl"), "wb") as f:
    pickle.dump(R, f)
stage5_plot.make_figure(R, os.path.join(C.FIG_DIR, "stage5_timing.png"))
