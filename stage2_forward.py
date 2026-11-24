"""
stage2_forward.py -- forward solver with the adjoint trick + validation (exp. D).
Run:  python3 stage2_forward.py
Takes ~1 minute on a laptop.
"""
import os
import time
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import fem
from mesh import build_cylinder_mesh

rng = np.random.default_rng(C.RNG_SEED)


def electrode_line(m):
    z = (np.arange(C.N_ELEC_LINEAR) - (C.N_ELEC_LINEAR - 1) / 2) * C.ELEC_SPACING
    return fem.skin_electrode_positions(m, 0.0, z)


def nmse(x_num, x_ref):
    """Normalised mean square error, as defined in the brief."""
    return np.sum((x_num - x_ref) ** 2) / np.sum(x_ref ** 2)


# ======================================================================
# PART 1: the full-resolution mesh -- adjoint solve for 16 electrodes
# ======================================================================
print("=== Fine mesh: adjoint method ===")
m = build_cylinder_mesh()
t = time.perf_counter(); A = fem.assemble_stiffness(m); t_asm = time.perf_counter() - t
print(f"assemble A ({A.shape[0]:,} x {A.shape[0]:,}, {A.nnz:,} nonzeros): {t_asm:.2f} s")
print(f"  symmetry error max|A-A^T| = {abs(A - A.T).max():.1e};  max|A 1| = "
      f"{abs(A @ np.ones(A.shape[0])).max():.1e}  (should be ~0: constants are in the null space)")

elec = electrode_line(m)
S = fem.selection_matrix(m, elec)
t = time.perf_counter(); K, iters = fem.adjoint_K(A, S); t_adj = time.perf_counter() - t
print(f"adjoint: {S.shape[0]} CG solves, {t_adj:.1f} s total ({t_adj/S.shape[0]:.2f} s each, "
      f"{int(np.mean(iters))} CG iterations on average)")

basis = fem.BasisSolutions(m, K)
n_s = len(basis.muscle_nodes)
t_one = t_adj / S.shape[0]
print(f"basis sources n_s = {n_s:,} (muscle vertices);  electrodes n_e = {S.shape[0]}")
print(f"  brute force would need {n_s:,} solves  ~ {n_s*t_one/3600:.1f} hours;"
      f" adjoint needed {S.shape[0]} solves = {t_adj:.0f} s"
      f"  -> {n_s/S.shape[0]:.0f}x fewer solves")

# Spot-check on the FINE mesh: 3 random basis vertices solved the slow way.
print("  spot check (3 random sources, brute force on the fine mesh):")
for j in rng.choice(n_s, 3, replace=False):
    b = np.zeros(A.shape[0]); b[basis.muscle_nodes[j]] = 1.0     # unit source at vertex j
    x, _ = fem.cg_solve(A, b, rtol=1e-10)
    print(f"    source {j:6d}: NMSE(adjoint column vs direct solve) = "
          f"{nmse(basis.V[:, j], S @ x):.2e}")

# ======================================================================
# PART 2: EXPERIMENT D -- validation on a coarse mesh
# ======================================================================
print("\n=== Experiment D: adjoint vs brute force on a coarse mesh ===")
mc = build_cylinder_mesh(C.H_XY_COARSE, C.H_Z_COARSE)
Ac = fem.assemble_stiffness(mc)
Sc = fem.selection_matrix(mc, electrode_line(mc))
print(f"coarse mesh: {len(mc.nodes):,} vertices, {len(mc.tets):,} tets")
Kc, _ = fem.adjoint_K(Ac, Sc, rtol=1e-12)
bc = fem.BasisSolutions(mc, Kc)
ns_c = len(bc.muscle_nodes)

# (i) brute force with the SAME iterative solver: one solve per basis source
t = time.perf_counter()
V_bf = np.empty((Sc.shape[0], ns_c))
for col, node in enumerate(bc.muscle_nodes):
    b = np.zeros(Ac.shape[0]); b[node] = 1.0
    x, _ = fem.cg_solve(Ac, b, rtol=1e-12)
    V_bf[:, col] = Sc @ x
t_bf = time.perf_counter() - t
# (ii) brute force with a completely independent method: dense pseudo-inverse
#      V_ref = S A^+ B   (A^+ = Moore-Penrose inverse, handles the null space)
t = time.perf_counter()
Apinv = np.linalg.pinv(Ac.toarray(), hermitian=True)
V_pinv = (Sc @ Apinv)[:, bc.muscle_nodes]
t_pinv = time.perf_counter() - t
print(f"  n_s = {ns_c} brute-force CG solves: {t_bf:.1f} s   |   dense pinv: {t_pinv:.1f} s")
print(f"  NMSE(adjoint, brute-force CG)   = {nmse(bc.V, V_bf):.2e}")
print(f"  NMSE(adjoint, dense pinv)       = {nmse(bc.V, V_pinv):.2e}")
print(f"  NMSE(brute-force CG, dense pinv)= {nmse(V_bf, V_pinv):.2e}  (checks the solver itself)")

# (iii) check Eq. (5): a source at an ARBITRARY point = barycentric blend of
#       the 4 corner solutions.  Brute force: put the 4 weights into b.
pts = []
while len(pts) < 5:                                   # random points inside muscle
    p = np.array([rng.uniform(-0.015, 0.015), rng.uniform(-0.015, 0.015), rng.uniform(-0.03, 0.03)])
    if 0.009 < np.hypot(p[0], p[1]) < 0.018:
        pts.append(p)
pts = np.array(pts)
V_interp = bc.point_potentials(pts)                   # Eq. (5)
tet, lam, _ = fem.locate_points(mc, pts)
V_direct = np.empty_like(V_interp)
for n in range(len(pts)):
    b = np.zeros(Ac.shape[0])
    b[mc.tets[tet[n]]] = lam[n]                       # b = B lambda, for ONE tet
    V_direct[:, n] = Sc @ (Apinv @ (b - b.mean()))
print(f"  NMSE(Eq.5 barycentric blend, direct source at the point) = {nmse(V_interp, V_direct):.2e}")
assert nmse(bc.V, V_pinv) < 1e-8 and nmse(V_interp, V_direct) < 1e-8
print("  PASSED")

# ======================================================================
# PART 3: pictures
# ======================================================================
def to_grid(m, values):
    """Scatter per-vertex values onto the regular (x, y, z) grid (NaN outside)."""
    ix = np.rint((m.nodes[:, 0] - m.xs[0]) / m.h_xy).astype(int)
    iy = np.rint((m.nodes[:, 1] - m.ys[0]) / m.h_xy).astype(int)
    iz = np.rint((m.nodes[:, 2] - m.zs[0]) / m.h_z).astype(int)
    g = np.full((m.n_xy + 1, m.n_xy + 1, m.n_z + 1), np.nan)
    g[ix, iy, iz] = values
    return g


e0 = 7                                    # show the lead field of electrode 8 (index 7)
lead = to_grid(m, K[e0])                  # volts per ampere  (= ohm)
kz = int(np.rint((elec[e0, 2] - m.zs[0]) / m.h_z))    # z-slice of that electrode
jy = m.n_xy // 2
vmax = np.nanpercentile(lead[:, :, kz], 99)
vmin = np.nanpercentile(lead, 1)
levels = np.linspace(vmin, vmax, 25)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
a = ax[0]
cs = a.contourf(m.xs * 1e2, m.ys * 1e2, lead[:, :, kz].T, levels=levels, cmap="magma", extend="both")
th = np.linspace(0, 2 * np.pi, 200)
for r in (C.R_BONE, C.R_MUSCLE, C.R_FAT, C.R_SKIN):
    a.plot(r * 1e2 * np.cos(th), r * 1e2 * np.sin(th), "w-", lw=0.6)
a.plot(elec[e0, 0] * 1e2, elec[e0, 1] * 1e2, "co", ms=6)
a.set_aspect("equal"); a.set_xlabel("x (cm)"); a.set_ylabel("y (cm)")
a.set_title("(a) Lead field of one electrode (cross-section)")
fig.colorbar(cs, ax=a, label="V per A of source current (Ω)")

a = ax[1]
cs = a.contourf(m.zs * 1e2, m.xs * 1e2, lead[:, jy, :], levels=levels, cmap="magma", extend="both")
for r in (C.R_BONE, C.R_MUSCLE, C.R_FAT):
    a.axhline(r * 1e2, color="w", lw=0.6); a.axhline(-r * 1e2, color="w", lw=0.6)
a.plot(elec[:, 2] * 1e2, elec[:, 0] * 1e2, "c.", ms=5)
a.set_aspect("equal"); a.set_xlabel("z (cm)"); a.set_ylabel("x (cm)")
a.set_title("(b) Same, lengthwise slice (y≈0)")

a = ax[2]
# a unit source in the muscle 3 mm below the muscle surface, above the array, at z = 0
src = np.array([[C.R_MUSCLE - 3e-3, 0.0, 0.0], [C.R_MUSCLE - 3e-3, 0.0, 1.5e-2],
                [C.R_MUSCLE - 10e-3, 0.0, 0.0]])
vp = basis.point_potentials(src)
labels = ["3 mm below muscle surface, z=0", "same depth, z=1.5 cm", "10 mm below muscle surface, z=0"]
for n in range(3):
    a.plot(elec[:, 2] * 1e2, vp[:, n], "o-", ms=4, label=labels[n])
a.set_xlabel("electrode position z (cm)"); a.set_ylabel("voltage at electrode (V per A)")
a.set_title("(c) What the 16 electrodes see from one unit source")
a.legend(fontsize=7); a.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(C.FIG_DIR, "stage2_forward.png"), dpi=130, bbox_inches="tight")
print("saved stage2_forward.png")

np.savez_compressed(os.path.join(C.CACHE_DIR, "cache_linear_array.npz"), K=K, elec=elec)   # reused by later stages
