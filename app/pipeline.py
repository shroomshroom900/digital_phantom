"""
Single-fibre pipeline = stage2_forward.py + stage3_single_fibre.py, driven by a
SimConfig instead of the config.py globals.

Every numerical step calls the original functions:
    mesh.build_cylinder_mesh   fem.assemble_stiffness   fem.skin_electrode_positions
    fem.selection_matrix       fem.adjoint_K (-> fem.cg_solve)   fem.BasisSolutions
    fibre.FibreGeometry        fibre.precompute_basis_points     fibre.simulate
so boundary conditions (insulating Neumann), source balancing (mean removal),
the zero-mean voltage reference, the CG tolerance, the adjoint formulation and
point-electrode sampling are exactly those of the scripts.  This module only adds
validation, caching and convergence checks.
"""
import copy
import math
import time

import numpy as np

import config as C
import fem
import fibre
from mesh import build_cylinder_mesh

from app import cache
from app.schemas import SimConfig

MAX_VERTICES = 400_000
FIBRE_END_MARGIN = 5e-3        # m, stored fibre stops this far from the end caps [ASSUMED]
MIN_PAIR_DISTANCE = 1e-3       # m, smallest allowed distance between paired electrodes [ASSUMED]
RESIDUAL_SLACK = 100.0         # accept a recomputed CG residual up to 100 x rtol
_MESHES_USED_BY_RUNS = set()   # mesh keys a completed run has used (for honest cache reporting)


class SimulationError(Exception):
    """A failure the user can understand and fix (bad input or numerical failure)."""

    def __init__(self, message, stage="validation", details=None):
        super().__init__(message)
        self.stage, self.details = stage, details or {}


# ============================================================================ validation
def fibre_z_half(cfg: SimConfig):
    return min(C.FIBRE_Z_RANGE, cfg.geometry.length / 2 - FIBRE_END_MARGIN)


def fibre_xy(cfg: SimConfig):
    r = cfg.geometry.muscle_radius - cfg.fibre.depth
    th = cfg.fibre.theta
    return r * math.cos(th), r * math.sin(th)


def estimate_vertices(cfg: SimConfig):
    r_skin = cfg.geometry.radii()[3]
    return int(math.pi * r_skin ** 2 / cfg.mesh.h_xy ** 2 * (cfg.geometry.length / cfg.mesh.h_z + 1))


def static_checks(cfg: SimConfig):
    """Checks that need no mesh.  Returns (errors, warnings) as lists of str."""
    errors, warnings = [], []
    g, h = cfg.geometry, cfg.mesh
    r_bone, r_muscle, r_fat, r_skin = g.radii()
    for name, thick in (("skin", g.skin_thickness), ("fat", g.fat_thickness),
                        ("muscle", r_muscle - r_bone), ("bone radius", r_bone)):
        if thick < h.h_xy - 1e-12:
            errors.append(f"{name} ({thick*1e3:.2f} mm) is thinner than one mesh cell (h_xy = {h.h_xy*1e3:.2f} mm); "
                          f"the staircase mesh would lose that tissue in places")
        elif thick < 2 * h.h_xy - 1e-12 and name in ("fat", "muscle"):
            warnings.append(f"{name} is less than two mesh cells thick; its staircase boundary is coarse")
    nv = estimate_vertices(cfg)
    if nv > MAX_VERTICES:
        errors.append(f"mesh would have ~{nv:,} vertices (limit {MAX_VERTICES:,}); increase h_xy / h_z or shrink the arm")

    # fibre inside its muscle and inside the stored fibre geometry
    f = cfg.fibre
    r_f = r_muscle - f.depth
    if not r_bone < r_f < r_muscle:
        errors.append(f"fibre at radius {r_f*1e3:.2f} mm is outside the muscle "
                      f"({r_bone*1e3:.1f}-{r_muscle*1e3:.1f} mm); reduce the depth")
    zh = fibre_z_half(cfg)
    n_cells = int(round(2 * zh / f.dz))
    z_lo, z_hi = -zh, -zh + f.dz * n_cells
    if f.nmj_z - f.l2 < z_lo - 1e-12 or f.nmj_z + f.l1 > z_hi + 1e-12:
        errors.append(f"active fibre [{(f.nmj_z-f.l2)*1e3:.1f}, {(f.nmj_z+f.l1)*1e3:.1f}] mm must lie inside the "
                      f"stored fibre [{z_lo*1e3:.1f}, {z_hi*1e3:.1f}] mm (tendon ends too far out)")
    if f.taper > min(f.l1, f.l2) / 2:
        warnings.append("taper is longer than half a semi-fibre; the Tukey window never reaches 1")
    if f.dz > h.h_z:
        warnings.append("fibre cells are longer than the mesh z spacing; Eq. (10) under-samples the lead field")

    # electrodes on the lateral skin, away from the end caps, with valid pair spacing
    pts = cfg.electrodes.surface_points(r_skin)
    z_max = g.length / 2 - h.h_z
    for k, (th, z) in enumerate(pts):
        if abs(z) > z_max:
            errors.append(f"electrode {k+1} at z = {z*1e3:.1f} mm is beyond the usable skin "
                          f"(|z| <= {z_max*1e3:.1f} mm; the end caps are not skin)")
    xyz = np.array([[r_skin * math.cos(th), r_skin * math.sin(th), z] for th, z in pts])
    for a, b in cfg.electrodes.pairs():
        d = float(np.linalg.norm(xyz[a] - xyz[b]))
        if d < MIN_PAIR_DISTANCE:
            errors.append(f"differential pair e{a+1}-e{b+1} is only {d*1e3:.2f} mm apart "
                          f"(minimum {MIN_PAIR_DISTANCE*1e3:.1f} mm)")
        elif d < h.h_z:
            warnings.append(f"pair e{a+1}-e{b+1} is closer ({d*1e3:.2f} mm) than the mesh z spacing; "
                            f"the P1 field cannot resolve it well")
    return errors, warnings


# ============================================================================ cached steps
def mesh_key(cfg: SimConfig):
    g = cfg.geometry
    return cache.make_key("mesh", g.radii(), g.length, cfg.mesh.h_xy, cfg.mesh.h_z)


def stiffness_key(cfg: SimConfig):
    return cache.make_key("A", mesh_key(cfg), sorted(cfg.conductivity.table().items()))


def _freeze(m):
    for name in ("nodes", "tets", "labels", "sigma", "vol", "xs", "ys", "zs", "box_label", "col_index"):
        getattr(m, name).flags.writeable = False
    return m


def get_mesh(cfg: SimConfig):
    """(mesh, cache_hit).  The cached mesh is read-only and shared between runs."""
    def build():
        m = build_cylinder_mesh(cfg.mesh.h_xy, cfg.mesh.h_z, cfg.geometry.length, radii=cfg.geometry.radii())
        present = np.bincount(m.labels, minlength=4)
        missing = [C.TISSUE_NAMES[k] for k in range(4) if present[k] == 0]
        if missing:
            raise SimulationError(f"the mesh contains no {', '.join(missing)} cells; make those layers thicker",
                                  stage="mesh")
        if not (m.vol > 0).all():
            raise SimulationError("degenerate tetrahedron in the mesh", stage="mesh")
        return _freeze(m)
    return cache.MESHES.get_or_build(mesh_key(cfg), build)


def with_sigma(m, cfg: SimConfig):
    """Shallow copy of the shared mesh carrying this run's conductivity table."""
    mm = copy.copy(m)
    mm.sigma = C.sigma_array(m.labels, cfg.conductivity.table())
    return mm


def get_stiffness(cfg: SimConfig, m):
    def build():
        A = fem.assemble_stiffness(with_sigma(m, cfg))
        if not np.isfinite(A.data).all():
            raise SimulationError("stiffness matrix contains non-finite entries", stage="assembly")
        return A
    return cache.STIFFNESS.get_or_build(stiffness_key(cfg), build)


def resolve_electrodes(cfg: SimConfig, m):
    """3-D positions via fem.skin_electrode_positions (step inwards onto the staircase
    skin), plus the tissue each lands in.  Requires static_checks to have passed
    (an electrode beyond the end caps would never land inside the mesh)."""
    r_skin = cfg.geometry.radii()[3]
    sp_ = cfg.electrodes.surface_points(r_skin)
    th = np.array([p[0] for p in sp_]); z = np.array([p[1] for p in sp_])
    pos = fem.skin_electrode_positions(m, th, z, r_skin=r_skin)
    tet, _, ok = fem.locate_points(m, pos)
    labels = np.where(ok, m.labels[np.maximum(tet, 0)], -1)
    return pos, th, z, labels


def electrode_row_keys(cfg: SimConfig, pos):
    ak = stiffness_key(cfg)
    return [cache.make_key("Krow", ak, [round(float(c), 12) for c in p], cfg.solver.cg_rtol) for p in pos]


def mesh_checks(cfg: SimConfig, m):
    """Checks that need the mesh: electrodes on skin, fibre inside muscle tets."""
    errors = []
    pos, th, z, labels = resolve_electrodes(cfg, m)
    for k, lab in enumerate(labels):
        if lab != C.SKIN:
            where = "outside the mesh" if lab < 0 else f"in {C.TISSUE_NAMES[lab]}"
            errors.append(f"electrode {k+1} does not land on skin (it is {where}); the skin layer has a gap there")
    geom, x, y = make_fibre_geometry(cfg)
    tet, _, ok = fem.locate_points(m, geom.pts)
    if not ok.all() or (m.labels[tet[ok]] != C.MUSCLE).any():
        errors.append("part of the fibre lies outside muscle cells of the mesh (staircase boundary); "
                      "move it away from the bone / fat interface")
    return errors, pos, labels, geom


def make_fibre_geometry(cfg: SimConfig):
    x, y = fibre_xy(cfg)
    return fibre.FibreGeometry(x=x, y=y, z_half=fibre_z_half(cfg), dz=cfg.fibre.dz), x, y


# ============================================================================ preview
def preview(cfg: SimConfig):
    """Validation + what a run would reuse, without solving anything."""
    errors, warnings = static_checks(cfg)
    out = dict(valid=False, errors=errors, warnings=warnings, mesh_key=mesh_key(cfg),
               estimated_vertices=estimate_vertices(cfg), model_version=cache.MODEL_VERSION,
               config_hash=cfg.config_hash())
    if errors:
        return out
    try:
        m, mesh_hit = get_mesh(cfg)
    except SimulationError as e:
        out["errors"].append(str(e)); return out
    errs, pos, labels, geom = mesh_checks(cfg, m)
    keys = electrode_row_keys(cfg, pos)
    cached_rows = [cache.ROWS.has(k) for k in keys]
    out.update(
        valid=not errs, errors=errors + errs,
        mesh=dict(n_vertices=len(m.nodes), n_tets=len(m.tets)),
        electrodes=_electrode_block(cfg, pos, labels),
        fibre=_fibre_block(cfg, geom),
        plan=dict(mesh="cached" if mesh_hit else "built now (cached)",
                  stiffness="cached" if cache.STIFFNESS.peek(stiffness_key(cfg)) else "assemble",
                  adjoint_rows_cached=int(sum(cached_rows)), adjoint_rows_to_solve=int(len(keys) - sum(cached_rows)),
                  per_electrode_cached=cached_rows),
    )
    return out


def _electrode_block(cfg, pos, labels):
    r_skin = cfg.geometry.radii()[3]
    sp_ = cfg.electrodes.surface_points(r_skin)
    return dict(
        positions_m=pos.tolist(),
        surface=[dict(theta=t, z=z) for t, z in sp_],
        inset_below_nominal_skin_m=[float(r_skin - math.hypot(p[0], p[1])) for p in pos],
        tissue=[C.TISSUE_NAMES[l] if l >= 0 else "outside" for l in labels],
        pairs=cfg.electrodes.pairs(),
        channel_labels=[f"ch{j+1}: e{a+1}-e{b+1}" for j, (a, b) in enumerate(cfg.electrodes.pairs())],
        sampling="point (barycentric interpolation of the P1 potential, fem.selection_matrix)",
    )


def _fibre_block(cfg, geom):
    f = cfg.fibre
    return dict(x_m=float(geom.pts[0, 0]), y_m=float(geom.pts[0, 1]),
                stored_z_m=[float(geom.edges[0]), float(geom.edges[-1])],
                active_z_m=[f.nmj_z - f.l2, f.nmj_z + f.l1], nmj_z_m=f.nmj_z,
                cell_centres_z_m=geom.z.tolist(), dz_m=geom.dz, cv_m_s=f.cv, radius_m=f.radius,
                direction=[0.0, 0.0, 1.0])


# ============================================================================ run
def _solve_rows(cfg, m, A, pos, progress):
    """Adjoint rows K_e, reusing cached ones.  New rows come from fem.adjoint_K
    (one electrode at a time, identical to the batched call because the solves are
    independent) and are checked by recomputing the true residual."""
    keys = electrode_row_keys(cfg, pos)
    rtol = cfg.solver.cg_rtol
    rows, info = [], []
    n_new = sum(not cache.ROWS.has(k) for k in keys)
    done = 0
    for e, k in enumerate(keys):
        hit = cache.ROWS.get(k)
        if hit is not None:
            row, it, res = hit
            rows.append(row); info.append(dict(electrode=e + 1, cached=True, iterations=it, rel_residual=res))
            continue
        done += 1
        progress(f"adjoint solve {done}/{n_new} (electrode {e+1})")
        S = fem.selection_matrix(m, pos[e:e + 1])
        try:
            K, iters = fem.adjoint_K(A, S, rtol=rtol)
        except RuntimeError as exc:                      # fem.cg_solve: CG did not converge
            raise SimulationError(f"adjoint solve for electrode {e+1} failed: {exc}", stage="adjoint")
        row = K[0]
        b = S.toarray().ravel(); b = b - b.mean()        # the balanced right-hand side cg_solve used
        res = float(np.linalg.norm(A @ row - b) / np.linalg.norm(b))
        if not np.isfinite(row).all() or res > RESIDUAL_SLACK * rtol:
            raise SimulationError(f"adjoint solve for electrode {e+1} did not reach the tolerance "
                                  f"(relative residual {res:.2e}, rtol {rtol:.0e})", stage="adjoint")
        cache.ROWS.put(k, row, iters[0], res)
        rows.append(row); info.append(dict(electrode=e + 1, cached=False, iterations=int(iters[0]), rel_residual=res))
    return np.vstack(rows), info


def run(cfg: SimConfig, progress=lambda msg: None):
    """Full single-fibre simulation.  Returns a JSON-ready dict; raises SimulationError."""
    T = {}
    t_all = time.perf_counter()
    errors, warnings = static_checks(cfg)
    if errors:
        raise SimulationError("; ".join(errors), stage="validation", details=dict(errors=errors))

    progress("mesh")
    t0 = time.perf_counter(); m, mesh_hit = get_mesh(cfg); T["mesh_s"] = time.perf_counter() - t0
    mk = mesh_key(cfg)
    if not mesh_hit:
        mesh_status = "built"
    elif mk in _MESHES_USED_BY_RUNS:
        mesh_status = "reused"
    else:
        mesh_status = "built for this geometry (during preview)"
    _MESHES_USED_BY_RUNS.add(mk)
    errs, pos, labels, geom = mesh_checks(cfg, m)
    if errs:
        raise SimulationError("; ".join(errs), stage="validation", details=dict(errors=errs))

    progress("assemble stiffness matrix")
    t0 = time.perf_counter(); A, a_hit = get_stiffness(cfg, m); T["assemble_s"] = time.perf_counter() - t0

    t0 = time.perf_counter(); K, row_info = _solve_rows(cfg, m, A, pos, progress); T["adjoint_s"] = time.perf_counter() - t0

    progress("fibre basis points + signal")
    t0 = time.perf_counter()
    basis = fem.BasisSolutions(m, K)                                   # Eq. (7)
    try:
        P = fibre.precompute_basis_points(basis, geom)                 # Eq. (5)
    except ValueError as exc:
        raise SimulationError(f"fibre points could not be located in muscle: {exc}", stage="basis")
    T["basis_s"] = time.perf_counter() - t0

    f = cfg.fibre
    prm = fibre.FibreParams(z0=f.nmj_z, L1=f.l1, L2=f.l2, v=f.cv, taper=f.taper, radius=f.radius, sigma_in=f.sigma_in)
    t = np.arange(cfg.timing.n_samples()) / cfg.timing.fs
    t0 = time.perf_counter()
    mono, D = fibre.simulate(P, geom, t, prm)                          # Eq. (10), volts
    pairs = np.array(cfg.electrodes.pairs())
    diff = mono[pairs[:, 0]] - mono[pairs[:, 1]]                       # V_EMG = phi_1 - phi_2
    T["signal_s"] = time.perf_counter() - t0

    if not (np.isfinite(mono).all() and np.isfinite(D).all()):
        raise SimulationError("the simulated signal contains non-finite values", stage="signal")
    conservation = float(np.abs(D.sum(axis=0)).max() / max(np.abs(D).max(), 1e-300))
    if conservation > 1e-10:
        raise SimulationError(f"fibre current is not conserved (relative {conservation:.1e})", stage="signal")
    tail = float(np.abs(diff[:, -3:]).max() / max(np.abs(diff).max(), 1e-300))
    if tail > 0.05:
        warnings.append(f"signals are still {100*tail:.0f}% of their peak at the end of the window; "
                        f"increase the duration")
    T["total_s"] = time.perf_counter() - t_all

    cell_current = (f.sigma_in * np.pi * f.radius ** 2) * D            # A per fibre cell (Eq. 8 integrated)
    send_cells = cell_current.size <= 400_000
    return dict(
        kind="single_fibre",
        model_version=cache.MODEL_VERSION,
        config=cfg.model_dump(mode="json"),
        config_hash=cfg.config_hash(),
        units=dict(time="s", voltage="V", length="m", current="A", velocity="m/s"),
        time_s=t.tolist(),
        mono_V=mono.tolist(),
        diff_V=diff.tolist(),
        metrics=dict(
            mono_p2p_V=(mono.max(1) - mono.min(1)).tolist(), mono_rms_V=np.sqrt((mono ** 2).mean(1)).tolist(),
            diff_p2p_V=(diff.max(1) - diff.min(1)).tolist(), diff_rms_V=np.sqrt((diff ** 2).mean(1)).tolist(),
            duration_s=float(cfg.timing.n_samples() / cfg.timing.fs), fs_Hz=cfg.timing.fs,
            n_samples=int(len(t)),
        ),
        conduction_velocity=dict(configured_m_s=f.cv, **estimate_cv(cfg, pos, t, diff)),
        electrodes=_electrode_block(cfg, pos, labels),
        fibre=dict(**_fibre_block(cfg, geom),
                   cell_current_A=cell_current.tolist() if send_cells else None,
                   cell_current_note="current leaving each fibre cell into the tissue, A (source > 0); "
                                     "rows = cells, columns = time samples",
                   # Stage 4 integration point: motor-unit id of this fibre once MU pools are integrated
                   motor_unit=None),
        solver=dict(
            method="P1 FEM, insulating (Neumann) boundary, zero-mean potential reference, "
                   "Jacobi-preconditioned CG, adjoint (reciprocity) lead fields",
            cg_rtol=cfg.solver.cg_rtol, n_vertices=len(m.nodes), n_tets=len(m.tets),
            n_basis_nodes=int(len(basis.muscle_nodes)), rows=row_info,
            max_rel_residual=max(r["rel_residual"] for r in row_info),
            current_conservation=conservation,
        ),
        cache=dict(mesh=mesh_status, stiffness="reused" if a_hit else "assembled",
                   adjoint_rows_reused=sum(r["cached"] for r in row_info),
                   adjoint_rows_solved=sum(not r["cached"] for r in row_info),
                   basis_points="recomputed", signal="recomputed"),
        timings=T,
        warnings=warnings,
    )


# ============================================================================ analysis
def _xcorr_lag(t, a, b):
    """Same method as stage3_single_fibre.py check 3: cross-correlation after 20x
    linear interpolation; lag of the maximum = travel time from a to b."""
    tt = np.linspace(0, t[-1], len(t) * 20)
    fa, fb = np.interp(tt, t, a), np.interp(tt, t, b)
    cc = np.correlate(fb - fb.mean(), fa - fa.mean(), "full")
    return (np.argmax(cc) - (len(tt) - 1)) * (tt[1] - tt[0])


def estimate_cv(cfg: SimConfig, pos, t, diff):
    """Velocity ESTIMATED from the recorded signals (distinct from the input value).

    Uses differential channels whose two electrodes both lie over the propagating
    part of the +z semi-fibre (20 mm past the NMJ to 15 mm before the tendon) and
    are separated mainly along z; takes the first and last of them."""
    f = cfg.fibre
    lo, hi = f.nmj_z + 20e-3, f.nmj_z + f.l1 - 15e-3
    cand = []
    for j, (a, b) in enumerate(cfg.electrodes.pairs()):
        za, zb = pos[a, 2], pos[b, 2]
        d3 = np.linalg.norm(pos[a] - pos[b])
        if lo <= min(za, zb) and max(za, zb) <= hi and abs(za - zb) >= 0.7 * d3:
            cand.append((0.5 * (za + zb), j))
    method = ("cross-correlation of two differential channels over the propagating zone "
              "(stage 3 method, 20x interpolation)")
    if len(cand) < 2:
        return dict(estimated_m_s=None, method=method,
                    note="not estimable: fewer than two channels lie over the propagating zone "
                         f"z in [{lo*1e3:.0f}, {hi*1e3:.0f}] mm")
    cand.sort()
    (z1, j1), (z2, j2) = cand[0], cand[-1]
    lag = _xcorr_lag(t, diff[j1], diff[j2])
    if lag <= 0:
        return dict(estimated_m_s=None, method=method, note="not estimable: non-positive lag", channels=[j1 + 1, j2 + 1])
    return dict(estimated_m_s=float((z2 - z1) / lag), method=method, channels=[j1 + 1, j2 + 1],
                distance_m=float(z2 - z1), lag_s=float(lag), note="")
