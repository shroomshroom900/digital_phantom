"""
Analytical benchmark: current through stacked conductive blocks.

    R_i    = L_i / (sigma_i * S)
    DeltaV = I * sum_i R_i

Set-up (all handled by the SAME FEM code as the arm: fem.assemble_stiffness,
fem.cg_solve, Kuhn-split tetrahedra from mesh._KUHN_PATHS):
  * a square prism of cross-section S = W x W, layers stacked along z, each with
    its own constant (isotropic) conductivity; ideal contact between layers
    (shared mesh nodes);
  * insulated side faces = the natural (Neumann) boundary condition of the FEM;
  * current I injected as a uniform flux I/S over the top face (z = z_max) and
    withdrawn uniformly over the bottom face (z = 0): consistent P1 nodal loads
    b_i = (I/S) * (area of the face triangles touching node i) / 3, so sum(b) = 0
    exactly and the pure-Neumann problem is solvable;
  * voltage reference: the zero-mean gauge of fem.cg_solve (only differences
    matter; DeltaV is reference-independent).
Layer thicknesses are snapped to whole mesh cells so the interfaces coincide with
node planes; the exact solution (piecewise linear in z) then lies in the P1 space,
so FEM and formula should agree to solver precision.  The snapped (effective)
thicknesses are reported and used for the analytical value.
"""
import numpy as np

import fem
from mesh import _KUHN_PATHS, Mesh
from app.verification.registry import register

DEFAULT_LAYERS = [  # outermost -> innermost, like the arm, but stacked along z
    dict(name="skin", length=1e-3, sigma=1.0),
    dict(name="fat", length=3e-3, sigma=0.04),
    dict(name="muscle (transverse)", length=13e-3, sigma=0.1),
    dict(name="bone", length=7e-3, sigma=0.02),
]


def block_mesh(width, layers, h_xy, h_z):
    n_xy = max(1, int(round(width / h_xy)))
    n_k = [max(1, int(round(l["length"] / h_z))) for l in layers]
    n_z = sum(n_k)
    xs = h_xy * np.arange(n_xy + 1); ys = xs.copy(); zs = h_z * np.arange(n_z + 1)
    I, J, K = np.meshgrid(np.arange(n_xy), np.arange(n_xy), np.arange(n_z), indexing="ij")
    I, J, K = I.ravel(), J.ravel(), K.ravel()
    # layers[0] is the TOP layer: k = 0 is the bottom cell, so stack in reverse
    layer_of_k = np.repeat(np.arange(len(layers))[::-1], n_k[::-1])
    ny1, nz1 = n_xy + 1, n_z + 1

    def node_id(i, j, k):
        return (i * ny1 + j) * nz1 + k
    tets = np.empty((len(I) * 6, 4), dtype=np.int64)
    for t, path in enumerate(_KUHN_PATHS):
        for c, (di, dj, dk) in enumerate(path):
            tets[t::6, c] = node_id(I + di, J + dj, K + dk)
    gi, gj, gk = np.meshgrid(np.arange(n_xy + 1), np.arange(n_xy + 1), np.arange(n_z + 1), indexing="ij")
    nodes = np.column_stack([xs[gi.ravel()], ys[gj.ravel()], zs[gk.ravel()]])
    labels = np.repeat(layer_of_k[K], 6)
    sig = np.array([l["sigma"] for l in layers], dtype=float)
    p = nodes[tets]
    m = Mesh()
    m.nodes, m.tets, m.labels = nodes, tets, labels
    m.sigma = np.repeat(sig[labels][:, None], 3, axis=1)
    m.vol = np.abs(np.linalg.det(p[:, 1:, :] - p[:, :1, :])) / 6.0
    m.n_k, m.zs, m.w_eff = n_k, zs, n_xy * h_xy
    return m


def face_loads(m, z_face, flux):
    """Consistent P1 load of a uniform normal flux on the plane z = z_face."""
    b = np.zeros(len(m.nodes))
    on = np.isclose(m.nodes[:, 2], z_face)
    for face in ([0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]):
        f = m.tets[:, face]
        sel = on[f].all(axis=1)
        if not sel.any():
            continue
        q = m.nodes[f[sel]]
        area = 0.5 * np.linalg.norm(np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]), axis=1)
        np.add.at(b, f[sel].ravel(), np.repeat(flux * area / 3.0, 3))
    return b, on


@register(
    name="stacked_blocks",
    title="Current through stacked conductive blocks",
    formula="R_i = L_i / (sigma_i * S);   DeltaV = I * sum_i R_i",
    params=[
        dict(key="layers", label="Layers (top -> bottom)", default=DEFAULT_LAYERS,
             columns=[dict(key="name", label="name"),
                      dict(key="length", label="thickness", unit="m", display_unit="mm", display_scale=1e3),
                      dict(key="sigma", label="sigma", unit="S/m", display_unit="S/m", display_scale=1)]),
        dict(key="width", label="Cross-section side W (S = W^2)", unit="m", display_unit="mm", display_scale=1e3,
             default=10e-3, min=2e-3, max=50e-3),
        dict(key="current", label="Current I", unit="A", display_unit="mA", display_scale=1e3,
             default=1e-3, min=1e-6, max=1.0),
        dict(key="h_xy", label="Mesh spacing x, y", unit="m", display_unit="mm", display_scale=1e3,
             default=1e-3, min=0.25e-3, max=5e-3),
        dict(key="h_z", label="Mesh spacing z", unit="m", display_unit="mm", display_scale=1e3,
             default=0.5e-3, min=0.1e-3, max=5e-3),
        dict(key="rtol", label="CG relative tolerance", unit="1", display_unit="", display_scale=1,
             default=1e-12, min=1e-14, max=1e-6),
    ],
    assumptions=["constant isotropic conductivity per layer", "shared cross-section S", "ideal contact",
                 "insulated side faces (natural Neumann BC)", "uniform flux in at the top, out at the bottom",
                 "zero-mean voltage reference (fem.cg_solve)"],
)
def stacked_blocks(layers, width, current, h_xy, h_z, rtol, tolerance=1e-6):
    if not layers:
        raise ValueError("need at least one layer")
    for l in layers:
        if not (l["length"] > 0 and l["sigma"] > 0):
            raise ValueError("layer thickness and conductivity must be > 0")
    m = block_mesh(width, layers, h_xy, h_z)
    if len(m.nodes) > 300_000:
        raise ValueError(f"block mesh too large ({len(m.nodes):,} vertices); coarsen h_xy / h_z")
    S = m.w_eff ** 2
    A = fem.assemble_stiffness(m)
    z_top = m.zs[-1]
    b_top, on_top = face_loads(m, z_top, +current / S)   # current enters at the top
    b_bot, on_bot = face_loads(m, 0.0, -current / S)     # and leaves at the bottom
    b = b_top + b_bot
    v, iters = fem.cg_solve(A, b, rtol=rtol, maxiter=20000)
    res = float(np.linalg.norm(A @ v - (b - b.mean())) / np.linalg.norm(b))

    # analytical, with the effective (mesh-snapped) thicknesses
    L_eff = [n * h_z for n in m.n_k]
    R = [L / (l["sigma"] * S) for L, l in zip(L_eff, layers)]
    dV_ana = current * sum(R)
    R_nom = [l["length"] / (l["sigma"] * S_nom) for l, S_nom in zip(layers, [width ** 2] * len(layers))]

    # FEM: mean potential on each node plane -> per-layer drops (top layer first)
    zn = m.nodes[:, 2]
    plane = np.rint(zn / h_z).astype(int)
    v_plane = np.bincount(plane, weights=v) / np.bincount(plane)
    dV_fem = float(v_plane[-1] - v_plane[0])
    bounds = np.concatenate([[0], np.cumsum(m.n_k)])     # plane index of each interface (bottom-up)
    rows = []
    # layers are listed top -> bottom; layer i occupies planes [n_z - bounds[i+1], n_z - bounds[i]]
    nz = bounds[-1]
    for i, (l, L, Ri) in enumerate(zip(layers, L_eff, R)):
        top = nz - bounds[i]; bot = nz - bounds[i + 1]
        dfem = float(v_plane[top] - v_plane[bot])
        rows.append(dict(layer=l["name"], length_nominal_m=l["length"], length_effective_m=L, sigma_S_m=l["sigma"],
                         R_ohm=Ri, dV_analytic_V=current * Ri, dV_fem_V=dfem,
                         rel_diff=abs(dfem - current * Ri) / (current * Ri)))
    nonuni = float(max(np.ptp(v[on_top]), np.ptp(v[on_bot])) / abs(dV_fem))
    abs_diff = abs(dV_fem - dV_ana)
    rel = abs_diff / abs(dV_ana)
    z_profile = (np.arange(len(v_plane)) * h_z).tolist()
    # analytic profile: potential relative to the bottom face
    phi_ana = np.zeros(len(v_plane))
    for k in range(1, len(v_plane)):
        cell_from_top = nz - k                        # the cell between planes k-1 and k
        layer = np.searchsorted(bounds, cell_from_top, side="right") - 1
        phi_ana[k] = phi_ana[k - 1] + current * h_z / (layers[layer]["sigma"] * S)
    return dict(
        passed=bool(rel < tolerance and res < 1e3 * rtol),
        tolerance=tolerance,
        summary=[
            dict(quantity="Analytical DeltaV = I sum R_i", value=dV_ana, unit="V"),
            dict(quantity="FEM DeltaV (mean top face - mean bottom face)", value=dV_fem, unit="V"),
            dict(quantity="Absolute difference", value=abs_diff, unit="V"),
            dict(quantity="Relative difference", value=rel, unit="1"),
            dict(quantity="sum R_i (effective thicknesses)", value=sum(R), unit="ohm"),
            dict(quantity="sum R_i (nominal thicknesses and width)", value=sum(R_nom), unit="ohm"),
            dict(quantity="Face non-uniformity (ptp / DeltaV)", value=nonuni, unit="1"),
            dict(quantity="CG iterations", value=iters, unit=""),
            dict(quantity="CG relative residual (recomputed)", value=res, unit="1"),
            dict(quantity="Mesh vertices", value=len(m.nodes), unit=""),
            dict(quantity="Effective cross-section S", value=S, unit="m^2"),
        ],
        layers=rows,
        profile=dict(x=z_profile, x_label="z (m), bottom face = 0", unit="V",
                     series=[dict(name="FEM (plane mean, relative to bottom)", y=(v_plane - v_plane[0]).tolist()),
                             dict(name="analytical", y=phi_ana.tolist())]),
        notes=["layers[0] is the TOP layer (current enters there)",
               "thicknesses are snapped to multiples of h_z; the analytical value uses the snapped values"],
    )
