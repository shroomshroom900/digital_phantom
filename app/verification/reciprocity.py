"""
Adjoint (reciprocity) check -- a light version of experiment D in stage2_forward.py.

For randomly chosen muscle vertices j, the electrode voltages caused by a unit
current source at j are computed two ways on the coarse arm mesh:
    adjoint     V_basis[:, j]  from  A K^T = S^T          (fem.adjoint_K, n_e solves)
    brute force S A^-1 e_j     one CG solve per source    (fem.cg_solve)
and compared by NMSE, as in the brief.  The random choice uses an explicit seed.
"""
import numpy as np

import config as C
import fem
from mesh import build_cylinder_mesh
from app.verification.registry import register


@register(
    name="adjoint_reciprocity",
    title="Adjoint lead fields vs brute-force solves (experiment D, light)",
    formula="NMSE( K B e_j ,  S A^-1 e_j ),  K = S A^-1 from A K^T = S^T",
    params=[
        dict(key="n_sources", label="Random source vertices", unit="", display_unit="", display_scale=1,
             default=40, min=1, max=400, integer=True),
        dict(key="seed", label="Random seed", unit="", display_unit="", display_scale=1,
             default=C.RNG_SEED, min=0, max=2**31 - 1, integer=True),
    ],
    assumptions=["coarse mesh (h_xy = 4 mm, h_z = 8 mm) and default conductivities from config.py",
                 "16-electrode line array of stage 2", "CG rtol 1e-12"],
)
def adjoint_reciprocity(n_sources, seed, tolerance=1e-8):
    n_sources = int(n_sources)
    if not 1 <= n_sources <= 400:
        raise ValueError("n_sources must be between 1 and 400")
    rng = np.random.default_rng(int(seed))
    m = build_cylinder_mesh(C.H_XY_COARSE, C.H_Z_COARSE)
    A = fem.assemble_stiffness(m)
    z = (np.arange(C.N_ELEC_LINEAR) - (C.N_ELEC_LINEAR - 1) / 2) * C.ELEC_SPACING
    S = fem.selection_matrix(m, fem.skin_electrode_positions(m, 0.0, z))
    K, iters = fem.adjoint_K(A, S, rtol=1e-12)
    basis = fem.BasisSolutions(m, K)
    cols = rng.choice(len(basis.muscle_nodes), min(n_sources, len(basis.muscle_nodes)), replace=False)
    V_bf = np.empty((S.shape[0], len(cols)))
    for c, j in enumerate(cols):
        b = np.zeros(A.shape[0]); b[basis.muscle_nodes[j]] = 1.0
        x, _ = fem.cg_solve(A, b, rtol=1e-12)
        V_bf[:, c] = S @ x
    V_adj = basis.V[:, cols]
    nmse = float(np.sum((V_adj - V_bf) ** 2) / np.sum(V_bf ** 2))
    return dict(
        passed=nmse < tolerance, tolerance=tolerance,
        summary=[dict(quantity="NMSE(adjoint, brute force)", value=nmse, unit="1"),
                 dict(quantity="Sources compared", value=len(cols), unit=""),
                 dict(quantity="Adjoint solves", value=S.shape[0], unit=""),
                 dict(quantity="Mean CG iterations (adjoint)", value=float(np.mean(iters)), unit=""),
                 dict(quantity="Mesh vertices", value=len(m.nodes), unit=""),
                 dict(quantity="Seed", value=int(seed), unit="")],
    )
