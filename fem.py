"""
fem.py -- the finite-element machinery (paper: Methods, "Forward problem").

Cast of characters (same letters as the paper)
----------------------------------------------
  A      (n_v x n_v) sparse symmetric "stiffness" matrix           Eq. (2)
  v      (n_v,)      potential at every mesh vertex (the unknowns)
  b      (n_v,)      source vector ("how much current enters at each vertex")
  S      (n_e x n_v) selection matrix: picks the electrode potentials   Eq. (3)
  K = S A^-1         (n_e x n_v) the "adjoint" / lead-field matrix        Eq. (6)
  V_basis = K B      (n_e x n_s) potential at each electrode caused by a
                     unit current source sitting on each basis vertex   Eq. (4),(7)

The story in one paragraph
--------------------------
For ONE unit current source at vertex j, the electrode voltages are
S A^-1 e_j  (e_j = vector of zeros with a 1 in place j).  Doing that for
every one of the ~10^5 possible source vertices means ~10^5 expensive
solves.  The ADJOINT trick notices that S A^-1 is a small-and-fat matrix
(n_e rows) and that A is symmetric, so  (S A^-1)^T = A^-1 S^T.  Solving
A y = S^T for the n_e columns of S^T gives ALL the rows of K at once:
only n_e ~ 16 solves instead of ~10^5.   (Physical meaning = "reciprocity":
the voltage an electrode sees from a source at P equals the voltage at P
when the same current is pushed in through the electrode.)
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import config as C
from mesh import _KUHN_PATHS


# ======================================================================
# 1. The stiffness matrix A   (paper Eq. 2 and the formula for A_ij)
# ======================================================================
def assemble_stiffness(m):
    """Build A with  A_ij = integral over the volume of  grad(w_i)^T sigma grad(w_j).

    First-year explanation
    ----------------------
    w_i is a "hat" function: equal to 1 at vertex i, 0 at every other vertex,
    and straight-line (linear) inside each tet.  A tet only "sees" the 4 hat
    functions of its own corners, and inside ONE tet each hat function has a
    CONSTANT gradient g_a (a vector of 3 numbers).  So the integral over one
    tet is simply   (tet volume) * g_a^T sigma g_b   -- no calculus needed.
    We compute that 4x4 block for every tet and add it into the big matrix
    wherever the tet's corner numbers say it belongs.

    Gradients of the hat functions
    ------------------------------
    Put the tet's first corner p0 at the origin.  The other corners give
    three edge vectors e1, e2, e3.  Barycentric coordinates lam_1..3 satisfy
    x - p0 = [e1 e2 e3] (lam_1, lam_2, lam_3)^T, hence
    grad(lam_a) = row a of the inverse of [e1 e2 e3].
    The 4th hat function: lam_0 = 1 - lam_1 - lam_2 - lam_3, so
    grad(lam_0) = -(sum of the other three).
    """
    p = m.nodes[m.tets]                                   # (n_t, 4, 3) corners
    E = (p[:, 1:, :] - p[:, :1, :])                       # (n_t, 3, 3), rows=edges
    Einv = np.linalg.inv(np.transpose(E, (0, 2, 1)))      # inverse of [e1 e2 e3]
    G = np.empty((len(m.tets), 4, 3))                     # grad of each hat fn
    G[:, 1:, :] = Einv                                    # rows 1..3
    G[:, 0, :] = -Einv.sum(axis=1)                        # row 0

    # local 4x4 stiffness blocks:  Kloc[t,a,b] = vol * g_a . (sigma g_b)
    # sigma is diagonal, so sigma g_b = (sx*gx, sy*gy, sz*gz)
    Kloc = m.vol[:, None, None] * np.einsum("tai,ti,tbi->tab", G, m.sigma, G)

    # scatter-add all blocks into one sparse matrix (duplicates are summed)
    rows = np.repeat(m.tets, 4, axis=1).ravel()           # i of every entry
    cols = np.tile(m.tets, (1, 4)).ravel()                # j of every entry
    nv = len(m.nodes)
    A = sp.coo_matrix((Kloc.ravel(), (rows, cols)), shape=(nv, nv)).tocsr()
    return A


# ======================================================================
# 2. Solving A x = b  with the conjugate-gradient (CG) method
# ======================================================================
def cg_solve(A, b, rtol=1e-8, maxiter=5000, Minv_diag=None):
    """Solve A x = b iteratively; returns (x, n_iterations).

    NULL SPACE (paper: "constrain potentials v to have a zero sum")
    ---------------------------------------------------------------
    Only potential DIFFERENCES matter physically, so adding a constant to
    every v still solves the equations: A has a one-dimensional null space
    (the constant vector 1; indeed A 1 = 0).  Two consequences:
      (i)  a solution exists only if b sums to zero (total current in =
           total current out, because no current leaves through the
           insulating boundary);
      (ii) the solution is unique only after fixing the constant.
    We deal with both by:
      * removing the mean of b   ->  b_p = b - mean(b).  Physically this
        adds a tiny uniform "sink" of -1/n_v at every vertex to balance a
        lone unit source.  Later this has NO effect on EMG, because the
        current of an active fibre sums to ZERO along the fibre (Eq. 8 is a
        derivative), so these little uniform sinks cancel exactly.
      * CG started from x = 0 with b_p in range(A) never leaves the
        zero-sum subspace, so the answer automatically satisfies sum(v)=0.
        We still re-subtract the mean to kill round-off.
    """
    b = b - b.mean()
    if Minv_diag is None:                                  # Jacobi preconditioner:
        Minv_diag = 1.0 / A.diagonal()                     # "divide by the diagonal"
    M = spla.LinearOperator(A.shape, matvec=lambda r: Minv_diag * r)
    count = [0]

    def _cb(_):
        count[0] += 1
    x, info = spla.cg(A, b, rtol=rtol, atol=0.0, maxiter=maxiter, M=M, callback=_cb)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info}) in {count[0]} iterations")
    return x - x.mean(), count[0]


# ======================================================================
# 3. Finding which tetrahedron contains a point; barycentric coordinates
# ======================================================================
# Table: axis order (a, b, c) encoded as 9a + 3b + c  ->  index of the Kuhn tet
# that walks along axis a first, then b, then c.
_ORDER_TABLE = np.full(27, -1, dtype=np.int64)
for _t, _p in enumerate(_KUHN_PATHS):
    _a = [int(np.argmax(np.array(_p[c]) - np.array(_p[c - 1]))) for c in (1, 2, 3)]
    _ORDER_TABLE[_a[0] * 9 + _a[1] * 3 + _a[2]] = _t



def locate_points(m, pts):
    """For each 3-D point find (tet index, 4 barycentric coordinates, valid).

    Barycentric coordinates (paper, text after Eq. 5)
    -------------------------------------------------
    Every point r inside a tet with corners p0..p3 can be written as a
    weighted average   r = l0 p0 + l1 p1 + l2 p2 + l3 p3   with weights
    l0..l3 >= 0 that add up to 1.  They tell you "how close to each corner"
    the point is; the field at r is then  l0*v0 + l1*v1 + l2*v2 + l3*v3.
    (That is exactly what a P1 -- linear -- field means.)

    How we find them FAST on our structured grid
    --------------------------------------------
    1. which box?   integer division of the coordinates by the box size;
    2. fractional position (fx, fy, fz) in [0,1]^3 inside the box;
    3. which of the 6 Kuhn tets?   The tet "x then y then z" holds the points
       with fx >= fy >= fz, and similarly for the other 5 orderings, so we
       just sort the three fractions;
    4. the weights follow from the sorted fractions:  if f_a >= f_b >= f_c,
          l0 = 1 - f_a,  l1 = f_a - f_b,  l2 = f_b - f_c,  l3 = f_c.
       (Check: l0+l1+l2+l3 = 1, and sum_k l_k * corner_k = point. OK.)
    """
    pts = np.atleast_2d(pts)
    fx = (pts[:, 0] - m.xs[0]) / m.h_xy
    fy = (pts[:, 1] - m.ys[0]) / m.h_xy
    fz = (pts[:, 2] - m.zs[0]) / m.h_z
    i = np.floor(fx).astype(int); j = np.floor(fy).astype(int); k = np.floor(fz).astype(int)
    ok = (i >= 0) & (i < m.n_xy) & (j >= 0) & (j < m.n_xy) & (k >= 0) & (k < m.n_z)
    ic, jc, kc = np.clip(i, 0, m.n_xy - 1), np.clip(j, 0, m.n_xy - 1), np.clip(k, 0, m.n_z - 1)
    col = m.col_index[ic, jc]
    ok &= col >= 0                                        # box must be part of the limb
    box = col * m.n_z + kc
    f = np.column_stack([fx - i, fy - j, fz - k])         # fractions in the box
    order = np.argsort(-f, axis=1)                        # axes, largest fraction first
    fs = np.take_along_axis(f, order, axis=1)             # f_a >= f_b >= f_c
    lam = np.column_stack([1 - fs[:, 0], fs[:, 0] - fs[:, 1], fs[:, 1] - fs[:, 2], fs[:, 2]])
    # which Kuhn path has this axis order?  (table lookup, fully vectorised)
    tet_in_box = _ORDER_TABLE[order[:, 0] * 9 + order[:, 1] * 3 + order[:, 2]]
    tet = 6 * box + tet_in_box
    tet[~ok] = -1
    return tet, lam, ok


# ======================================================================
# 4. Selection matrix S  (paper Eq. 3 and the paragraph before it)
# ======================================================================
def selection_matrix(m, pts):
    """S has one row per electrode; row e holds the 4 barycentric weights of
    the electrode position at the 4 corners of its tet (zeros elsewhere).
    Then (S v)_e = potential interpolated at electrode e.  (The paper also
    allows a row to average over an electrode's area; we use point electrodes.)
    """
    tet, lam, ok = locate_points(m, pts)
    if not ok.all():
        raise ValueError(f"electrode(s) outside the mesh: {np.nonzero(~ok)[0]}")
    ne = len(pts)
    rows = np.repeat(np.arange(ne), 4)
    cols = m.tets[tet].ravel()
    return sp.csr_matrix((lam.ravel(), (rows, cols)), shape=(ne, len(m.nodes)))


def skin_electrode_positions(m, theta, z, r_skin=None):
    """Electrode positions on the outer skin surface at angle theta, height z.

    Our skin has a staircase edge, so a point at exactly r = 2.4 cm may fall
    outside the mesh.  We start at the nominal radius and step inwards in
    0.1 mm steps until the point lies inside the mesh: the electrode then
    touches the (staircase) outer surface.  [ASSUMED: point electrodes]
    r_skin: starting radius; default = config.R_SKIN (the web app passes its own).
    """
    theta = np.atleast_1d(theta) * np.ones_like(np.atleast_1d(z), dtype=float)
    z = np.atleast_1d(z) * np.ones_like(theta)
    out = np.empty((len(z), 3))
    for n in range(len(z)):
        r = C.R_SKIN if r_skin is None else r_skin
        while True:
            p = np.array([[r * np.cos(theta[n]), r * np.sin(theta[n]), z[n]]])
            if locate_points(m, p)[2][0]:
                break
            r -= 1e-4
        out[n] = p[0]
    return out


# ======================================================================
# 5. The adjoint method  (paper Eqs. 6 and 7)
# ======================================================================
def adjoint_K(A, S, rtol=1e-8):
    """Solve  A K^T = S^T  column by column (ONE CG solve per electrode).

    Because A is symmetric, K^T = A^-1 S^T  means  K = S A^-1.
    Row e of K tells, for every vertex j, the voltage seen by electrode e
    when a unit current source sits at vertex j.
    Returns K (n_e x n_v dense) and a list of CG iteration counts.
    """
    ST = S.T.tocsc()
    ne = S.shape[0]
    K = np.empty((ne, A.shape[0]))
    iters = []
    Minv = 1.0 / A.diagonal()
    for e in range(ne):
        y, it = cg_solve(A, ST[:, e].toarray().ravel(), rtol=rtol, Minv_diag=Minv)
        K[e] = y
        iters.append(it)
    return K, iters


# ======================================================================
# 6. Basis sources  (paper Eqs. 4, 5, 7)
# ======================================================================
class BasisSolutions:
    """V_basis (n_e x n_s) for sources restricted to the muscle sub-domain.

    The paper (after Eq. 5): "one may restrict potential sources to be
    located inside specific subdomains ... n_s is the number of vertices of
    these subdomains and B is a submatrix of the identity".  Fibres only live
    in muscle, so n_s = number of vertices touching a muscle tet.
    """

    def __init__(self, m, K):
        self.m = m
        self.muscle_nodes = np.unique(m.tets[m.labels == C.MUSCLE])   # n_s vertex ids
        self.node_to_col = np.full(len(m.nodes), -1, dtype=np.int64)
        self.node_to_col[self.muscle_nodes] = np.arange(len(self.muscle_nodes))
        # Eq. (7): V_basis = K B.  B just selects columns, so no matrix product:
        self.V = K[:, self.muscle_nodes]                              # (n_e, n_s)

    def point_potentials(self, pts):
        """Eq. (5):  v_point(r) = V_basis lambda(r)  for many points at once.

        pts : (n_p, 3) positions (must lie inside muscle tets)
        returns (n_e, n_p): column p = electrode voltages caused by a unit
        current source at pts[p].
        """
        tet, lam, ok = locate_points(self.m, pts)
        if not ok.all():
            raise ValueError("some source points are outside the mesh")
        cols = self.node_to_col[self.m.tets[tet]]                     # (n_p, 4)
        if (cols < 0).any():
            raise ValueError("some source points are not inside muscle tets")
        # sum over the 4 corners of  lambda_c * V_basis[:, corner]
        return np.einsum("epc,pc->ep", self.V[:, cols], lam)
