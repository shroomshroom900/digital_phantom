"""
mesh.py  --  build the volume conductor as a mesh of tetrahedra.

BIG IDEA (paper, "Biophysics" + Methods "Forward problem")
----------------------------------------------------------
We cannot solve the Poisson equation with pen and paper for a weird shape,
so we chop the limb into thousands of tiny tetrahedra ("tets"; a tet is a
pyramid with a triangular base: 4 corners, 4 faces).  The unknowns are the
electric potentials at the corners ("vertices" or "nodes").  Inside each
tet the potential is assumed to vary linearly -- that is what "P1
piecewise-linear" means.

HOW WE BUILD IT (differs from the paper!)
-----------------------------------------
The paper uses CGAL to make an unstructured mesh that hugs the surfaces.
We don't have CGAL, so:
   1. lay a regular 3-D grid of little boxes over the cylinder,
   2. keep the boxes whose centre is inside the skin radius,
   3. label each box by the tissue at its centre (bone/muscle/fat/skin),
   4. cut every box into 6 tetrahedra (the "Kuhn" split, below).
The curved layer boundaries therefore become little "staircases".  That is
a real approximation (flagged in the summary), but it keeps the code short
and the maths identical to the paper's.
"""
import numpy as np
import config as C


# ----------------------------------------------------------------------
# The Kuhn (a.k.a. Freudenthal) split of a box into 6 tetrahedra.
# ----------------------------------------------------------------------
# Number the 8 corners of a box by their (i, j, k) offsets in {0,1}^3.
# Every tet is a path that starts at corner (0,0,0), takes one unit step
# along x, y and z in some ORDER, and ends at (1,1,1).  There are 3! = 6
# orders -> 6 tets, and they fill the box exactly with no gaps.
# Because every box is cut the same way, faces shared by neighbouring boxes
# are cut identically, so the mesh has no "hanging" vertices (it is
# "conforming" -- the FEM needs this).
_KUHN_PATHS = [
    # each row = the 4 corners visited, as (di, dj, dk)
    [(0,0,0), (1,0,0), (1,1,0), (1,1,1)],   # x then y then z
    [(0,0,0), (1,0,0), (1,0,1), (1,1,1)],   # x, z, y
    [(0,0,0), (0,1,0), (1,1,0), (1,1,1)],   # y, x, z
    [(0,0,0), (0,1,0), (0,1,1), (1,1,1)],   # y, z, x
    [(0,0,0), (0,0,1), (1,0,1), (1,1,1)],   # z, x, y
    [(0,0,0), (0,0,1), (0,1,1), (1,1,1)],   # z, y, x
]


class Mesh:
    """Plain container for everything the FEM needs to know."""
    pass


def build_cylinder_mesh(h_xy=C.H_XY, h_z=C.H_Z, length=C.LENGTH, radii=None, sigma=None):
    """Build the four-layer cylinder mesh.

    radii : optional (R_bone, R_muscle, R_fat, R_skin) in metres; default = config.
    sigma : optional {label: (sx, sy, sz)} conductivity table; default = config.SIGMA.
            (Both exist so the web app can pass per-run values without editing config.)

    Returns a Mesh object with
      .nodes   (n_v, 3)  float  vertex coordinates in metres
      .tets    (n_t, 4)  int    the 4 vertex indices of every tetrahedron
      .labels  (n_t,)    int    tissue label of every tet (C.BONE, ...)
      .sigma   (n_t, 3)  float  diagonal conductivity (sx, sy, sz) per tet
      .vol     (n_t,)    float  volume of every tet (m^3)
      plus the underlying grid info (needed to FIND which tet contains a
      point quickly -- used in Stage 2 for the "basis source" idea).
    """
    r_bone, r_muscle, r_fat, r_skin = (C.R_BONE, C.R_MUSCLE, C.R_FAT, C.R_SKIN) if radii is None else radii

    # ---- 1. regular grid of boxes covering the bounding box of the limb ----
    n_xy = int(round(2 * r_skin / h_xy))        # boxes across the diameter
    n_z = int(round(length / h_z))                # boxes along the length
    x0 = -n_xy * h_xy / 2.0                       # grid starts at -x0 .. +x0
    z0 = -n_z * h_z / 2.0                         # centred on z = 0

    # Node (corner) coordinates along each axis.
    xs = x0 + h_xy * np.arange(n_xy + 1)
    ys = x0 + h_xy * np.arange(n_xy + 1)
    zs = z0 + h_z * np.arange(n_z + 1)

    # ---- 2. decide which boxes are inside the limb, and what tissue ----
    # Box centres:
    xc = 0.5 * (xs[:-1] + xs[1:])
    yc = 0.5 * (ys[:-1] + ys[1:])
    XC, YC = np.meshgrid(xc, yc, indexing="ij")   # (n_xy, n_xy)
    RC = np.hypot(XC, YC)                          # distance from the axis

    # Tissue label of every (i, j) column of boxes (does not depend on z,
    # because the cylinder is uniform along its length).
    box_label = np.full(RC.shape, -1, dtype=np.int8)       # -1 = outside
    box_label[RC <= r_skin] = C.SKIN
    box_label[RC <= r_fat] = C.FAT
    box_label[RC <= r_muscle] = C.MUSCLE
    box_label[RC <= r_bone] = C.BONE
    inside = box_label >= 0                                 # (n_xy, n_xy)

    ii, jj = np.nonzero(inside)                             # kept columns
    n_col = len(ii)
    # col_index[i, j] = which kept column the box column (i, j) is (-1 if
    # outside).  Box number = col * n_z + k, tets of that box = 6*box + (0..5).
    # fem.locate_points() uses this to jump straight to the right tets.
    col_index = np.full(inside.shape, -1, dtype=np.int64)
    col_index[ii, jj] = np.arange(n_col)
    # every kept column is repeated for every layer k along z
    I = np.repeat(ii, n_z)
    J = np.repeat(jj, n_z)
    K = np.tile(np.arange(n_z), n_col)
    lab_box = box_label[I, J]                               # (n_boxes,)

    # ---- 3. node numbering on the full grid, then 6 tets per box ----
    ny1, nz1 = n_xy + 1, n_z + 1

    def node_id(i, j, k):
        return (i * ny1 + j) * nz1 + k

    tets_full = np.empty((len(I) * 6, 4), dtype=np.int64)
    for t, path in enumerate(_KUHN_PATHS):
        for c, (di, dj, dk) in enumerate(path):
            tets_full[t::6, c] = node_id(I + di, J + dj, K + dk)
    labels = np.repeat(lab_box, 6).astype(np.int8)

    # ---- 4. throw away unused grid nodes, renumber the rest 0..n_v-1 ----
    used, inverse = np.unique(tets_full, return_inverse=True)
    tets = inverse.reshape(-1, 4)
    # recover (i, j, k) of each used node, then its coordinates
    ui = used // (ny1 * nz1)
    uj = (used // nz1) % ny1
    uk = used % nz1
    nodes = np.column_stack([xs[ui], ys[uj], zs[uk]])

    # ---- 5. geometric quantities per tet ----
    # Volume of a tet with corners p0..p3 is |det[p1-p0, p2-p0, p3-p0]| / 6.
    p = nodes[tets]                                          # (n_t, 4, 3)
    edges = p[:, 1:, :] - p[:, :1, :]                        # (n_t, 3, 3)
    vol = np.abs(np.linalg.det(edges)) / 6.0

    m = Mesh()
    m.nodes, m.tets, m.labels = nodes, tets, labels
    m.sigma = C.sigma_array(labels, sigma)
    m.vol = vol
    # grid bookkeeping (used later to locate points)
    m.h_xy, m.h_z = h_xy, h_z
    m.n_xy, m.n_z = n_xy, n_z
    m.xs, m.ys, m.zs = xs, ys, zs
    m.box_label = box_label            # (n_xy, n_xy) tissue per column
    m.col_index = col_index            # (n_xy, n_xy) kept-column number or -1
    m.length = length
    m.radii = (r_bone, r_muscle, r_fat, r_skin)
    return m


def summarize(m):
    """Print a short report on the mesh (sizes, volumes per tissue)."""
    print(f"  vertices        : {len(m.nodes):,}")
    print(f"  tetrahedra      : {len(m.tets):,}")
    print(f"  grid spacing    : {m.h_xy*1e3:.1f} mm (x,y), {m.h_z*1e3:.1f} mm (z)")
    print(f"  smallest tet vol: {m.vol.min()*1e9:.4f} mm^3  (must be > 0)")
    tot = m.vol.sum()
    exact = np.pi * C.R_SKIN**2 * m.length
    print(f"  total volume    : {tot*1e6:.2f} cm^3   (exact cylinder: {exact*1e6:.2f} cm^3,"
          f" staircase error {100*(tot-exact)/exact:+.2f}%)")
    for lab, name in enumerate(C.TISSUE_NAMES):
        v = m.vol[m.labels == lab].sum()
        print(f"  {name:<7s} volume : {v*1e6:7.2f} cm^3   ({100*v/tot:5.1f}%)   "
              f"sigma = {C.SIGMA[lab]} S/m")
