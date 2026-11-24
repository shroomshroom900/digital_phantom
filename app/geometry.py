"""
Display surfaces derived from the computational mesh.

The mesh is a structured grid of boxes; every box column (i, j) carries one tissue
label (mesh.box_label) that is constant along z, and the tetrahedra inherit it.
So the exact tissue boundaries the FEM sees are axis-aligned "staircase" faces:
  * side faces between a column of tissue L and an outer neighbour (higher label
    or outside), running the full length of the mesh,
  * end-cap faces at z = z_min and z = z_max.
We emit those as triangles, so what is drawn is what is solved.

Illustrative overlays (individual muscles) are NOT in the FEM and are flagged so.
"""
import math

import numpy as np

import config as C
import muscle as M   # only for the Stage 4 muscle-belly ellipse (annotation)

TISSUES = [
    dict(label=C.SKIN, key="skin", name="Skin", color="#e9b99a", in_fem=True,
         note="Outer layer of the FEM volume conductor."),
    dict(label=C.FAT, key="fat", name="Subcutaneous fat", color="#f2d16b", in_fem=True,
         note="FEM layer."),
    dict(label=C.MUSCLE, key="muscle", name="Muscle (single homogeneous annulus)", color="#c0392b", in_fem=True,
         note="FEM layer: one transversely isotropic muscle.  Biceps, triceps and brachialis are not "
              "separate regions in the FEM."),
    dict(label=C.BONE, key="bone", name="Bone (idealised humerus)", color="#ece6d6", in_fem=True,
         note="FEM layer: a centred cylinder, not a humerus shape."),
]


def _quad(out, a, b, c, d):
    """Two triangles a-b-c, a-c-d (counter-clockwise seen from outside)."""
    out.extend([*a, *b, *c, *a, *c, *d])


def tissue_surfaces(m):
    lab = m.box_label
    n = lab.shape[0]
    xs, ys = m.xs, m.ys
    z0, z1 = float(m.zs[0]), float(m.zs[-1])
    padded = np.full((n + 2, n + 2), -1, dtype=int)
    padded[1:-1, 1:-1] = lab
    surfaces = []
    for t in TISSUES:
        L = t["label"]
        tri = []
        ii, jj = np.nonzero(lab == L)
        for i, j in zip(ii.tolist(), jj.tolist()):
            xa, xb, ya, yb = xs[i], xs[i + 1], ys[j], ys[j + 1]
            # side faces towards outer neighbours
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = padded[i + 1 + di, j + 1 + dj]
                if nb == -1 or nb > L:
                    if di == 1:
                        _quad(tri, (xb, ya, z0), (xb, yb, z0), (xb, yb, z1), (xb, ya, z1))
                    elif di == -1:
                        _quad(tri, (xa, yb, z0), (xa, ya, z0), (xa, ya, z1), (xa, yb, z1))
                    elif dj == 1:
                        _quad(tri, (xb, yb, z0), (xa, yb, z0), (xa, yb, z1), (xb, yb, z1))
                    else:
                        _quad(tri, (xa, ya, z0), (xb, ya, z0), (xb, ya, z1), (xa, ya, z1))
            # end caps
            _quad(tri, (xa, ya, z1), (xb, ya, z1), (xb, yb, z1), (xa, yb, z1))
            _quad(tri, (xa, ya, z0), (xa, yb, z0), (xb, yb, z0), (xb, ya, z0))
        surfaces.append(dict(key=t["key"], name=t["name"], color=t["color"], in_fem=t["in_fem"],
                             note=t["note"], status="FEM tissue (mesh-derived staircase surface)",
                             n_triangles=len(tri) // 9,
                             volume_m3=float(m.vol[m.labels == L].sum()),
                             positions_m=np.round(np.array(tri, dtype=float), 7).tolist()))
    return surfaces


def _sector_lines(r_in, r_out, th0, th1, z0, z1, n=24):
    """Wireframe of an annular sector prism, as a flat list of segment endpoints."""
    th = np.linspace(th0, th1, n)
    seg = []

    def ring(z):
        outer = [(r_out * math.cos(a), r_out * math.sin(a), z) for a in th]
        inner = [(r_in * math.cos(a), r_in * math.sin(a), z) for a in th[::-1]]
        loop = outer + inner + [outer[0]]
        for p, q in zip(loop[:-1], loop[1:]):
            seg.extend([*p, *q])
    ring(z0); ring(z1)
    for r in (r_in, r_out):
        for a in (th0, th1):
            seg.extend([r * math.cos(a), r * math.sin(a), z0, r * math.cos(a), r * math.sin(a), z1])
    return seg


def illustrations(radii, length):
    """Labelled illustrations that are NOT part of the FEM."""
    r_bone, r_muscle, r_fat, r_skin = radii
    mid = 0.5 * (r_bone + r_muscle)
    z0, z1 = -length / 2, length / 2
    d = math.radians
    status = "ILLUSTRATIVE ONLY - not a separate FEM region (same conductivity as the muscle layer)"
    note = ("Placement is an assumption for orientation (+x = anterior, the side of the default electrode "
            "array); it has no anatomical validation.")
    out = [
        dict(key="biceps", name="Biceps brachii (illustrative)", color="#ff7f7f", status=status, in_fem=False,
             note=note, lines_m=_sector_lines(mid, r_muscle, d(-55), d(55), z0, z1)),
        dict(key="brachialis", name="Brachialis (illustrative)", color="#ffb347", status=status, in_fem=False,
             note=note, lines_m=_sector_lines(r_bone, mid, d(-65), d(65), z0, z1)),
        dict(key="triceps", name="Triceps brachii (illustrative)", color="#7fb2ff", status=status, in_fem=False,
             note=note, lines_m=_sector_lines(r_bone, r_muscle, d(120), d(240), z0, z1)),
    ]
    # Stage 4 muscle belly ellipse (fibre territory of stage4_muscle.py; fixed in muscle.py)
    (cx, cy), (ax, ay) = M.MUSCLE_CENTER, M.MUSCLE_AXES
    th = np.linspace(0, 2 * math.pi, 49)
    seg = []
    for z in (z0, z1, 0.0):
        pts = [(cx + ax * math.cos(a), cy + ay * math.sin(a), z) for a in th]
        for p, q in zip(pts[:-1], pts[1:]):
            seg.extend([*p, *q])
    out.append(dict(key="stage4_belly", name="Stage 4 muscle belly (fibre territory)", color="#b48cff",
                    in_fem=False, status="STAGE 4 - used by stage4_muscle.py only; not a conductivity region "
                                         "and not used by the single-fibre app",
                    note="Ellipse from muscle.MUSCLE_CENTER / MUSCLE_AXES (fixed, does not follow geometry edits).",
                    lines_m=seg))
    return out


def geometry_payload(m):
    return dict(
        coordinate_system=dict(
            axes="right-handed (x, y, z); z = arm long axis = fibre direction; origin on the axis at mid-length; "
                 "theta = atan2(y, x), theta = 0 is +x",
            unit="m"),
        bounds_m=dict(x=[float(m.xs[0]), float(m.xs[-1])], y=[float(m.ys[0]), float(m.ys[-1])],
                      z=[float(m.zs[0]), float(m.zs[-1])]),
        radii_m=dict(zip(["bone", "muscle", "fat", "skin"], [float(r) for r in m.radii])),
        mesh=dict(n_vertices=len(m.nodes), n_tets=len(m.tets), h_xy_m=m.h_xy, h_z_m=m.h_z,
                  element="P1 tetrahedra, 6 per box (Kuhn split)"),
        tissues=tissue_surfaces(m),
        illustrations=illustrations(m.radii, m.length),
    )
