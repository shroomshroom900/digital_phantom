"""
stage1_geometry.py -- build the mesh, check it, and draw it.
Run:  python3 stage1_geometry.py
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import config as C
from mesh import build_cylinder_mesh, summarize

t0 = time.perf_counter()
m = build_cylinder_mesh()
print(f"Mesh built in {time.perf_counter() - t0:.2f} s")
summarize(m)

# ---------------- sanity checks (always test your geometry!) --------------
assert (m.vol > 0).all(), "degenerate tetrahedron found"
# The 6 tets of a box must add up to the box volume h_xy*h_xy*h_z:
box_vol = m.h_xy * m.h_xy * m.h_z
assert np.isclose(m.vol[:6].sum(), box_vol), "Kuhn split does not fill the box"
print("  checks passed: all volumes > 0, six tets fill a box exactly")

# ---------------- plotting ------------------------------------------------
cols = ["#e8e2d0", "#c0392b", "#f4d03f", "#f5cba7"]   # bone, muscle, fat, skin
cmap = ListedColormap(cols)
legend = [Patch(color=cols[k], label=f"{C.TISSUE_NAMES[k]}  "
                f"σ={C.SIGMA[k][0]:g}/{C.SIGMA[k][2]:g} S/m") for k in range(4)]

fig = plt.figure(figsize=(14, 4.8))

# (a) cross-section (x-y plane): box_label is exactly what the mesh uses
ax = fig.add_subplot(1, 3, 1)
lab = np.ma.masked_less(m.box_label, 0).astype(float)
ax.pcolormesh(m.xs * 1e2, m.ys * 1e2, lab.T, cmap=cmap, vmin=0, vmax=3)
ax.set_aspect("equal"); ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
ax.set_title("(a) Cross-section, tissue labels")
ax.legend(handles=legend, fontsize=7, loc="lower left",
          bbox_to_anchor=(0.0, -0.32), ncol=2)

# (b) zoom on one corner of the cross-section: show the actual triangles
ax = fig.add_subplot(1, 3, 2)
lo, hi = 1.3e-2, 2.5e-2                  # zoom window in x,y (m)
for i in range(m.n_xy):
    for j in range(m.n_xy):
        if m.box_label[i, j] < 0:
            continue
        x, y = m.xs[i], m.ys[j]
        if not (lo <= x <= hi and lo - 0.3e-2 <= y <= hi):
            continue
        ax.add_patch(plt.Rectangle((x * 1e2, y * 1e2), m.h_xy * 1e2, m.h_xy * 1e2,
                                   fc=cols[m.box_label[i, j]], ec="k", lw=0.4))
        # the Kuhn diagonal (000 -> 110) visible on every z-face
        ax.plot([x * 1e2, (x + m.h_xy) * 1e2], [y * 1e2, (y + m.h_xy) * 1e2],
                "k-", lw=0.3)
ax.set_xlim(lo * 1e2, hi * 1e2); ax.set_ylim((lo - 0.3e-2) * 1e2, hi * 1e2)
ax.set_aspect("equal"); ax.set_xlabel("x (cm)")
ax.set_title("(b) Zoom: 1 mm boxes, diagonal = tet edge\n(note the staircase edges)")

# (c) side view (x-z plane through the axis)
ax = fig.add_subplot(1, 3, 3)
j_mid = m.n_xy // 2                     # a column of boxes next to y = 0
side = np.repeat(m.box_label[:, j_mid][:, None], m.n_z, axis=1).astype(float)
side = np.ma.masked_less(side, 0)
ax.pcolormesh(m.zs * 1e2, m.xs * 1e2, side, cmap=cmap, vmin=0, vmax=3)
ax.set_xlabel("z, along the arm (cm)"); ax.set_ylabel("x (cm)")
ax.set_aspect("equal"); ax.set_title("(c) Longitudinal slice through the axis")

fig.tight_layout()
fig.savefig(os.path.join(C.FIG_DIR, "stage1_geometry.png"), dpi=130, bbox_inches="tight")
print("saved stage1_geometry.png")
